"""Store des jobs async pour les gros audios.

Etat en memoire, double sur disque quand JOBS_DIR est defini : un redemarrage
du service (deploiement) ne fait plus disparaitre les jobs.
  - un job termine reste recuperable par le client jusqu'a son expiration ;
  - un job en cours au moment du redemarrage est perdu (son execution vivait
    dans le process) : il passe en erreur avec un message clair, et sa
    reservation de credits est rendue (voir interrupted_holds).

Sur disque, par job : <id>.meta.json (etat, leger) et <id>.result.json (le
resultat, qui contient l'audio en base64 — il n'est donc pas garde en memoire).

Cycle de vie :
    pending   -> running   : la coroutine demarre
    running   -> done      : result renseigne, status=done
    running   -> error     : error renseigne, status=error
    (done | error) -> expire au bout de JOB_TTL_SECONDS

Un job inclut user_id pour verifier que seul le proprietaire peut poll.
Instance pipeline unique : pour du multi-worker, migrer vers Redis ou Postgres.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["pending", "running", "done", "error"]

JOB_TTL_SECONDS         = 60 * 60   # garde les jobs 1h apres completion
JOB_MAX_IN_MEMORY       = 500       # garde-fou OOM : rejette au-dela
JOB_CLEANUP_EVERY       = 60        # nettoyage asynchrone chaque minute

INTERRUPTED = (
    "Traitement interrompu par un redémarrage du service. "
    "Aucun crédit n'a été débité : tu peux le relancer."
)


@dataclass
class Job:
    id:         str
    user_id:    int | None
    status:     JobStatus              = "pending"
    progress:   int                    = 0          # 0..100, estime
    result:     dict[str, Any] | None  = None
    error:      str | None             = None
    created_at: float                  = field(default_factory=time.time)
    started_at: float | None           = None
    ended_at:   float | None           = None
    hold_id:    str | None             = None       # reservation de credits du job


class JobStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self._jobs:   dict[str, Job]    = {}
        self._lock:   asyncio.Lock      = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._dir: Path | None = None
        # Reservations des jobs interrompus par le redemarrage, a rendre a la Gateway
        self.interrupted_holds: list[str] = []
        if directory:
            try:
                self._dir = Path(directory)
                self._dir.mkdir(parents=True, exist_ok=True)
                self._load()
            except OSError as e:
                print(f"[jobs] {directory} inutilisable ({e}) : jobs en memoire seulement", flush=True)
                self._dir = None

    # ── Disque ───────────────────────────────────────────────────────────────
    def _meta_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.meta.json"      # type: ignore[operator]

    def _result_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.result.json"    # type: ignore[operator]

    def _write_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.chmod(tmp, 0o600)        # transcriptions et audio des utilisateurs
        os.replace(tmp, path)       # ecriture atomique : jamais de fichier a moitie ecrit

    def _save_meta(self, job: Job) -> None:
        if self._dir is None:
            return
        meta = asdict(job)
        meta.pop("result")
        try:
            self._write_json(self._meta_path(job.id), meta)
        except OSError as e:
            print(f"[jobs] ecriture de {job.id} impossible : {e}", flush=True)

    def _delete_files(self, job_id: str) -> None:
        if self._dir is None:
            return
        for path in (self._meta_path(job_id), self._result_path(job_id)):
            path.unlink(missing_ok=True)

    def _load(self) -> None:
        """Recharge les jobs du disque au demarrage."""
        now = time.time()
        for path in self._dir.glob("*.meta.json"):  # type: ignore[union-attr]
            try:
                job = Job(result=None, **json.loads(path.read_text()))
            except (ValueError, TypeError, OSError):
                path.unlink(missing_ok=True)
                continue
            if job.status in ("pending", "running"):
                # Son execution vivait dans l'ancien process : elle n'existe plus.
                if job.hold_id:
                    self.interrupted_holds.append(job.hold_id)
                job.status, job.error, job.ended_at = "error", INTERRUPTED, now
                self._save_meta(job)
            if job.ended_at is not None and now - job.ended_at > JOB_TTL_SECONDS:
                self._delete_files(job.id)
                continue
            self._jobs[job.id] = job
        if self._jobs:
            print(f"[jobs] {len(self._jobs)} job(s) recharges, "
                  f"{len(self.interrupted_holds)} interrompu(s) par le redemarrage", flush=True)

    # ── API ──────────────────────────────────────────────────────────────────
    async def create(self, user_id: int | None, hold_id: str | None = None) -> Job:
        async with self._lock:
            # Eviction si on explose la limite — vire les finis les plus anciens
            if len(self._jobs) >= JOB_MAX_IN_MEMORY:
                finished = sorted(
                    (j for j in self._jobs.values() if j.status in ("done", "error")),
                    key=lambda j: j.ended_at or j.created_at,
                )
                for j in finished[: max(1, len(finished) // 4)]:
                    self._jobs.pop(j.id, None)
                    self._delete_files(j.id)
            job = Job(id=str(uuid.uuid4()), user_id=user_id, hold_id=hold_id)
            self._jobs[job.id] = job
            self._save_meta(job)
            return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "done" or job.result is not None or self._dir is None:
                return job
            path = self._result_path(job_id)
        # Resultat sur disque : lu hors du verrou (plusieurs Mo avec l'audio)
        try:
            result = json.loads(await asyncio.to_thread(path.read_text))
        except (OSError, ValueError):
            return replace(job, status="error", error="Résultat introuvable. Relance le traitement.")
        return replace(job, result=result)

    async def update(
        self,
        job_id: str,
        *,
        status:   JobStatus | None       = None,
        progress: int | None             = None,
        result:   dict[str, Any] | None  = None,
        error:    str | None             = None,
    ) -> None:
        if result is not None and self._dir is not None:
            # Ecrit avant de publier le statut "done" : un client qui poll ne
            # voit jamais un job termine sans son resultat.
            try:
                await asyncio.to_thread(self._write_json, self._result_path(job_id), result)
                result = None   # sur disque, pas en memoire
            except OSError as e:
                print(f"[jobs] resultat de {job_id} garde en memoire ({e})", flush=True)
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status is not None:
                job.status = status
                if status == "running" and job.started_at is None:
                    job.started_at = time.time()
                if status in ("done", "error"):
                    job.ended_at = time.time()
            if progress is not None:
                job.progress = max(0, min(100, progress))
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if status is not None:
                self._save_meta(job)

    async def start_cleanup(self) -> None:
        """Lance la boucle d'eviction en arriere-plan (idempotent)."""
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _evict_expired(self) -> int:
        now = time.time()
        async with self._lock:
            to_drop = [
                j.id for j in self._jobs.values()
                if j.status in ("done", "error")
                and j.ended_at is not None
                and (now - j.ended_at) > JOB_TTL_SECONDS
            ]
            for jid in to_drop:
                self._jobs.pop(jid, None)
                self._delete_files(jid)
        return len(to_drop)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(JOB_CLEANUP_EVERY)
            dropped = await self._evict_expired()
            if dropped:
                print(f"[jobs] evicted {dropped} jobs > {JOB_TTL_SECONDS}s", flush=True)


# Singleton partage par le process (chaque worker uvicorn a le sien ; en
# single-worker ca suffit)
store = JobStore(os.getenv("JOBS_DIR") or None)
