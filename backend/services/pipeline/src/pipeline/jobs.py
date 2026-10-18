"""Store in-memory des jobs async pour les gros audios.

Approche minimale : un dict protege par asyncio.Lock, eviction auto apres TTL.
Pas de persistance (restart = jobs perdus) — acceptable pour une instance
pipeline unique et des jobs < 1h. Pour multi-worker ou durabilite, migrer
vers Redis ou Postgres advisory lock + une table jobs.

Cycle de vie :
    pending   -> running   : la coroutine demarre
    running   -> done      : result renseigne, status=done
    running   -> error     : error renseigne, status=error
    (done | error) -> expire au bout de JOB_TTL_SECONDS

Un job inclut user_id pour verifier que seul le proprietaire peut poll.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["pending", "running", "done", "error"]

JOB_TTL_SECONDS         = 60 * 60   # garde les jobs 1h apres completion
JOB_MAX_IN_MEMORY       = 500       # garde-fou OOM : rejette au-dela
JOB_CLEANUP_EVERY       = 60        # nettoyage asynchrone chaque minute


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


class JobStore:
    def __init__(self) -> None:
        self._jobs:   dict[str, Job]    = {}
        self._lock:   asyncio.Lock      = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def create(self, user_id: int | None) -> Job:
        async with self._lock:
            # Eviction si on explose la limite — vire les finis les plus anciens
            if len(self._jobs) >= JOB_MAX_IN_MEMORY:
                finished = sorted(
                    (j for j in self._jobs.values() if j.status in ("done", "error")),
                    key=lambda j: j.ended_at or j.created_at,
                )
                for j in finished[: max(1, len(finished) // 4)]:
                    self._jobs.pop(j.id, None)
            job = Job(id=str(uuid.uuid4()), user_id=user_id)
            self._jobs[job.id] = job
            return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(
        self,
        job_id: str,
        *,
        status:   JobStatus | None       = None,
        progress: int | None             = None,
        result:   dict[str, Any] | None  = None,
        error:    str | None             = None,
    ) -> None:
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

    async def start_cleanup(self) -> None:
        """Lance la boucle d'eviction en arriere-plan (idempotent)."""
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(JOB_CLEANUP_EVERY)
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
            if to_drop:
                print(f"[jobs] evicted {len(to_drop)} jobs > {JOB_TTL_SECONDS}s", flush=True)


# Singleton partage par le process (chaque worker uvicorn a le sien ; en
# single-worker ca suffit)
store = JobStore()
