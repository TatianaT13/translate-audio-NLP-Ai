"""Tests unitaires — jobs async persistés sur disque (survie à un redéploiement)."""

from __future__ import annotations

import asyncio
import time

from pipeline import jobs
from pipeline.jobs import JobStore

RESULT = {"translation": "Hello everyone.", "audio_b64": "QUJD" * 1000}


def _finished_job(directory) -> str:
    async def run():
        store = JobStore(directory)
        job = await store.create(user_id=7, hold_id="h-1")
        await store.update(job.id, status="running", progress=15)
        await store.update(job.id, status="done", progress=100, result=RESULT)
        return job.id
    return asyncio.run(run())


class TestSurvivesRestart:
    def test_finished_job_is_still_readable_after_restart(self, tmp_path):
        job_id = _finished_job(tmp_path)

        restarted = JobStore(tmp_path)          # nouveau process
        job = asyncio.run(restarted.get(job_id))
        assert job.status == "done" and job.user_id == 7
        assert job.result == RESULT
        assert restarted.interrupted_holds == []

    def test_job_running_at_restart_becomes_a_clear_error(self, tmp_path):
        async def run():
            store = JobStore(tmp_path)
            job = await store.create(user_id=7, hold_id="h-42")
            await store.update(job.id, status="running", progress=15)
            return job.id
        job_id = asyncio.run(run())

        restarted = JobStore(tmp_path)
        job = asyncio.run(restarted.get(job_id))
        assert job.status == "error" and job.error == jobs.INTERRUPTED
        assert restarted.interrupted_holds == ["h-42"]   # crédits réservés à rendre

        # Deuxième redémarrage : le job est déjà en erreur, rien à rendre de nouveau
        assert JobStore(tmp_path).interrupted_holds == []

    def test_pending_job_without_hold_is_also_closed(self, tmp_path):
        job_id = asyncio.run(JobStore(tmp_path).create(user_id=None)).id
        restarted = JobStore(tmp_path)
        assert asyncio.run(restarted.get(job_id)).status == "error"
        assert restarted.interrupted_holds == []

    def test_expired_jobs_are_not_reloaded(self, tmp_path, monkeypatch):
        job_id = _finished_job(tmp_path)
        monkeypatch.setattr(jobs, "JOB_TTL_SECONDS", -1)
        assert asyncio.run(JobStore(tmp_path).get(job_id)) is None
        assert list(tmp_path.iterdir()) == []

    def test_corrupted_file_is_ignored(self, tmp_path):
        (tmp_path / "abc.meta.json").write_text("{pas du json")
        assert asyncio.run(JobStore(tmp_path).get("abc")) is None


class TestStorage:
    def test_result_is_kept_on_disk_not_in_memory(self, tmp_path):
        async def run():
            store = JobStore(tmp_path)
            job = await store.create(user_id=7)
            await store.update(job.id, status="done", result=RESULT)
            return store, job.id
        store, job_id = asyncio.run(run())
        assert store._jobs[job_id].result is None
        assert (tmp_path / f"{job_id}.result.json").exists()
        assert asyncio.run(store.get(job_id)).result == RESULT

    def test_files_are_private(self, tmp_path):
        job_id = _finished_job(tmp_path)
        for name in (f"{job_id}.meta.json", f"{job_id}.result.json"):
            assert (tmp_path / name).stat().st_mode & 0o777 == 0o600

    def test_expired_job_files_are_deleted(self, tmp_path, monkeypatch):
        async def run():
            store = JobStore(tmp_path)
            job = await store.create(user_id=7)
            await store.update(job.id, status="done", result=RESULT)
            store._jobs[job.id].ended_at = time.time() - jobs.JOB_TTL_SECONDS - 1
            return await store._evict_expired(), await store.get(job.id)
        dropped, job = asyncio.run(run())
        assert dropped == 1 and job is None
        assert list(tmp_path.iterdir()) == []

    def test_without_directory_everything_stays_in_memory(self):
        async def run():
            store = JobStore()
            job = await store.create(user_id=7)
            await store.update(job.id, status="done", result=RESULT)
            return await store.get(job.id)
        assert asyncio.run(run()).result == RESULT

    def test_unusable_directory_falls_back_to_memory(self, tmp_path):
        blocker = tmp_path / "fichier"
        blocker.write_text("x")
        store = JobStore(blocker / "jobs")     # impossible à créer
        assert store._dir is None
        assert asyncio.run(store.create(user_id=1)).status == "pending"


class TestStartup:
    def test_credits_of_interrupted_jobs_are_released(self, monkeypatch):
        from fastapi.testclient import TestClient
        from pipeline import main

        released = []

        async def fake_release(hold_id):
            released.append(hold_id)

        monkeypatch.setattr(main, "_release_credits", fake_release)
        monkeypatch.setattr(main._jobs, "store", JobStore())
        main._jobs.store.interrupted_holds = ["h-1", "h-2"]

        with TestClient(main.app):      # déclenche le démarrage de l'application
            pass
        assert released == ["h-1", "h-2"]
        assert main._jobs.store.interrupted_holds == []
