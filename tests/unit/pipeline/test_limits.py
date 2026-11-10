"""Tests unitaires — limites de charge du pipeline.

Taille des fichiers envoyés, nombre de traitements simultanés sur le serveur
et par compte.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from pipeline import main, uploads


class TestUploadSize:
    def test_file_under_the_limit_is_read(self, monkeypatch):
        monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 1000)
        data = asyncio.run(uploads.read_capped(UploadFile(file=io.BytesIO(b"x" * 1000))))
        assert len(data) == 1000

    def test_file_over_the_limit_is_refused(self, monkeypatch):
        monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 1000)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(uploads.read_capped(UploadFile(file=io.BytesIO(b"x" * 1001))))
        assert exc.value.status_code == 413

    def test_reading_stops_at_the_limit(self, monkeypatch):
        """Un envoi énorme n'est jamais chargé en entier en mémoire."""
        monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 2 * 1024 * 1024)
        source = io.BytesIO(b"x" * (20 * 1024 * 1024))
        with pytest.raises(HTTPException):
            asyncio.run(uploads.read_capped(UploadFile(file=source)))
        assert source.tell() <= 3 * 1024 * 1024

    def test_oversized_request_is_refused_on_its_header(self, monkeypatch):
        """Refus avant authentification et avant lecture du corps."""
        monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 1000)
        monkeypatch.setattr(uploads, "_BODY_OVERHEAD", 0)
        r = TestClient(main.app).post("/process-async", content=b"x" * 5000)
        assert r.status_code == 413
        assert "trop volumineux" in r.json()["detail"]

    def test_normal_request_passes_the_size_check(self):
        r = TestClient(main.app).post("/process-async", files={"file": ("a.mp3", b"abc", "audio/mpeg")})
        assert r.status_code == 401   # arrive jusqu'à l'authentification


@pytest.fixture
def no_jobs(monkeypatch):
    monkeypatch.setattr(main, "_user_jobs", {})
    monkeypatch.setattr(main, "MAX_JOBS_PER_USER", 2)
    monkeypatch.setattr(main, "_jobs_total", 0)


class TestJobsPerUser:
    def test_third_job_of_the_same_account_is_refused(self, no_jobs):
        main._enter_user(7)
        main._enter_user(7)
        with pytest.raises(HTTPException) as exc:
            main._enter_user(7)
        assert exc.value.status_code == 429

    def test_finishing_a_job_frees_a_slot(self, no_jobs):
        main._enter_user(7)
        main._enter_user(7)
        main._leave_user(7)
        main._enter_user(7)
        assert main._user_jobs == {7: 2}

    def test_accounts_do_not_block_each_other(self, no_jobs):
        main._enter_user(7)
        main._enter_user(7)
        main._enter_user(8)
        assert main._user_jobs == {7: 2, 8: 1}

    def test_counter_is_removed_when_idle(self, no_jobs):
        main._enter_user(7)
        main._leave_user(7)
        main._leave_user(7)   # un appel en trop ne rend pas le compteur négatif
        assert main._user_jobs == {}

    def test_internal_calls_are_not_counted(self, no_jobs):
        for _ in range(10):
            main._enter_user(None)
        assert main._user_jobs == {}

    def test_failed_job_frees_the_account(self, no_jobs, monkeypatch):
        async def boom(state):
            raise HTTPException(status_code=422, detail="audio illisible")

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(main, "_run_pipeline", boom)
        monkeypatch.setattr(main, "_release_credits", noop)
        main._enter_user(7)

        async def run():
            job = await main._jobs.store.create(user_id=7)
            await main._execute_job(
                job_id=job.id, audio_bytes=b"x", filename="a.mp3", target_lang="en",
                llm_model="openai/gpt-4o-mini", prompt_version="v1.1", whisper_model="small",
                diarize=False, hold_id="h-1", user_id=7,
            )
            return await main._jobs.store.get(job.id)

        job = asyncio.run(run())
        assert job.status == "error" and job.error == "audio illisible"
        assert main._user_jobs == {}


class TestServerSlots:
    def test_jobs_beyond_the_limit_wait_their_turn(self, monkeypatch):
        running = peak = 0

        class FakeChain:
            async def ainvoke(self, state):
                nonlocal running, peak
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.01)
                running -= 1
                return state

        monkeypatch.setattr(main, "pipeline_chain", FakeChain())

        async def run():
            monkeypatch.setattr(main, "_job_slots", asyncio.Semaphore(3))
            return await asyncio.gather(*(main._run_pipeline({"n": n}) for n in range(10)))

        results = asyncio.run(run())
        assert peak == 3
        assert [r["n"] for r in results] == list(range(10))


class TestServerSaturation:
    def test_full_queue_refuses_new_jobs_cleanly(self, no_jobs, monkeypatch):
        """Un job en attente garde son audio en mémoire : la file est bornée."""
        monkeypatch.setattr(main, "MAX_CONCURRENT_JOBS", 2)
        monkeypatch.setattr(main, "MAX_QUEUED_JOBS", 3)
        for account in range(5):                 # 2 en cours + 3 en attente
            main._enter_user(account)
        with pytest.raises(HTTPException) as exc:
            main._enter_user(99)
        assert exc.value.status_code == 503
        assert exc.value.headers["Retry-After"] == "120"
        assert 99 not in main._user_jobs

    def test_a_finished_job_makes_room(self, no_jobs, monkeypatch):
        monkeypatch.setattr(main, "MAX_CONCURRENT_JOBS", 1)
        monkeypatch.setattr(main, "MAX_QUEUED_JOBS", 0)
        main._enter_user(1)
        main._leave_user(1)
        main._enter_user(2)
        assert main._jobs_total == 1

    def test_internal_calls_count_towards_the_server_total(self, no_jobs, monkeypatch):
        monkeypatch.setattr(main, "MAX_CONCURRENT_JOBS", 1)
        monkeypatch.setattr(main, "MAX_QUEUED_JOBS", 0)
        main._enter_user(None)
        with pytest.raises(HTTPException):
            main._enter_user(None)

    def test_refused_account_job_is_not_counted(self, no_jobs):
        main._enter_user(7)
        main._enter_user(7)
        with pytest.raises(HTTPException):
            main._enter_user(7)                  # 429
        assert main._jobs_total == 2
