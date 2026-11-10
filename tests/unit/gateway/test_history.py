"""Tests — historique des traitements (« Mon historique »)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from gateway import auth, credits, history, main, models
from gateway.database import SessionLocal

client = TestClient(main.app)

TRANSLATION = {
    "kind": "translation", "title": "reunion-budget.mp3", "lang": "en", "duration_s": 312,
    "content": {"source_text": "Bonjour à tous, parlons du budget confidentiel.",
                "translation": "Hello everyone, let's talk about the confidential budget."},
}


def _account(is_admin: bool = False) -> tuple[int, dict]:
    db = SessionLocal()
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x", is_admin=is_admin)
    db.add(user)
    db.commit()
    uid, email = user.id, user.email
    db.close()
    return uid, {"Authorization": f"Bearer {auth.create_access_token(uid, email, is_admin)}"}


@pytest.fixture
def user():
    return _account()


def _save(headers: dict, **overrides) -> dict:
    r = client.post("/history", headers=headers, json={**TRANSLATION, **overrides})
    assert r.status_code == 201, r.text
    return r.json()


def _titles(headers: dict) -> list[str]:
    return [i["title"] for i in client.get("/history", headers=headers).json()["items"]]


class TestSaveAndRead:
    def test_saved_item_appears_in_the_list_without_its_content(self, user):
        _, headers = user
        saved = _save(headers)
        (item,) = client.get("/history", headers=headers).json()["items"]
        assert item["id"] == saved["id"]
        assert (item["kind"], item["title"], item["lang"], item["duration_s"]) == \
               ("translation", "reunion-budget.mp3", "en", 312)
        assert "content" not in item

    def test_item_content_is_returned_on_demand(self, user):
        _, headers = user
        saved = _save(headers)
        full = client.get(f"/history/{saved['id']}", headers=headers).json()
        assert full["content"] == TRANSLATION["content"]

    def test_list_is_newest_first(self, user):
        _, headers = user
        for title in ("premier", "deuxième", "troisième"):
            _save(headers, title=title)
        assert _titles(headers) == ["troisième", "deuxième", "premier"]

    def test_meeting_summary_can_be_replaced(self, user):
        """Compte-rendu régénéré dans un autre style : même élément, pas un doublon."""
        _, headers = user
        saved = _save(headers, kind="meeting", title="Réunion du 10/11", content={"summary": "v1"})
        r = client.put(f"/history/{saved['id']}", headers=headers,
                       json={"kind": "meeting", "title": "Réunion du 10/11", "content": {"summary": "v2"}})
        assert r.status_code == 200
        assert client.get(f"/history/{saved['id']}", headers=headers).json()["content"] == {"summary": "v2"}
        assert len(_titles(headers)) == 1

    def test_expiry_date_is_announced(self, user):
        _, headers = user
        saved = _save(headers)
        created = datetime.fromisoformat(saved["created_at"])
        assert datetime.fromisoformat(saved["expires_at"]) - created == timedelta(days=history.RETENTION_DAYS)


class TestPrivacy:
    def test_content_is_encrypted_in_the_database(self, user):
        uid, headers = user
        _save(headers)
        db = SessionLocal()
        payload = db.query(models.HistoryItem).filter_by(user_id=uid).one().payload
        db.close()
        assert "budget" not in payload and "confidential" not in payload

    def test_another_account_cannot_read_update_or_delete(self, user):
        _, owner = user
        _, intruder = _account()
        item_id = _save(owner)["id"]
        assert client.get(f"/history/{item_id}", headers=intruder).status_code == 404
        assert client.put(f"/history/{item_id}", headers=intruder, json=TRANSLATION).status_code == 404
        assert client.delete(f"/history/{item_id}", headers=intruder).status_code == 404
        assert _titles(intruder) == []
        assert client.get(f"/history/{item_id}", headers=owner).status_code == 200

    @pytest.mark.parametrize("method, path", [
        ("get", "/history"), ("post", "/history"), ("delete", "/history"),
        ("get", "/history/x"), ("put", "/history/x"), ("delete", "/history/x"),
    ])
    def test_login_is_required(self, method, path):
        assert getattr(client, method)(path).status_code == 401

    def test_items_expire_after_the_retention_period(self, user):
        uid, headers = user
        old, recent = _save(headers, title="ancien"), _save(headers, title="récent")
        db = SessionLocal()
        db.query(models.HistoryItem).filter_by(id=old["id"]).update(
            {"created_at": datetime.now(timezone.utc) - timedelta(days=history.RETENTION_DAYS, minutes=1)})
        db.commit()
        db.close()
        assert _titles(headers) == ["récent"]
        assert client.get(f"/history/{old['id']}", headers=headers).status_code == 404
        assert recent["id"]

    def test_unreadable_item_reports_it_instead_of_crashing(self, user):
        """Clé de chiffrement perdue (restauration sans history.key)."""
        uid, headers = user
        saved = _save(headers)
        db = SessionLocal()
        db.query(models.HistoryItem).filter_by(id=saved["id"]).update({"payload": "illisible"})
        db.commit()
        db.close()
        assert client.get(f"/history/{saved['id']}", headers=headers).status_code == 410
        assert _titles(headers) == ["reunion-budget.mp3"]      # la liste reste consultable


class TestDelete:
    def test_delete_one_item(self, user):
        _, headers = user
        keep, drop = _save(headers, title="garder"), _save(headers, title="supprimer")
        assert client.delete(f"/history/{drop['id']}", headers=headers).json() == {"deleted": 1}
        assert _titles(headers) == ["garder"]
        assert keep["id"]

    def test_delete_everything(self, user):
        _, headers = user
        _, other = _account()
        _save(headers), _save(headers), _save(other)
        assert client.delete("/history", headers=headers).json() == {"deleted": 2}
        assert _titles(headers) == []
        assert len(_titles(other)) == 1                        # pas celui des autres

    def test_deleting_the_account_removes_all_its_data(self):
        uid, headers = _account()
        _save(headers)
        db = SessionLocal()
        credits.apply_tier(db, uid, "pro")
        credits.reserve(db, uid, "processing", 5)
        db.close()

        assert client.delete("/auth/account", headers=headers).status_code == 200

        db = SessionLocal()
        for model in (models.HistoryItem, models.UserCredits, models.CreditTransaction, models.CreditHold):
            assert db.query(model).filter_by(user_id=uid).count() == 0, model.__name__
        db.close()

    def test_admin_deleting_an_account_removes_its_history(self):
        uid, headers = _account()
        _, admin = _account(is_admin=True)
        _save(headers)
        assert client.delete(f"/admin/users/{uid}", headers=admin).status_code == 200
        db = SessionLocal()
        assert db.query(models.HistoryItem).filter_by(user_id=uid).count() == 0
        db.close()


class TestLimits:
    def test_unknown_kind_is_refused(self, user):
        assert client.post("/history", headers=user[1], json={**TRANSLATION, "kind": "autre"}).status_code == 400

    def test_oversized_content_is_refused(self, user, monkeypatch):
        monkeypatch.setattr(history, "MAX_CONTENT_CHARS", 100)
        r = client.post("/history", headers=user[1], json={**TRANSLATION, "content": {"t": "x" * 200}})
        assert r.status_code == 413

    def test_empty_title_is_refused(self, user):
        assert client.post("/history", headers=user[1], json={**TRANSLATION, "title": ""}).status_code == 422

    def test_oldest_items_make_room_beyond_the_cap(self, user, monkeypatch):
        monkeypatch.setattr(history, "MAX_ITEMS_PER_USER", 3)
        _, headers = user
        for n in range(5):
            _save(headers, title=f"n{n}")
        assert _titles(headers) == ["n4", "n3", "n2"]
