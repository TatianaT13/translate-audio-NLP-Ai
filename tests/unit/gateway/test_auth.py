"""Tests unitaires de la chaîne de sécurité auth (JWT + password hashing).

Couvre :
  - hash_password / verify_password  (bcrypt)
  - create_access_token / decode_access_token (JWT HS256)
  - make_token_pair / hash_token (refresh & reset tokens SHA-256)
  - Résistance : tampering, expiration, mauvaise signature, type mismatch
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from gateway.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    SECRET_KEY,
    create_access_token,
    decode_access_token,
    hash_password,
    hash_token,
    make_token_pair,
    verify_password,
)


# ── Password hashing (bcrypt) ────────────────────────────────────────────────

class TestPasswordHashing:
    """bcrypt : salt aléatoire, vérification stricte, immuabilité du secret."""

    def test_hash_password_produces_different_hash_each_time(self):
        """Le salt aléatoire garantit deux hashes différents pour le même mdp."""
        h1 = hash_password("Test123!")
        h2 = hash_password("Test123!")
        assert h1 != h2

    def test_hash_password_never_contains_the_plaintext(self):
        h = hash_password("MonMotDePasseTresSecret2026!")
        assert "MonMotDePasseTresSecret2026" not in h

    def test_verify_password_accepts_correct_password(self):
        h = hash_password("Test123!")
        assert verify_password("Test123!", h) is True

    def test_verify_password_rejects_wrong_password(self):
        h = hash_password("Test123!")
        assert verify_password("WrongPassword", h) is False

    def test_verify_password_rejects_empty_string(self):
        h = hash_password("Test123!")
        assert verify_password("", h) is False

    def test_verify_password_case_sensitive(self):
        h = hash_password("Test123!")
        assert verify_password("test123!", h) is False

    def test_hash_supports_unicode(self):
        h = hash_password("Mötdepássé_ñ_日本語")
        assert verify_password("Mötdepássé_ñ_日本語", h) is True


# ── JWT access tokens ────────────────────────────────────────────────────────

class TestAccessTokenRoundtrip:
    """Un token créé doit se décoder avec les bonnes claims."""

    def test_encode_decode_preserves_user_id(self):
        t = create_access_token(user_id=42, email="a@b.c")
        payload = decode_access_token(t)
        assert payload is not None
        assert payload["sub"] == "42"

    def test_encode_decode_preserves_email(self):
        t = create_access_token(user_id=1, email="tatiana@example.com")
        payload = decode_access_token(t)
        assert payload["email"] == "tatiana@example.com"

    def test_admin_flag_preserved(self):
        t = create_access_token(user_id=1, email="a@b.c", is_admin=True)
        assert decode_access_token(t)["is_admin"] is True

    def test_admin_defaults_to_false(self):
        t = create_access_token(user_id=1, email="a@b.c")
        assert decode_access_token(t)["is_admin"] is False

    def test_token_has_type_access_claim(self):
        t = create_access_token(user_id=1, email="a@b.c")
        assert decode_access_token(t)["type"] == "access"


# ── JWT security : rejet des tokens invalides ────────────────────────────────

class TestAccessTokenSecurity:
    """Résistance aux tokens forgés / expirés / altérés."""

    def test_expired_token_is_rejected(self):
        """Un token dont l'exp est dans le passé doit être refusé."""
        expired = jwt.encode(
            {
                "sub": "1", "email": "a@b.c", "is_admin": False, "type": "access",
                "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            },
            SECRET_KEY, algorithm=ALGORITHM,
        )
        assert decode_access_token(expired) is None

    def test_wrong_signature_is_rejected(self):
        """Un token signé avec une autre clé doit être refusé."""
        forged = jwt.encode(
            {
                "sub": "1", "email": "a@b.c", "is_admin": True, "type": "access",
                "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            },
            "attacker-secret-key",
            algorithm=ALGORITHM,
        )
        assert decode_access_token(forged) is None

    def test_wrong_algorithm_is_rejected(self):
        """Un token forgé avec HS512 doit être rejeté (config accepte HS256 uniquement)."""
        forged = jwt.encode(
            {"sub": "1", "email": "a@b.c", "type": "access",
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            SECRET_KEY, algorithm="HS512",
        )
        assert decode_access_token(forged) is None

    def test_refresh_type_token_rejected_as_access(self):
        """Un refresh token forgé ne doit pas être accepté comme access."""
        refresh_like = jwt.encode(
            {
                "sub": "1", "email": "a@b.c", "type": "refresh",
                "exp": datetime.now(timezone.utc) + timedelta(days=7),
            },
            SECRET_KEY, algorithm=ALGORITHM,
        )
        assert decode_access_token(refresh_like) is None

    def test_garbage_token_is_rejected(self):
        assert decode_access_token("not.a.valid.jwt") is None
        assert decode_access_token("") is None
        assert decode_access_token("Bearer abc") is None

    def test_tampered_payload_is_rejected(self):
        """Modifier le payload sans re-signer casse la signature."""
        t = create_access_token(user_id=1, email="a@b.c", is_admin=False)
        # On modifie un char AU MILIEU du payload (partie 2 du JWT).
        # NB : pas la signature (partie 3) — ses derniers chars base64url
        # contiennent des bits ignorés (256 bits ne rentrent pas pile dans
        # 43 chars × 6 bits), donc changer le dernier char peut décoder
        # aux mêmes bytes → faux négatif selon le SECRET_KEY.
        parts = t.split(".")
        payload = parts[1]
        i = len(payload) // 2  # milieu, jamais un bit ignoré
        tampered_payload = payload[:i] + ("X" if payload[i] != "X" else "Y") + payload[i + 1:]
        tampered = f"{parts[0]}.{tampered_payload}.{parts[2]}"
        assert decode_access_token(tampered) is None

    def test_expiration_is_15_minutes(self):
        t = create_access_token(user_id=1, email="a@b.c")
        payload = jwt.decode(t, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        assert abs((exp - expected).total_seconds()) < 5  # tolérance 5s


# ── Refresh / reset token pairs (SHA-256) ────────────────────────────────────

class TestTokenPair:
    """Les tokens refresh/reset : raw envoyé au client, hash stocké en DB."""

    def test_make_token_pair_returns_distinct_raw_and_hash(self):
        raw, hashed = make_token_pair()
        assert raw != hashed
        assert len(raw) >= 40      # secrets.token_urlsafe(64) → ~86 chars
        assert len(hashed) == 64   # SHA-256 hex = 64 chars

    def test_make_token_pair_is_unique_each_call(self):
        pairs = {make_token_pair()[0] for _ in range(100)}
        assert len(pairs) == 100  # 100 appels = 100 raw tokens uniques

    def test_hash_token_is_deterministic(self):
        """Le même raw hashé deux fois donne le même hash (permet lookup en DB)."""
        assert hash_token("abc") == hash_token("abc")

    def test_hash_token_matches_make_token_pair_hash(self):
        """hash_token(raw) doit reproduire le hash retourné par make_token_pair."""
        raw, expected_hash = make_token_pair()
        assert hash_token(raw) == expected_hash

    def test_hash_token_different_inputs_different_hash(self):
        assert hash_token("token_a") != hash_token("token_b")

    def test_raw_token_never_reveals_itself_via_hash(self):
        """Le hash ne doit pas contenir le raw en clair (defense-in-depth)."""
        raw, hashed = make_token_pair()
        assert raw not in hashed
