"""Tests unitaires — LLM fallback + retry (fix c7ae5fb).

Capture le fix contre les 500 opaques quand un provider LLM tombe (ex. modèle
Groq déprécié en août 2026). Vérifie que :
  - _FALLBACKS est bien chargé depuis LLM_FALLBACKS avec un default sûr
  - call_llm() passe num_retries=2 et fallbacks=[...] à litellm.completion
  - le modèle demandé n'est jamais réutilisé comme son propre fallback
  - une exception litellm remonte en HTTPException(502) propre, pas un 500

Tests basés sur mock de litellm.completion (pas d'appel réseau).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from llm import main as llm_main


def _mock_response(
    text: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Fabrique un objet response litellm minimal."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


class TestFallbacksConfig:
    """_FALLBACKS est chargé au module-load depuis LLM_FALLBACKS."""

    def test_default_includes_openai_and_anthropic(self):
        """Sans override, la liste par défaut couvre au moins 2 providers."""
        assert "openai/gpt-4o-mini" in llm_main._FALLBACKS
        assert any("anthropic" in m for m in llm_main._FALLBACKS)

    def test_no_empty_or_whitespace_entries(self):
        """Chaque entrée est trimmée, pas de string vide."""
        assert all(m.strip() != "" for m in llm_main._FALLBACKS)
        assert all(m == m.strip() for m in llm_main._FALLBACKS)

    def test_default_model_not_deprecated_groq(self):
        """Régression : le default LLM_MODEL ne doit plus être le Groq déprécié.

        Historique : avant le fix c7ae5fb, DEFAULT_MODEL fallback hardcodé
        était `groq/openai/gpt-oss-20b`, modèle déprécié par Groq → 500 en prod
        si LLM_MODEL n'était pas set dans .env.
        """
        assert llm_main.DEFAULT_MODEL != "groq/openai/gpt-oss-20b"


class TestCallLLMHappyPath:
    """call_llm() route correctement les paramètres vers litellm."""

    def test_returns_translation_and_usage(self):
        with patch.object(
            llm_main.litellm,
            "completion",
            return_value=_mock_response("Hello world", 100, 20),
        ) as mock:
            translation, latency_ms, usage = llm_main.call_llm(
                prompt="bonjour", model="openai/gpt-4o-mini", timeout=10
            )
            assert translation == "Hello world"
            assert latency_ms >= 0
            assert usage["prompt_tokens"] == 100
            assert usage["completion_tokens"] == 20
            assert usage["total_tokens"] == 120
            mock.assert_called_once()

    def test_passes_num_retries_2(self):
        """Régression : num_retries doit toujours être passé pour couvrir les
        erreurs transitoires (rate limit court, timeout ponctuel)."""
        with patch.object(llm_main.litellm, "completion", return_value=_mock_response()):
            llm_main.call_llm(prompt="test", model="openai/gpt-4o-mini")
            call_kwargs = llm_main.litellm.completion.call_args.kwargs
            assert call_kwargs.get("num_retries") == 2

    def test_passes_fallbacks_list(self):
        """Régression : la liste de fallbacks doit être passée à litellm."""
        with patch.object(llm_main.litellm, "completion", return_value=_mock_response()):
            llm_main.call_llm(prompt="test", model="openai/gpt-4o-mini")
            call_kwargs = llm_main.litellm.completion.call_args.kwargs
            assert "fallbacks" in call_kwargs
            # Peut être None si _FALLBACKS ne contient QUE le modèle demandé,
            # sinon doit être une liste non vide
            fb = call_kwargs["fallbacks"]
            assert fb is None or (isinstance(fb, list) and len(fb) > 0)


class TestCallLLMFallbackExclusion:
    """Le modèle demandé n'apparaît jamais dans sa propre liste de fallbacks."""

    def test_current_model_removed_from_fallbacks(self, monkeypatch):
        """Si model=openai/gpt-4o-mini et _FALLBACKS inclut ce même modèle,
        il doit être filtré (sinon retry sur soi-même = boucle inutile)."""
        monkeypatch.setattr(
            llm_main,
            "_FALLBACKS",
            ["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku-20241022"],
        )
        with patch.object(llm_main.litellm, "completion", return_value=_mock_response()):
            llm_main.call_llm(prompt="test", model="openai/gpt-4o-mini")
            fb = llm_main.litellm.completion.call_args.kwargs["fallbacks"]
            assert fb == ["anthropic/claude-3-5-haiku-20241022"], (
                "Le modèle demandé doit être exclu de sa propre liste de fallbacks"
            )


class TestCallLLMErrorHandling:
    """Une exception LiteLLM (tous providers down) doit remonter en 502 propre,
    pas en 500 opaque qui casse le user."""

    def test_upstream_exception_becomes_http_502(self):
        with patch.object(
            llm_main.litellm,
            "completion",
            side_effect=Exception("all providers down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                llm_main.call_llm(prompt="test", model="openai/gpt-4o-mini")
            assert exc_info.value.status_code == 502

    def test_upstream_exception_detail_names_type(self):
        """Le detail doit inclure le type d'exception pour aider le debug."""
        with patch.object(
            llm_main.litellm,
            "completion",
            side_effect=ValueError("bad request"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                llm_main.call_llm(prompt="test", model="openai/gpt-4o-mini")
            detail = exc_info.value.detail
            assert "LLM upstream unavailable" in detail
            assert "ValueError" in detail

    def test_upstream_exception_detail_truncated(self):
        """Le message d'erreur est tronqué à 200 chars pour ne pas leak
        d'infos sensibles ou spammer les logs client."""
        long_msg = "x" * 500
        with patch.object(
            llm_main.litellm,
            "completion",
            side_effect=RuntimeError(long_msg),
        ):
            with pytest.raises(HTTPException) as exc_info:
                llm_main.call_llm(prompt="test", model="openai/gpt-4o-mini")
            # Le detail contient le préfixe + le type + max 200 chars du message
            # Cadre large pour absorber les évolutions de format
            assert len(exc_info.value.detail) < 400
