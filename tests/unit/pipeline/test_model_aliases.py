"""Tests unitaires — alias serveur des modèles LLM obsolètes / instables.

Capture le fix `d3ff283` : le pipeline redirige automatiquement les modèles
Groq problématiques vers openai/gpt-4o-mini pour protéger les visiteurs dont
le browser garde d'anciens modèles en localStorage.
"""

from __future__ import annotations

from pipeline.main import _LEGACY_MODEL_ALIASES


class TestLegacyModelAliases:
    """Le pipeline redirige les modèles Groq obsolètes vers OpenAI."""

    def test_all_aliases_point_to_stable_openai_model(self):
        """Tous les alias doivent pointer vers un modèle OpenAI stable."""
        for legacy, alias in _LEGACY_MODEL_ALIASES.items():
            assert alias == "openai/gpt-4o-mini", (
                f"{legacy} devrait alias vers openai/gpt-4o-mini, pas {alias}"
            )

    def test_deprecated_groq_llama_31_8b(self):
        """llama-3.1-8b-instant déprécié par Groq (août 2026) → OpenAI."""
        assert "groq/llama-3.1-8b-instant" in _LEGACY_MODEL_ALIASES

    def test_unstable_gpt_oss_20b_aliased(self):
        """gpt-oss-20b déclenche 60% de faux positifs prompt_guard → OpenAI."""
        assert _LEGACY_MODEL_ALIASES["groq/openai/gpt-oss-20b"] == "openai/gpt-4o-mini"

    def test_all_deprecated_groq_llama_variants_covered(self):
        """Les 3 variantes Llama Groq dépréciées doivent être toutes couvertes."""
        deprecated_llamas = [
            "groq/llama-3.1-8b-instant",
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-70b-versatile",
        ]
        for m in deprecated_llamas:
            assert m in _LEGACY_MODEL_ALIASES, f"{m} pas dans les alias"

    def test_openai_models_not_aliased(self):
        """Les modèles OpenAI valides ne doivent PAS être aliasés."""
        assert "openai/gpt-4o-mini" not in _LEGACY_MODEL_ALIASES
        assert "openai/gpt-4o" not in _LEGACY_MODEL_ALIASES

    def test_no_infinite_alias_loop(self):
        """Aucun alias ne doit pointer vers un modèle lui-même aliasé (loop)."""
        for legacy, alias in _LEGACY_MODEL_ALIASES.items():
            assert alias not in _LEGACY_MODEL_ALIASES, (
                f"Boucle : {legacy} → {alias} qui est aussi un alias"
            )
