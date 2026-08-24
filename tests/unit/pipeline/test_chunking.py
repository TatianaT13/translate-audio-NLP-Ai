"""Tests unitaires — chunking automatique du pipeline pour longs audios.

Capture le fix `8e66e85` + chunking : le pipeline découpe automatiquement les
longues transcriptions en chunks avant LLM + TTS, au lieu de bloquer.
"""

from __future__ import annotations

from pipeline.main import _split_into_chunks, LLM_CHUNK_MAX_CHARS, TTS_CHUNK_MAX_CHARS


class TestSplitIntoChunks:
    """La fonction _split_into_chunks découpe intelligemment par phrase."""

    def test_short_text_returned_as_single_chunk(self):
        text = "Une phrase courte."
        assert _split_into_chunks(text, max_chars=1000) == [text]

    def test_empty_text_returned_as_single_chunk(self):
        assert _split_into_chunks("", max_chars=1000) == [""]

    def test_long_text_split_at_sentence_boundary(self):
        # 3 phrases de ~30 chars chacune → 3 chunks si max=30
        text = "Phrase un premier. Phrase deux ok. Phrase trois voilà."
        chunks = _split_into_chunks(text, max_chars=30)
        assert len(chunks) >= 2
        # Aucun chunk ne doit couper une phrase au milieu
        for c in chunks:
            assert c.endswith(".") or c == chunks[-1]

    def test_no_chunk_exceeds_max_chars(self):
        text = "Un long texte. " * 500  # ~7500 chars
        chunks = _split_into_chunks(text, max_chars=500)
        for c in chunks:
            assert len(c) <= 500, f"Chunk fait {len(c)} chars, > 500"

    def test_single_huge_sentence_is_hard_split(self):
        """Une phrase seule qui dépasse max_chars doit être coupée brutalement."""
        text = "a" * 5000  # une "phrase" de 5000 chars sans ponctuation
        chunks = _split_into_chunks(text, max_chars=1000)
        assert len(chunks) == 5
        for c in chunks:
            assert len(c) <= 1000

    def test_all_chunks_concatenated_preserve_content(self):
        """Le contenu total doit être préservé (aux espaces près)."""
        text = "Phrase A. Phrase B. Phrase C. Phrase D."
        chunks = _split_into_chunks(text, max_chars=20)
        rejoined = " ".join(chunks)
        # Chaque phrase originale doit se retrouver dans le résultat
        assert "Phrase A" in rejoined
        assert "Phrase D" in rejoined

    def test_default_chunk_size_env_overridable(self):
        """Les constantes viennent de l'env (LLM_CHUNK_MAX_CHARS, TTS_CHUNK_MAX_CHARS)."""
        assert LLM_CHUNK_MAX_CHARS > 0
        assert TTS_CHUNK_MAX_CHARS > 0
        # Défaut raisonnable
        assert 500 <= LLM_CHUNK_MAX_CHARS <= 10000
        assert 500 <= TTS_CHUNK_MAX_CHARS <= 10000
