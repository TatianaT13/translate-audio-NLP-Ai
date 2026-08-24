"""Tests unitaires — cleanup des sorties LLM (traduction vs summarize).

Capture 2 fixes récents :
  - `d8350ce` : summarize retournait juste "# Notes de réunion" car les regex
    _clean_meta (pensées pour la traduction single-line) mangeaient tout le
    markdown au premier ":"
  - `36b39f4` : le LLM wrappe parfois sa réponse dans ```markdown ... ```,
    ce qui casse le rendu direct côté UI

Ces tests importent uniquement des fonctions pures (pas de call réseau).
"""

from __future__ import annotations

from llm.main import _clean_meta


class TestCleanMetaForTranslation:
    """_clean_meta est safe pour la traduction single-line (comportement historique)."""

    def test_removes_here_is_prefix(self):
        assert _clean_meta("Here is the translation:\nBonjour") == "Bonjour"

    def test_removes_sure_prefix(self):
        # La regex matche "sure[,.!]" suivi de contenu jusqu'à \n ou :
        result = _clean_meta("Sure! The translation is: Bonjour")
        # Doit avoir mangé le préfixe métabolique
        assert "Sure" not in result or result == "Bonjour"

    def test_preserves_pure_translation(self):
        """Sans préfixe méta, le texte passe intact."""
        assert _clean_meta("Attention bouchons A6") == "Attention bouchons A6"

    def test_removes_alternatives_block(self):
        text = "Traduction principale\n\nAlternatively, you could also say..."
        assert "alternatively" not in _clean_meta(text).lower()


class TestCleanMetaDangerForMarkdown:
    """Démontre POURQUOI on ne peut PAS passer un résumé markdown dans _clean_meta.

    C'est la raison du fix d8350ce : summarize passe désormais clean=False.
    """

    def test_meta_removes_content_before_first_colon(self):
        """La regex "^here.*?(?:\\n|:)" mange tout jusqu'au premier ":".

        Un markdown de résumé commence souvent par :
          ## Summary
          Topic: X
        → le "Topic:" est le premier ":" — le contenu structuré serait mangé.
        """
        markdown = "Here is the summary:\n## Summary\nTopic: décision release\n..."
        cleaned = _clean_meta(markdown)
        # _clean_meta a mangé jusqu'au 1er : ou \n → on perd du contenu
        # Ce test DOCUMENTE le comportement pour empêcher qu'on repasse
        # summarize par _clean_meta.
        # Si ce test échoue un jour, c'est peut-être qu'on a assoupli la regex.
        # Le vrai check c'est : summarize_meeting doit appeler call_llm(clean=False).
        assert cleaned != markdown, "clean_meta a modifié la sortie (ce qui casse le markdown)"


class TestMarkdownWrapperStrip:
    """Le LLM wrappe parfois sa réponse dans ```markdown``` — on strip côté serveur.

    Reproduit la logique de strip du fix 36b39f4 dans summarize_meeting.
    """

    @staticmethod
    def _strip_wrapper(summary: str) -> str:
        """Copie de la logique du fix 36b39f4."""
        summary = summary.strip()
        if summary.startswith("```"):
            lines = summary.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            summary = "\n".join(lines).strip()
        return summary

    def test_strip_removes_markdown_wrapper(self):
        raw = "```markdown\n## Summary\nContent here\n```"
        assert self._strip_wrapper(raw) == "## Summary\nContent here"

    def test_strip_removes_generic_code_wrapper(self):
        raw = "```\n## Summary\nContent\n```"
        assert self._strip_wrapper(raw) == "## Summary\nContent"

    def test_strip_no_wrapper_leaves_intact(self):
        raw = "## Summary\nJust plain markdown"
        assert self._strip_wrapper(raw) == raw

    def test_strip_handles_no_closing_backticks(self):
        """Cas défensif : LLM ouvre le wrapper mais ne le ferme pas."""
        raw = "```markdown\n## Summary\nContent without closing"
        result = self._strip_wrapper(raw)
        assert "```" not in result
        assert "## Summary" in result

    def test_strip_preserves_inline_code_in_content(self):
        """Un `code inline` dans le contenu ne doit pas être touché."""
        raw = "```markdown\nUtilise la commande `docker ps` pour lister\n```"
        result = self._strip_wrapper(raw)
        assert "`docker ps`" in result
        assert not result.startswith("```")
