"""Regression tests for `marvin.index` low-level helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marvin.index import MemoryIndex, _sanitize_fts_query, chunk_markdown
from marvin.models import MemoryKind, NoteMetadata, NoteRecord


class TestSanitizeFtsQuery:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("hello world", '"hello" OR "world"'),
            ('with "quotes"', '"with" OR "quotes"'),
            ("trailing question?", '"trailing" OR "question"'),
            (
                "punctuation: a, b; c.",
                '"punctuation" OR "a" OR "b" OR "c"',
            ),
            ("op chars + - ^ * ! ?", '"op" OR "chars"'),
            ("(parenthesised)", '"parenthesised"'),
            ("AND OR NOT NEAR", '"AND" OR "OR" OR "NOT" OR "NEAR"'),
            ("   ", ""),
            ("", ""),
        ],
    )
    def test_sanitizes_to_or_query(self, raw, expected):
        assert _sanitize_fts_query(raw) == expected


class TestFtsHitsRobustness:
    """The FTS5 backend must not crash on common query punctuation."""

    def _make_note(self, title: str, body: str) -> NoteRecord:
        meta = NoteMetadata(kind=MemoryKind.EPISODIC, title=title)
        return NoteRecord(
            path=Path(f"{title}.md"), metadata=meta, body=body, raw_text=body
        )

    def test_query_with_question_mark_does_not_crash(self):
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        note = self._make_note("Quokka", "the quokka is a small marsupial")
        chunks = chunk_markdown(note, 1200, 200)
        embeds = [np.zeros(8, dtype=np.float32) for _ in chunks]
        index.upsert_note(note, "quokka", chunks=chunks, embeddings=embeds)

        rows = index._fts_hits(query="What is a quokka?", limit=5, kind=None)
        assert any(row["relative_path"] == "quokka" for row in rows)
        index.close()

    def test_empty_query_returns_empty(self):
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        rows = index._fts_hits(query="???", limit=5, kind=None)
        assert rows == []
        index.close()
