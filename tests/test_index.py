"""Regression tests for `marvin.index` low-level helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


class TestFirstStageOverfetch:
    """``hybrid_search`` per-stream limit is driven by configurable knobs."""

    def _seed(self, index: MemoryIndex) -> None:
        for i in range(3):
            note = NoteRecord(
                path=Path(f"n{i}.md"),
                metadata=NoteMetadata(kind=MemoryKind.SEMANTIC, title=f"n{i}"),
                body=f"the quokka entry number {i}",
                raw_text="",
            )
            chunks = chunk_markdown(note, 1200, 200)
            embeds = [np.zeros(8, dtype=np.float32) for _ in chunks]
            index.upsert_note(note, f"n{i}", chunks=chunks, embeddings=embeds)

    @pytest.mark.parametrize(
        "limit, overfetch, overfetch_min, expected",
        [
            (3, 5, 20, 20),    # min wins
            (10, 5, 20, 50),   # multiplier wins
            (4, 2, 1, 8),      # multiplier with low floor
            (1, 5, 1, 5),      # multiplier wins even with min=1
        ],
    )
    def test_per_stream_limit_uses_settings(
        self,
        limit: int,
        overfetch: int,
        overfetch_min: int,
        expected: int,
    ) -> None:
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            first_stage_overfetch=overfetch,
            first_stage_overfetch_min=overfetch_min,
        )
        self._seed(index)

        captured: dict[str, int] = {}

        original_vec = index._vector_hits
        original_fts = index._fts_hits

        def vec_spy(*, query_embedding, limit, kind):
            captured["vec_limit"] = limit
            return original_vec(
                query_embedding=query_embedding, limit=limit, kind=kind
            )

        def fts_spy(*, query, limit, kind):
            captured["fts_limit"] = limit
            return original_fts(query=query, limit=limit, kind=kind)

        with (
            patch.object(index, "_vector_hits", side_effect=vec_spy),
            patch.object(index, "_fts_hits", side_effect=fts_spy),
        ):
            index.hybrid_search(
                query="quokka",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=limit,
            )

        assert captured["vec_limit"] == expected
        assert captured["fts_limit"] == expected
        index.close()

    def test_invalid_settings_clamped_to_one(self) -> None:
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            first_stage_overfetch=0,
            first_stage_overfetch_min=0,
        )
        assert index.first_stage_overfetch == 1
        assert index.first_stage_overfetch_min == 1
        index.close()
