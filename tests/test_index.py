"""Regression tests for `marvin.index` low-level helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from marvin.index import (
    MemoryIndex,
    _normalize_entity,
    _sanitize_fts_query,
    chunk_markdown,
)
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


class TestNormalizeEntity:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Apple Card", "apple card"),
            ("apple card", "apple card"),
            ("  APPLE CARD  ", "apple card"),
            ("Straße", "strasse"),  # casefold > lower
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_canonicalisation(self, raw: str, expected: str) -> None:
        assert _normalize_entity(raw) == expected


def _make_note(
    title: str,
    body: str,
    *,
    kind: MemoryKind = MemoryKind.SEMANTIC,
    links: list[str] | None = None,
) -> NoteRecord:
    meta = NoteMetadata(kind=kind, title=title, links=links or [])
    return NoteRecord(
        path=Path(f"{title}.md"), metadata=meta, body=body, raw_text=body
    )


def _empty_embeds(n: int, dim: int = 8) -> list[np.ndarray]:
    return [np.zeros(dim, dtype=np.float32) for _ in range(n)]


class TestKgSchemaAndHydration:
    """Schema and edge hydration for the K-Lines retrieval stream."""

    def _index(self) -> MemoryIndex:
        return MemoryIndex(Path(":memory:"), dimensions=8)

    def test_schema_creates_kg_tables(self) -> None:
        index = self._index()
        try:
            tables = {
                row[0]
                for row in index.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "entities" in tables
            assert "entity_edges" in tables
        finally:
            index.close()

    def test_upsert_populates_edges_from_links(self) -> None:
        index = self._index()
        try:
            note = _make_note(
                "Marsupials",
                "Quokkas live in Western Australia.",
                links=["Quokka", "Australia"],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "marsupials",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )

            entities = {
                row["normalized"]: row["name"]
                for row in index.conn.execute(
                    "SELECT name, normalized FROM entities"
                ).fetchall()
            }
            assert entities == {"quokka": "Quokka", "australia": "Australia"}

            edges = index.conn.execute(
                """
                SELECT e.normalized
                FROM entity_edges ee
                JOIN entities e ON e.id = ee.entity_id
                JOIN notes n ON n.id = ee.note_id
                WHERE n.relative_path = ?
                """,
                ("marsupials",),
            ).fetchall()
            assert {row["normalized"] for row in edges} == {
                "quokka",
                "australia",
            }
        finally:
            index.close()

    def test_reupsert_replaces_edges(self) -> None:
        index = self._index()
        try:
            note = _make_note(
                "Notes", "First", links=["Quokka", "Australia"]
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note, "notes", chunks=chunks, embeddings=_empty_embeds(len(chunks))
            )

            note2 = _make_note(
                "Notes", "Updated", links=["Wombat"]
            )
            chunks2 = chunk_markdown(note2, 1200, 200)
            index.upsert_note(
                note2,
                "notes",
                chunks=chunks2,
                embeddings=_empty_embeds(len(chunks2)),
            )

            edges = index.conn.execute(
                """
                SELECT e.normalized
                FROM entity_edges ee
                JOIN entities e ON e.id = ee.entity_id
                JOIN notes n ON n.id = ee.note_id
                WHERE n.relative_path = ?
                """,
                ("notes",),
            ).fetchall()
            assert {row["normalized"] for row in edges} == {"wombat"}

            # Stale entities stay in the registry (cheap, may come back).
            entity_count = index.conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]
            assert entity_count == 3
        finally:
            index.close()

    def test_prune_removes_edges(self) -> None:
        index = self._index()
        try:
            for path, links in [("a", ["X", "Y"]), ("b", ["Y", "Z"])]:
                note = _make_note(path, path, links=links)
                chunks = chunk_markdown(note, 1200, 200)
                index.upsert_note(
                    note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks))
                )

            assert index.prune_deleted_notes(existing_paths={"a"}) == 1

            remaining = index.conn.execute(
                """
                SELECT n.relative_path, e.normalized
                FROM entity_edges ee
                JOIN entities e ON e.id = ee.entity_id
                JOIN notes n ON n.id = ee.note_id
                ORDER BY n.relative_path, e.normalized
                """
            ).fetchall()
            assert [(r["relative_path"], r["normalized"]) for r in remaining] == [
                ("a", "x"),
                ("a", "y"),
            ]
        finally:
            index.close()

    def test_no_links_no_edges(self) -> None:
        index = self._index()
        try:
            note = _make_note("plain", "no wikilinks here", links=[])
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note, "plain", chunks=chunks, embeddings=_empty_embeds(len(chunks))
            )
            edge_count = index.conn.execute(
                "SELECT COUNT(*) FROM entity_edges"
            ).fetchone()[0]
            assert edge_count == 0
        finally:
            index.close()

    def test_case_insensitive_dedup_within_note(self) -> None:
        index = self._index()
        try:
            note = _make_note(
                "Notes", "x", links=["Apple Card", "apple card", "  APPLE CARD"]
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note, "notes", chunks=chunks, embeddings=_empty_embeds(len(chunks))
            )
            entities = index.conn.execute(
                "SELECT name, normalized FROM entities"
            ).fetchall()
            assert len(entities) == 1
            assert entities[0]["normalized"] == "apple card"
            edges = index.conn.execute(
                "SELECT COUNT(*) FROM entity_edges"
            ).fetchone()[0]
            assert edges == 1
        finally:
            index.close()
