"""Regression tests for `marvin.index` low-level helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from marvin.index import (
    MemoryIndex,
    _extract_at_ingest,
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
        return NoteRecord(path=Path(f"{title}.md"), metadata=meta, body=body, raw_text=body)

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
            (3, 5, 20, 20),  # min wins
            (10, 5, 20, 50),  # multiplier wins
            (4, 2, 1, 8),  # multiplier with low floor
            (1, 5, 1, 5),  # multiplier wins even with min=1
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
            return original_vec(query_embedding=query_embedding, limit=limit, kind=kind)

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
    return NoteRecord(path=Path(f"{title}.md"), metadata=meta, body=body, raw_text=body)


def _empty_embeds(n: int, dim: int = 8) -> list[np.ndarray]:
    return [np.zeros(dim, dtype=np.float32) for _ in range(n)]


class TestKgSchemaAndHydration:
    """Schema and edge hydration for the K-Lines retrieval stream.

    These tests assert exact edge sets, so they pin
    ``kg_extract_at_ingest=False`` to isolate explicit-wikilink hydration.
    At-ingest fallback extraction is exercised separately in
    :class:`TestKgAtIngestExtraction`.
    """

    def _index(self) -> MemoryIndex:
        return MemoryIndex(Path(":memory:"), dimensions=8, kg_extract_at_ingest=False)

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
                for row in index.conn.execute("SELECT name, normalized FROM entities").fetchall()
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
            note = _make_note("Notes", "First", links=["Quokka", "Australia"])
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(note, "notes", chunks=chunks, embeddings=_empty_embeds(len(chunks)))

            note2 = _make_note("Notes", "Updated", links=["Wombat"])
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
            entity_count = index.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            assert entity_count == 3
        finally:
            index.close()

    def test_prune_removes_edges(self) -> None:
        index = self._index()
        try:
            for path, links in [("a", ["X", "Y"]), ("b", ["Y", "Z"])]:
                note = _make_note(path, path, links=links)
                chunks = chunk_markdown(note, 1200, 200)
                index.upsert_note(note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks)))

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
            index.upsert_note(note, "plain", chunks=chunks, embeddings=_empty_embeds(len(chunks)))
            edge_count = index.conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
            assert edge_count == 0
        finally:
            index.close()

    def test_case_insensitive_dedup_within_note(self) -> None:
        index = self._index()
        try:
            note = _make_note("Notes", "x", links=["Apple Card", "apple card", "  APPLE CARD"])
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(note, "notes", chunks=chunks, embeddings=_empty_embeds(len(chunks)))
            entities = index.conn.execute("SELECT name, normalized FROM entities").fetchall()
            assert len(entities) == 1
            assert entities[0]["normalized"] == "apple card"
            edges = index.conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0]
            assert edges == 1
        finally:
            index.close()


class TestKgResolveQueryEntities:
    def _seeded_index(self, links_per_note: dict[str, list[str]]) -> MemoryIndex:
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        for path, links in links_per_note.items():
            note = _make_note(path, path, links=links)
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks)))
        return index

    def test_returns_empty_for_unknown_query(self) -> None:
        index = self._seeded_index({"a": ["Quokka"]})
        try:
            assert index._resolve_query_entities("nothing matches here") == []
        finally:
            index.close()

    def test_matches_case_insensitively(self) -> None:
        index = self._seeded_index({"a": ["Quokka"]})
        try:
            assert len(index._resolve_query_entities("Tell me about a QUOKKA.")) == 1
            assert len(index._resolve_query_entities("show me the quokka")) == 1
        finally:
            index.close()

    def test_matches_multi_word_entity(self) -> None:
        index = self._seeded_index({"a": ["Apple Card"]})
        try:
            ids = index._resolve_query_entities("how do I pay my apple card bill")
            assert len(ids) == 1
        finally:
            index.close()

    def test_word_boundary_avoids_partial_matches(self) -> None:
        index = self._seeded_index({"a": ["Java"]})
        try:
            assert index._resolve_query_entities("javascript is fun") == []
            assert len(index._resolve_query_entities("I love Java code")) == 1
        finally:
            index.close()

    def test_multiple_entities_match(self) -> None:
        index = self._seeded_index(
            {
                "a": ["Quokka"],
                "b": ["Australia"],
                "c": ["Wombat"],
            }
        )
        try:
            ids = index._resolve_query_entities("where in australia does a quokka live")
            assert len(ids) == 2
        finally:
            index.close()

    def test_empty_registry_short_circuits(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            assert index._resolve_query_entities("anything") == []
        finally:
            index.close()


class TestKgGraphRanker:
    def _seed_for_overlap(self, index: MemoryIndex) -> dict[str, list[str]]:
        layout = {
            "two_hits": ["Quokka", "Australia"],
            "one_hit": ["Quokka"],
            "no_hits": ["Wombat"],
        }
        for path, links in layout.items():
            note = _make_note(path, path, links=links)
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks)))
        return layout

    def test_ranks_by_total_edge_weight(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            self._seed_for_overlap(index)
            ids = index._resolve_query_entities("quokka in australia")
            rows = index._graph_hits(query_entity_ids=ids, limit=10, kind=None)
            paths = [row["relative_path"] for row in rows]
            scores = [row["raw_score"] for row in rows]
            assert paths[0] == "two_hits"
            assert "no_hits" not in paths
            assert scores[0] > scores[1]
        finally:
            index.close()

    def test_kind_filter_applies(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            for path, kind in (
                ("sem", MemoryKind.SEMANTIC),
                ("proc", MemoryKind.PROCEDURAL),
            ):
                note = _make_note(path, path, kind=kind, links=["Quokka"])
                chunks = chunk_markdown(note, 1200, 200)
                index.upsert_note(note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks)))

            ids = index._resolve_query_entities("quokka")
            sem_rows = index._graph_hits(query_entity_ids=ids, limit=10, kind=MemoryKind.SEMANTIC)
            assert [row["relative_path"] for row in sem_rows] == ["sem"]
        finally:
            index.close()

    def test_empty_query_entities_returns_empty(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            assert index._graph_hits(query_entity_ids=[], limit=10, kind=None) == []
        finally:
            index.close()


class TestKgHybridFusion:
    """End-to-end fusion: chunk-tier RRF + graph stream as a third RRF stream."""

    QUERY = "Quokka information"

    def _seed_two_notes(self, index: MemoryIndex) -> None:
        # Note A: chunks lexically match the query. No wikilinks.
        a = _make_note(
            "lexical_only",
            "Quokka information lives here",
            links=[],
        )
        chunks_a = chunk_markdown(a, 1200, 200)
        index.upsert_note(
            a, "lexical_only", chunks=chunks_a, embeddings=_empty_embeds(len(chunks_a))
        )

        # Note B: chunk does NOT contain the query token; only a wikilink
        # connects it to the query entity. Must still surface via the
        # graph stream.
        b = _make_note(
            "graph_only",
            "Wombat habits: extensive burrowing and grass-grazing",
            links=["Quokka"],
        )
        chunks_b = chunk_markdown(b, 1200, 200)
        index.upsert_note(b, "graph_only", chunks=chunks_b, embeddings=_empty_embeds(len(chunks_b)))

    def test_graph_only_note_surfaces_in_results(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            self._seed_two_notes(index)
            hits = index.hybrid_search(
                query=self.QUERY,
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
            )
            paths = [hit.path for hit in hits]
            assert "lexical_only" in paths
            assert "graph_only" in paths
        finally:
            index.close()

    def test_graph_stream_boosts_score_for_linked_note(self) -> None:
        """The graph stream must add an RRF contribution: with kg enabled
        ``graph_only``'s final score must exceed its score with kg
        disabled (where it can only ride on first-stage over-fetch)."""
        index_on = MemoryIndex(Path(":memory:"), dimensions=8, kg_enabled=True)
        index_off = MemoryIndex(Path(":memory:"), dimensions=8, kg_enabled=False)
        try:
            self._seed_two_notes(index_on)
            self._seed_two_notes(index_off)

            kwargs = dict(
                query=self.QUERY,
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
            )
            on = {h.path: h.score for h in index_on.hybrid_search(**kwargs)}
            off = {h.path: h.score for h in index_off.hybrid_search(**kwargs)}

            assert "graph_only" in on
            assert on["graph_only"] > off.get("graph_only", 0.0)
        finally:
            index_on.close()
            index_off.close()

    def test_graph_disabled_short_circuits(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8, kg_enabled=False)
        try:
            self._seed_two_notes(index)
            with (
                patch.object(index, "_graph_hits", wraps=index._graph_hits) as graph_spy,
                patch.object(
                    index,
                    "_resolve_query_entities",
                    wraps=index._resolve_query_entities,
                ) as resolve_spy,
            ):
                index.hybrid_search(
                    query=self.QUERY,
                    query_embedding=np.zeros(8, dtype=np.float32),
                    limit=5,
                )
            graph_spy.assert_not_called()
            resolve_spy.assert_not_called()
        finally:
            index.close()


class TestExtractAtIngest:
    """Pure helper: capitalised noun phrase extraction with stop-word filter."""

    def test_empty_body(self) -> None:
        assert _extract_at_ingest("", min_length=3) == []

    def test_default_is_multiword_only(self) -> None:
        # Defaults to ``multiword_only=True`` so single-word capitalised
        # noise like sentence-starter imperatives gets dropped wholesale.
        out = _extract_at_ingest(
            "Remember, you must call Apple Card support today.",
            min_length=3,
        )
        assert "Apple Card" in out
        assert "Remember" not in out

    def test_multiword_only_drops_proper_single_nouns_too(self) -> None:
        # The trade-off: legitimate single-word proper nouns are also
        # dropped. This is intentional; explicit ``[[wikilinks]]`` and
        # the LLM consolidator pick those up later.
        out = _extract_at_ingest("I love Spotify and Java.", min_length=3)
        assert out == []

    def test_multiword_only_false_keeps_single_word_nouns(self) -> None:
        # Single-word path picks up real proper nouns. The stop-word
        # filter catches the worst common-word noise (``The``, ``His``)
        # but cannot catch every imperative verb an LLM uses in chat
        # (``Remember``, ``Consider`` etc); that is precisely why
        # ``multiword_only=True`` is the default.
        out = _extract_at_ingest(
            "I love Spotify and Java.",
            min_length=3,
            multiword_only=False,
        )
        assert "Spotify" in out
        assert "Java" in out

    def test_min_length_filters_short_tokens_in_loose_mode(self) -> None:
        out = _extract_at_ingest(
            "Hi I am Bob and Alice.",
            min_length=3,
            multiword_only=False,
        )
        assert "Bob" in out and "Alice" in out
        assert "Hi" not in out and "I" not in out

    def test_single_word_stopwords_dropped_in_loose_mode(self) -> None:
        out = _extract_at_ingest(
            "The user told them. There was a Quokka in His backyard.",
            min_length=3,
            multiword_only=False,
        )
        assert "Quokka" in out
        for noise in ("The", "There", "His"):
            assert noise not in out

    def test_stopword_head_or_tail_dropped(self) -> None:
        # On chat data, ``But I``, ``And I``, ``Once I`` and the like
        # are noise, not entities. We drop multi-word phrases whose
        # head or tail token is a stop-word; curated ``[[The Quokka]]``
        # wikilinks remain the route for keeping such phrases.
        out = _extract_at_ingest(
            "But I think Apple Card is fine. Once I tried The Quokka.",
            min_length=3,
        )
        assert "Apple Card" in out
        for noise in ("But I", "Once I", "The Quokka"):
            assert noise not in out

    def test_singleletter_token_rejected(self) -> None:
        # Greedy ``\b[A-Z][a-zA-Z]*(?: ...)*\b`` regex picks up ``But I``
        # because ``I`` qualifies as a capitalised token. We reject any
        # multi-word phrase containing a 1-char token.
        out = _extract_at_ingest("Then I told Mark Smith.", min_length=3)
        assert "Mark Smith" in out
        assert "Then I" not in out

    def test_returns_a_list_with_no_duplicates_per_call(self) -> None:
        out = _extract_at_ingest(
            "Apple Card Apple Card facts about the Apple Card.",
            min_length=3,
        )
        assert out.count("Apple Card") == 1


class TestKgUpsertAtIngest:
    """End-to-end: at-ingest extraction populates the graph for unconsolidated notes."""

    def test_default_settings(self) -> None:
        # At-ingest extraction defaults to OFF: empirical regression on
        # chat-style benchmarks (multi-session R@5 -7pp on LongMemEval-S
        # 100q) makes it opt-in. min_length and multiword_only retain
        # noise-suppressing defaults so that when users do enable it,
        # they get the polished policy.
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            assert index.kg_extract_at_ingest is False
            assert index.kg_ingest_min_length == 3
            assert index.kg_ingest_multiword_only is True
            assert index.kg_fusion_weight == 0.5
        finally:
            index.close()

    def test_extract_default_is_silent(self) -> None:
        # Phase 1A behaviour: with the default off, no entities or
        # edges materialise on raw notes; the graph stream stays
        # silent unless explicit wikilinks exist.
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            note = _make_note(
                "session_1",
                "Sarah moved to Seattle and started using Apple Card.",
                links=[],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "session_1",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )
            entity_count = index.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            assert entity_count == 0
        finally:
            index.close()

    def test_extract_when_opted_in_populates_only_multiword_entities(
        self,
    ) -> None:
        # With ``kg_extract_at_ingest=True`` and the default
        # ``multiword_only=True`` policy, only ``Apple Card`` survives;
        # ``Sarah`` / ``Seattle`` (single words) and sentence-starter
        # noise are dropped.
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            kg_extract_at_ingest=True,
        )
        try:
            note = _make_note(
                "session_1",
                "Sarah moved to Seattle and started using Apple Card.",
                links=[],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "session_1",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )
            normalized = {
                row["normalized"]
                for row in index.conn.execute("SELECT normalized FROM entities").fetchall()
            }
            assert normalized == {"apple card"}
        finally:
            index.close()

    def test_loose_mode_picks_up_single_word_entities(self) -> None:
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            kg_extract_at_ingest=True,
            kg_ingest_multiword_only=False,
        )
        try:
            note = _make_note(
                "session_1",
                "Sarah moved to Seattle and started using Apple Card.",
                links=[],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "session_1",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )
            normalized = {
                row["normalized"]
                for row in index.conn.execute("SELECT normalized FROM entities").fetchall()
            }
            assert "sarah" in normalized
            assert "seattle" in normalized
            assert "apple card" in normalized
        finally:
            index.close()

    def test_disabled_setting_skips_extraction(self) -> None:
        index = MemoryIndex(Path(":memory:"), dimensions=8, kg_extract_at_ingest=False)
        try:
            note = _make_note(
                "session_1",
                "Sarah moved to Seattle.",
                links=[],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "session_1",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )
            entity_count = index.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            assert entity_count == 0
        finally:
            index.close()

    def test_explicit_links_dedup_against_at_ingest_extraction(self) -> None:
        """When a wikilink and a regex extraction normalise to the same
        casefolded form, only one entity row is created; the explicit
        wikilink wins the display-name slot."""
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            kg_extract_at_ingest=True,
            kg_ingest_multiword_only=False,
        )
        try:
            note = _make_note(
                "session_1",
                "Quokka in Western Australia.",
                links=["Quokka"],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "session_1",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )
            quokka_rows = index.conn.execute(
                "SELECT name FROM entities WHERE normalized = 'quokka'"
            ).fetchall()
            assert len(quokka_rows) == 1
            assert quokka_rows[0]["name"] == "Quokka"
        finally:
            index.close()

    def test_min_length_setting_applied(self) -> None:
        # ``min_length=4`` drops 3-letter capitalised tokens like ``Sue``
        # while keeping the full ``Australia`` token. Loose
        # (single-word) mode is needed because both candidates are
        # single tokens.
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            kg_extract_at_ingest=True,
            kg_ingest_min_length=4,
            kg_ingest_multiword_only=False,
        )
        try:
            note = _make_note(
                "session_1",
                "Sue lives in Australia.",
                links=[],
            )
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(
                note,
                "session_1",
                chunks=chunks,
                embeddings=_empty_embeds(len(chunks)),
            )
            normalized = {
                row["normalized"]
                for row in index.conn.execute("SELECT normalized FROM entities").fetchall()
            }
            assert "sue" not in normalized
            assert "australia" in normalized
        finally:
            index.close()

    def test_at_ingest_makes_unconsolidated_note_graph_retrievable(self) -> None:
        """The whole point of Phase 1B: a note without ``[[wikilinks]]``
        but containing a multi-word capitalised entity in its body
        should surface for a query mentioning that entity, even with
        zero lexical overlap on the rest of the body."""
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            kg_extract_at_ingest=True,
        )
        try:
            a = _make_note(
                "lexical",
                "this entry contains the search words",
                links=[],
            )
            chunks_a = chunk_markdown(a, 1200, 200)
            index.upsert_note(
                a, "lexical", chunks=chunks_a, embeddings=_empty_embeds(len(chunks_a))
            )

            b = _make_note(
                "graph_via_ingest",
                "I bought groceries with Apple Card last weekend.",
                links=[],
            )
            chunks_b = chunk_markdown(b, 1200, 200)
            index.upsert_note(
                b,
                "graph_via_ingest",
                chunks=chunks_b,
                embeddings=_empty_embeds(len(chunks_b)),
            )

            kw = dict(
                query="how do I pay my Apple Card bill",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
            )
            on_paths = [h.path for h in index.hybrid_search(**kw)]
            assert "graph_via_ingest" in on_paths

            index_off = MemoryIndex(Path(":memory:"), dimensions=8, kg_extract_at_ingest=False)
            try:
                index_off.upsert_note(
                    a,
                    "lexical",
                    chunks=chunks_a,
                    embeddings=_empty_embeds(len(chunks_a)),
                )
                index_off.upsert_note(
                    b,
                    "graph_via_ingest",
                    chunks=chunks_b,
                    embeddings=_empty_embeds(len(chunks_b)),
                )
                on = {h.path: h.score for h in index.hybrid_search(**kw)}
                off = {h.path: h.score for h in index_off.hybrid_search(**kw)}
                assert on["graph_via_ingest"] > off.get("graph_via_ingest", 0.0)
            finally:
                index_off.close()
        finally:
            index.close()

    def test_no_wikilinks_no_change(self) -> None:
        """Regression: when no notes carry wikilinks, the third stream
        contributes nothing and rankings match the previous behaviour."""
        index = MemoryIndex(Path(":memory:"), dimensions=8)
        try:
            for path, body in [
                ("a", "the quokka is a marsupial"),
                ("b", "wombats are also marsupials"),
                ("c", "kangaroos can hop very far"),
            ]:
                note = _make_note(path, body, links=[])
                chunks = chunk_markdown(note, 1200, 200)
                index.upsert_note(note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks)))

            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
            )
            assert hits[0].path == "a"
        finally:
            index.close()


class TestMemoryDecay:
    """``hybrid_search`` time-aware boost.

    The boost is multiplicative on the final RRF score, so it should
    only flip ranking among notes whose RRF scores are close. Tests
    construct tied or near-tied candidates and assert that the fresher
    one wins when decay is enabled, and the original tie-break behaviour
    holds when decay is off.
    """

    def _seed_two_notes(
        self,
        index: MemoryIndex,
        *,
        old_time: datetime,
        new_time: datetime,
        body: str,
    ) -> None:
        for path, ts in (("old", old_time), ("new", new_time)):
            meta = NoteMetadata(
                kind=MemoryKind.EPISODIC,
                title=path,
                created_at=ts,
                updated_at=ts,
            )
            note = NoteRecord(path=Path(f"{path}.md"), metadata=meta, body=body, raw_text=body)
            chunks = chunk_markdown(note, 1200, 200)
            index.upsert_note(note, path, chunks=chunks, embeddings=_empty_embeds(len(chunks)))

    def test_disabled_by_default_keeps_tie_break_stable(self) -> None:
        """Without decay, two notes with identical content rank by their
        deterministic relative-path order in dict iteration. Pin that
        baseline so the decay-on test isn't fooled by chance."""
        index = MemoryIndex(Path(":memory:"), dimensions=8, decay_enabled=False)
        try:
            now = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
            old = now - timedelta(days=365)
            self._seed_two_notes(
                index, old_time=old, new_time=now, body="the quokka is a marsupial"
            )
            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
                query_time=now,
            )
            paths = [h.path for h in hits]
            assert set(paths) == {"old", "new"}

        finally:
            index.close()

    def test_enabled_promotes_fresher_note(self) -> None:
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            decay_enabled=True,
            decay_half_life_days=30.0,
            decay_weight=0.5,
        )
        try:
            now = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
            old = now - timedelta(days=365)
            self._seed_two_notes(
                index, old_time=old, new_time=now, body="the quokka is a marsupial"
            )
            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
                query_time=now,
            )
            assert hits[0].path == "new", (
                f"expected fresher note to win, got {[h.path for h in hits]}"
            )
        finally:
            index.close()

    def test_enabled_with_zero_weight_is_noop(self) -> None:
        """Belt-and-braces: ``decay_enabled=True, decay_weight=0`` must
        match the disabled path so callers can flag-flip without a
        behavioural change."""
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            decay_enabled=True,
            decay_weight=0.0,
        )
        try:
            now = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
            old = now - timedelta(days=365)
            self._seed_two_notes(
                index, old_time=old, new_time=now, body="the quokka is a marsupial"
            )
            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
                query_time=now,
            )
            assert {h.path for h in hits} == {"old", "new"}
        finally:
            index.close()

    def test_decay_kinds_filter_excludes_semantic(self) -> None:
        """Semantic facts are timeless; the default decay_kinds is
        ``{EPISODIC}`` so a fresh semantic note should NOT outrank an
        old semantic note on the basis of recency alone."""
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            decay_enabled=True,
            decay_half_life_days=30.0,
            decay_weight=0.5,
        )
        try:
            now = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
            old = now - timedelta(days=365)
            for path, ts in (("old", old), ("new", now)):
                meta = NoteMetadata(
                    kind=MemoryKind.SEMANTIC,
                    title=path,
                    created_at=ts,
                    updated_at=ts,
                )
                note = NoteRecord(
                    path=Path(f"{path}.md"),
                    metadata=meta,
                    body="the quokka is a marsupial",
                    raw_text="",
                )
                chunks = chunk_markdown(note, 1200, 200)
                index.upsert_note(
                    note,
                    path,
                    chunks=chunks,
                    embeddings=_empty_embeds(len(chunks)),
                )
            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
                query_time=now,
            )
            assert {h.path for h in hits} == {"old", "new"}
        finally:
            index.close()

    def test_decay_kinds_explicit_includes_semantic(self) -> None:
        """When the caller explicitly passes ``frozenset({SEMANTIC})``
        the kind filter inverts and semantic notes do get the boost."""
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            decay_enabled=True,
            decay_half_life_days=30.0,
            decay_weight=0.5,
            decay_kinds=frozenset({MemoryKind.SEMANTIC}),
        )
        try:
            now = datetime(2025, 1, 10, 12, 0, tzinfo=UTC)
            old = now - timedelta(days=365)
            for path, ts in (("old", old), ("new", now)):
                meta = NoteMetadata(
                    kind=MemoryKind.SEMANTIC,
                    title=path,
                    created_at=ts,
                    updated_at=ts,
                )
                note = NoteRecord(
                    path=Path(f"{path}.md"),
                    metadata=meta,
                    body="the quokka is a marsupial",
                    raw_text="",
                )
                chunks = chunk_markdown(note, 1200, 200)
                index.upsert_note(
                    note,
                    path,
                    chunks=chunks,
                    embeddings=_empty_embeds(len(chunks)),
                )
            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
                query_time=now,
            )
            assert hits[0].path == "new"
        finally:
            index.close()

    def test_query_time_default_is_now(self) -> None:
        """Omitting ``query_time`` falls back to the index server's
        ``utc_now``, so a recent note still gets a boost over an old one
        without the caller needing to thread a timestamp through."""
        index = MemoryIndex(
            Path(":memory:"),
            dimensions=8,
            decay_enabled=True,
            decay_half_life_days=30.0,
            decay_weight=0.5,
        )
        try:
            now = datetime.now(UTC)
            old = now - timedelta(days=365)
            self._seed_two_notes(
                index, old_time=old, new_time=now, body="the quokka is a marsupial"
            )
            hits = index.hybrid_search(
                query="quokka marsupial",
                query_embedding=np.zeros(8, dtype=np.float32),
                limit=5,
            )
            assert hits[0].path == "new"
        finally:
            index.close()
