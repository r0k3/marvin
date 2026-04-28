from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import sqlite_vec

from .extraction import _fallback_extract
from .models import ChunkRecord, MemoryKind, NoteRecord, SearchHit


def chunk_markdown(note: NoteRecord, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    sections = _split_sections(note.body)
    chunks: list[ChunkRecord] = []
    carry = ""
    chunk_index = 0
    tags_text = " ".join(note.metadata.tags)

    for heading, section_body in sections:
        text = section_body.strip()
        if not text:
            continue
        parts = _window_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for part in parts:
            prefix = f"Title: {note.metadata.title}\nKind: {note.metadata.kind.value}\n"
            if carry:
                prefix += f"Context: {carry}\n"
            if heading:
                prefix += f"Heading: {heading}\n"
            chunk_text = f"{prefix}\n{part}".strip()
            chunks.append(
                ChunkRecord(
                    note_path=str(note.path),
                    note_title=note.metadata.title,
                    note_kind=note.metadata.kind,
                    chunk_index=chunk_index,
                    heading=heading,
                    tags_text=tags_text,
                    text=chunk_text,
                )
            )
            carry = part[-min(chunk_overlap, len(part)) :]
            chunk_index += 1

    if not chunks:
        chunks.append(
            ChunkRecord(
                note_path=str(note.path),
                note_title=note.metadata.title,
                note_kind=note.metadata.kind,
                chunk_index=0,
                heading="",
                tags_text=tags_text,
                text=(
                    f"Title: {note.metadata.title}\n"
                    f"Kind: {note.metadata.kind.value}\n\n"
                    f"{note.body.strip()}"
                ).strip(),
            )
        )
    return chunks


def _split_sections(body: str) -> list[tuple[str, str]]:
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.lstrip("# ").strip()
            current_lines = []
            continue
        if line.startswith("# ") and not sections and not current_lines:
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def _window_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    windows: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        windows.append(text[start:end].strip())
        if end == len(text):
            break
        start += step
    return windows


class MemoryIndex:
    def __init__(
        self,
        db_path: Path,
        dimensions: int,
        *,
        first_stage_overfetch: int = 5,
        first_stage_overfetch_min: int = 20,
        kg_enabled: bool = True,
        kg_rrf_k: float = 60.0,
        kg_fusion_weight: float = 0.5,
        kg_extract_at_ingest: bool = False,
        kg_ingest_min_length: int = 3,
        kg_ingest_multiword_only: bool = True,
    ) -> None:
        self.db_path = db_path
        self.dimensions = dimensions
        # Per-stream over-fetch tuning. ``hybrid_search`` pulls
        # ``max(limit * first_stage_overfetch, first_stage_overfetch_min)``
        # chunks from each ranker before RRF fusion. Higher = more recall,
        # more SQL work per query.
        self.first_stage_overfetch = max(1, first_stage_overfetch)
        self.first_stage_overfetch_min = max(1, first_stage_overfetch_min)
        # K-Lines graph stream toggles and RRF damping. Defaults to
        # enabled because hydration is cheap and the stream silently
        # contributes nothing when no entities exist. ``kg_fusion_weight``
        # scales the graph stream's RRF contribution relative to the
        # chunk-tier; values < 1 prevent the (noisier) entity ranking
        # from displacing strong chunk matches, while still allowing
        # graph signal to break ties and surface graph-only notes.
        self.kg_enabled = kg_enabled
        self.kg_rrf_k = kg_rrf_k
        self.kg_fusion_weight = max(0.0, kg_fusion_weight)
        # At-ingest fallback entity extraction (regex over capitalised
        # noun phrases). When enabled, ``upsert_note`` augments
        # ``metadata.links`` with extracted entities before hydrating
        # ``entity_edges``, so the graph stream has signal even on
        # unconsolidated notes.
        self.kg_extract_at_ingest = kg_extract_at_ingest
        self.kg_ingest_min_length = max(1, kg_ingest_min_length)
        self.kg_ingest_multiword_only = kg_ingest_multiword_only
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        # Register ``log`` so ``_graph_hits`` can compute IDF in SQL on
        # builds where the SQLite math extension is not compiled in
        # (Python's bundled sqlite3 typically isn't).
        self.conn.create_function("log", 1, math.log, deterministic=True)
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def note_is_current(self, relative_path: str, content_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT content_hash FROM notes WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()
        return row is not None and row["content_hash"] == content_hash

    def upsert_note(
        self,
        note: NoteRecord,
        relative_path: str,
        chunks: list[ChunkRecord],
        embeddings: list[np.ndarray],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match")

        content_hash = sha256(note.raw_text.encode("utf-8")).hexdigest()
        tags_json = json.dumps(note.metadata.tags)
        links_json = json.dumps(note.metadata.links)

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO notes (
                    relative_path, kind, title, tags_json, links_json, content_hash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    kind = excluded.kind,
                    title = excluded.title,
                    tags_json = excluded.tags_json,
                    links_json = excluded.links_json,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                RETURNING id
                """,
                (
                    relative_path,
                    note.metadata.kind.value,
                    note.metadata.title,
                    tags_json,
                    links_json,
                    content_hash,
                    note.metadata.updated_at.isoformat(),
                ),
            )
            note_id = cursor.fetchone()[0]

            existing_chunk_rows = self.conn.execute(
                "SELECT id FROM chunks WHERE note_id = ?",
                (note_id,),
            ).fetchall()
            existing_chunk_ids = [row["id"] for row in existing_chunk_rows]
            if existing_chunk_ids:
                placeholders = ",".join("?" for _ in existing_chunk_ids)
                self.conn.execute(
                    f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})",
                    existing_chunk_ids,
                )
            self.conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))

            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk_cursor = self.conn.execute(
                    """
                    INSERT INTO chunks (
                        note_id, chunk_index, title, kind, heading, tags_text, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note_id,
                        chunk.chunk_index,
                        chunk.note_title,
                        chunk.note_kind.value,
                        chunk.heading,
                        chunk.tags_text,
                        chunk.text,
                    ),
                )
                chunk_id = chunk_cursor.lastrowid
                packed = sqlite_vec.serialize_float32(np.asarray(embedding, dtype=np.float32))
                self.conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, packed),
                )

            self._replace_entity_edges(
                note_id, self._collect_entity_strings(note)
            )

    def _collect_entity_strings(self, note: NoteRecord) -> list[str]:
        """Combine explicit wikilinks with at-ingest regex extraction.

        Explicit ``metadata.links`` (parsed from ``[[wikilinks]]``) come
        first so they win the display-name slot in the entity registry;
        at-ingest extractions only contribute entities the wikilinks did
        not already cover. When ``kg_extract_at_ingest`` is disabled we
        return ``metadata.links`` unchanged, preserving Phase 1A
        behaviour.
        """
        explicit = list(note.metadata.links or [])
        if not self.kg_extract_at_ingest:
            return explicit
        seen = {_normalize_entity(link) for link in explicit}
        for ent in _extract_at_ingest(
            note.body,
            min_length=self.kg_ingest_min_length,
            multiword_only=self.kg_ingest_multiword_only,
        ):
            normalized = _normalize_entity(ent)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            explicit.append(ent)
        return explicit

    def _replace_entity_edges(self, note_id: int, links: list[str]) -> None:
        """Rebuild ``entity_edges`` for ``note_id`` from the note's wikilinks.

        Called inside ``upsert_note``'s open transaction. The resolution
        is case-fold based: ``[[Apple Card]]`` and ``[[apple card]]``
        collapse to a single entity row; the display name for newly seen
        entities is whatever the note used first.
        """
        self.conn.execute(
            "DELETE FROM entity_edges WHERE note_id = ?", (note_id,)
        )
        seen: set[str] = set()
        for link in links or []:
            normalized = _normalize_entity(link)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            display = link.strip()
            self.conn.execute(
                """
                INSERT INTO entities (name, normalized) VALUES (?, ?)
                ON CONFLICT(normalized) DO NOTHING
                """,
                (display, normalized),
            )
            row = self.conn.execute(
                "SELECT id FROM entities WHERE normalized = ?", (normalized,)
            ).fetchone()
            if row is None:
                continue
            self.conn.execute(
                """
                INSERT INTO entity_edges (note_id, entity_id, weight)
                VALUES (?, ?, 1.0)
                ON CONFLICT(note_id, entity_id) DO NOTHING
                """,
                (note_id, row["id"]),
            )

    def prune_deleted_notes(self, existing_paths: set[str]) -> int:
        rows = self.conn.execute("SELECT id, relative_path FROM notes").fetchall()
        removed = 0
        with self.conn:
            for row in rows:
                if row["relative_path"] in existing_paths:
                    continue
                chunk_rows = self.conn.execute(
                    "SELECT id FROM chunks WHERE note_id = ?", (row["id"],)
                ).fetchall()
                chunk_ids = [chunk_row["id"] for chunk_row in chunk_rows]
                if chunk_ids:
                    placeholders = ",".join("?" for _ in chunk_ids)
                    self.conn.execute(
                        f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})",
                        chunk_ids,
                    )
                self.conn.execute("DELETE FROM chunks WHERE note_id = ?", (row["id"],))
                self.conn.execute(
                    "DELETE FROM entity_edges WHERE note_id = ?", (row["id"],)
                )
                self.conn.execute("DELETE FROM notes WHERE id = ?", (row["id"],))
                removed += 1
        return removed

    def recent(self, limit: int, kind: MemoryKind | None = None) -> list[SearchHit]:
        query = "SELECT title, kind, relative_path, tags_json, links_json FROM notes"
        params: list[object] = []
        if kind is not None:
            query += " WHERE kind = ?"
            params.append(kind.value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        hits: list[SearchHit] = []
        for row in rows:
            hits.append(
                SearchHit(
                    title=row["title"],
                    kind=MemoryKind(row["kind"]),
                    path=row["relative_path"],
                    score=0.0,
                    excerpt="",
                    tags=json.loads(row["tags_json"] or "[]"),
                    links=json.loads(row["links_json"] or "[]"),
                )
            )
        return hits

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        limit: int,
        kind: MemoryKind | None = None,
        *,
        include_chunk_text: bool = False,
    ) -> list[SearchHit]:
        """Three-stream retrieval: vec + FTS chunks fused, then RRF-fused with graph.

        Two tiers:

        1. Chunk-level RRF over ``_vector_hits`` and ``_fts_hits``;
           max-pool chunk scores back to notes.
        2. Note-level RRF that fuses the chunk-tier note ranking with the
           K-Lines graph stream (``_graph_hits``). This is what lets a
           note surface even when none of its chunks lexically or
           semantically match the query, as long as it links to the
           query entities.

        When ``kg_enabled`` is ``False`` or no query entities resolve,
        the graph stream contributes nothing and the result reduces to
        the previous behaviour.
        """
        per_stream_limit = max(
            limit * self.first_stage_overfetch,
            self.first_stage_overfetch_min,
        )
        vec_hits = self._vector_hits(
            query_embedding=query_embedding, limit=per_stream_limit, kind=kind
        )
        fts_hits = self._fts_hits(query=query, limit=per_stream_limit, kind=kind)

        scores: dict[int, float] = defaultdict(float)
        details: dict[int, sqlite3.Row] = {}
        rrf_k = self.kg_rrf_k

        for rank, row in enumerate(vec_hits, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] += 1.0 / (rrf_k + rank)
            details[chunk_id] = row

        for rank, row in enumerate(fts_hits, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] += 1.0 / (rrf_k + rank)
            details[chunk_id] = row

        note_chunk_score: dict[str, float] = defaultdict(float)
        note_best_chunk: dict[str, sqlite3.Row] = {}
        for chunk_id, score in scores.items():
            row = details[chunk_id]
            relative_path = row["relative_path"]
            if score > note_chunk_score[relative_path]:
                note_chunk_score[relative_path] = score
                note_best_chunk[relative_path] = row

        chunk_ranking = sorted(
            note_chunk_score.items(), key=lambda kv: -kv[1]
        )

        graph_ranking: list[tuple[str, float]] = []
        if self.kg_enabled:
            query_entity_ids = self._resolve_query_entities(query)
            if query_entity_ids:
                graph_rows = self._graph_hits(
                    query_entity_ids=query_entity_ids,
                    limit=per_stream_limit,
                    kind=kind,
                )
                graph_ranking = [
                    (row["relative_path"], float(row["raw_score"]))
                    for row in graph_rows
                ]
                # Backfill best-chunk metadata for notes that the graph
                # stream surfaced but the chunk-tier missed; otherwise
                # they have no excerpt to render.
                missing = [
                    path for path, _ in graph_ranking if path not in note_best_chunk
                ]
                if missing:
                    fill_sql = """
                        SELECT
                            c.id AS chunk_id,
                            n.relative_path,
                            n.title,
                            n.kind,
                            n.tags_json,
                            n.links_json,
                            c.text
                        FROM chunks c
                        JOIN notes n ON n.id = c.note_id
                        WHERE n.relative_path IN ({placeholders})
                          AND c.chunk_index = 0
                    """.format(
                        placeholders=",".join("?" for _ in missing)
                    )
                    for row in self.conn.execute(fill_sql, missing).fetchall():
                        note_best_chunk[row["relative_path"]] = row

        # Second-tier RRF: chunk-fused note ranking + graph note ranking.
        # The graph stream is weighted by ``kg_fusion_weight`` (default
        # 0.5) so a noisy entity ranking cannot overrule strong chunk
        # matches; setting it to 1.0 reverts to symmetric fusion.
        final_scores: dict[str, float] = defaultdict(float)
        for rank, (path, _) in enumerate(chunk_ranking, start=1):
            final_scores[path] += 1.0 / (rrf_k + rank)
        for rank, (path, _) in enumerate(graph_ranking, start=1):
            final_scores[path] += self.kg_fusion_weight / (rrf_k + rank)

        sorted_paths = sorted(
            final_scores, key=lambda path: final_scores[path], reverse=True
        )[:limit]
        hits: list[SearchHit] = []
        for path in sorted_paths:
            row = note_best_chunk.get(path)
            if row is None:
                # Defensive: should not happen given the backfill above,
                # but guard against schema drift / empty-chunk notes.
                continue
            hits.append(
                SearchHit(
                    title=row["title"],
                    kind=MemoryKind(row["kind"]),
                    path=row["relative_path"],
                    score=round(final_scores[path], 6),
                    excerpt=_excerpt(row["text"]),
                    tags=json.loads(row["tags_json"] or "[]"),
                    links=json.loads(row["links_json"] or "[]"),
                    chunk_text=row["text"] if include_chunk_text else None,
                )
            )
        return hits

    def _resolve_query_entities(self, query: str) -> list[int]:
        """Return entity ids whose normalized form appears (word-bounded) in ``query``.

        Linear scan over the entity registry. Acceptable up to several
        thousand entities; profile and switch to a precompiled regex
        alternation or trie if a real vault outgrows that.
        """
        rows = self.conn.execute(
            "SELECT id, normalized FROM entities"
        ).fetchall()
        if not rows:
            return []
        casefold_query = query.casefold()
        matched: list[int] = []
        for row in rows:
            normalized = row["normalized"] or ""
            if not normalized:
                continue
            if re.search(
                r"\b" + re.escape(normalized) + r"\b", casefold_query
            ):
                matched.append(row["id"])
        return matched

    def _graph_hits(
        self,
        *,
        query_entity_ids: list[int],
        limit: int,
        kind: MemoryKind | None,
    ) -> list[sqlite3.Row]:
        """IDF-weighted note ranking by overlap with ``query_entity_ids``.

        Without IDF, a query that resolves to a common at-ingest entity
        (the speaker's name, the platform brand) drowns the chunk-tier
        signal because every session edges into it. With IDF, common
        entities contribute little and rare entities a lot -- the same
        intuition BM25 uses for terms.

        Score: ``sum_over_matched_entities(weight * idf(e))`` where
        ``idf(e) = log((N + 1) / (df(e) + 0.5))`` and ``N`` is total
        notes (``df`` is restricted to ``kind`` when filtering).

        SQL keeps it one round-trip: a CTE materialises per-entity df,
        the outer join applies the weight, and ``GROUP BY note_id``
        sums.
        """
        if not query_entity_ids:
            return []
        if kind is not None:
            n_notes_row = self.conn.execute(
                "SELECT COUNT(*) FROM notes WHERE kind = ?", (kind.value,)
            ).fetchone()
        else:
            n_notes_row = self.conn.execute(
                "SELECT COUNT(*) FROM notes"
            ).fetchone()
        n_notes = n_notes_row[0] if n_notes_row else 0
        if n_notes == 0:
            return []

        placeholders = ",".join("?" for _ in query_entity_ids)
        if kind is not None:
            df_cte = f"""
                SELECT ee.entity_id, COUNT(*) AS df
                FROM entity_edges ee
                JOIN notes n ON n.id = ee.note_id
                WHERE n.kind = ?
                GROUP BY ee.entity_id
            """
            df_params: list[object] = [kind.value]
        else:
            df_cte = """
                SELECT entity_id, COUNT(*) AS df
                FROM entity_edges
                GROUP BY entity_id
            """
            df_params = []

        sql = f"""
            WITH ent_df AS (
                {df_cte}
            )
            SELECT
                n.id AS note_id,
                n.relative_path,
                n.title,
                n.kind,
                n.tags_json,
                n.links_json,
                COUNT(*) AS overlap,
                SUM(
                    ee.weight
                    * log(({n_notes} + 1.0) / (COALESCE(ent_df.df, 1) + 0.5))
                ) AS raw_score
            FROM entity_edges ee
            JOIN notes n ON n.id = ee.note_id
            LEFT JOIN ent_df ON ent_df.entity_id = ee.entity_id
            WHERE ee.entity_id IN ({placeholders})
        """
        params: list[object] = [*df_params, *query_entity_ids]
        if kind is not None:
            sql += " AND n.kind = ?"
            params.append(kind.value)
        sql += " GROUP BY n.id ORDER BY raw_score DESC LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def _vector_hits(
        self, *, query_embedding: np.ndarray, limit: int, kind: MemoryKind | None
    ) -> list[sqlite3.Row]:
        sql = """
            SELECT
                c.id AS chunk_id,
                n.relative_path,
                n.title,
                n.kind,
                n.tags_json,
                n.links_json,
                c.text,
                distance
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.rowid
            JOIN notes n ON n.id = c.note_id
            WHERE embedding MATCH ? AND k = ?
        """
        params: list[object] = [
            sqlite_vec.serialize_float32(np.asarray(query_embedding, dtype=np.float32)),
            limit,
        ]
        if kind is not None:
            sql += " AND n.kind = ?"
            params.append(kind.value)
        sql += " ORDER BY distance"
        return self.conn.execute(sql, params).fetchall()

    def _fts_hits(self, *, query: str, limit: int, kind: MemoryKind | None) -> list[sqlite3.Row]:
        cleaned = _sanitize_fts_query(query)
        if not cleaned:
            return []
        sql = """
            SELECT
                c.id AS chunk_id,
                n.relative_path,
                n.title,
                n.kind,
                n.tags_json,
                n.links_json,
                c.text,
                rank
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN notes n ON n.id = c.note_id
            WHERE chunks_fts MATCH ?
        """
        params: list[object] = [cleaned]
        if kind is not None:
            sql += " AND n.kind = ?"
            params.append(kind.value)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    relative_path TEXT UNIQUE NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    links_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    heading TEXT NOT NULL,
                    tags_text TEXT NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )
            # Entity graph backing the K-Lines retrieval stream. ``entities``
            # is the canonical entity registry (one row per ``[[wikilink]]``
            # ever seen, deduped by case-folded form). ``entity_edges`` is
            # the many-to-many relation between notes and entities; rows are
            # rebuilt on every ``upsert_note`` and explicitly cleared on
            # ``prune_deleted_notes`` (sqlite-vec does not run with
            # ``PRAGMA foreign_keys=ON`` so we cannot rely on cascade).
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    normalized TEXT UNIQUE NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS entities_normalized_idx
                ON entities(normalized)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_edges (
                    note_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (note_id, entity_id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS entity_edges_entity_id_idx
                ON entity_edges(entity_id)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS entity_edges_note_id_idx
                ON entity_edges(note_id)
                """
            )
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    title,
                    kind,
                    heading,
                    tags_text,
                    text,
                    content='chunks',
                    content_rowid='id'
                )
                """
            )
            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    embedding float[{self.dimensions}]
                )
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, title, kind, heading, tags_text, text)
                    VALUES (new.id, new.title, new.kind, new.heading, new.tags_text, new.text);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, title, kind,
                        heading, tags_text, text)
                    VALUES ('delete', old.id, old.title, old.kind,
                        old.heading, old.tags_text, old.text);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, title, kind,
                        heading, tags_text, text)
                    VALUES ('delete', old.id, old.title, old.kind,
                        old.heading, old.tags_text, old.text);
                    INSERT INTO chunks_fts(rowid, title, kind,
                        heading, tags_text, text)
                    VALUES (new.id, new.title, new.kind,
                        new.heading, new.tags_text, new.text);
                END
                """
            )


def _excerpt(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit] + ("..." if len(normalized) > limit else "")


def _normalize_entity(name: str) -> str:
    """Canonical key for entity dedup and query-side resolution.

    Mirrors ``str.casefold()`` semantics used elsewhere in the vault for
    link comparison (more aggressive than ``lower()`` for international
    text such as German eszett / Turkish dotted I).
    """
    return name.strip().casefold()


# Common English determiners / pronouns / particles that get capitalised
# at sentence start and produce pure noise when fed to the regex
# entity extractor. Multi-word matches (``Apple Card``) are passed
# through even when their first token would be a stop-word in isolation;
# only *single-word* extractions are dropped if they hit this set.
_INGEST_STOPWORDS: frozenset[str] = frozenset({
    "the", "this", "that", "these", "those",
    "and", "but", "for", "yet", "nor",
    "with", "from", "into", "onto", "upon", "after", "before",
    "yes", "yeah", "yep", "no", "nope", "okay", "ok", "sure",
    "hello", "hi", "hey",
    "how", "what", "when", "where", "why", "who", "which",
    "she", "they", "you", "his", "her", "him", "them",
    "their", "your", "yours", "ours", "theirs",
    "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "well", "now", "then", "there", "here", "also",
    "good", "great", "nice", "thanks", "please",
})


def _extract_at_ingest(
    body: str,
    *,
    min_length: int,
    multiword_only: bool = True,
) -> list[str]:
    """Regex-extract capitalised noun phrases for at-ingest graph hydration.

    Wraps ``extraction._fallback_extract`` and layers three filters
    against the dominant noise patterns on chat-style data:

    * ``min_length`` filters very short tokens.
    * If ``multiword_only`` (default ``True``), single-word
      extractions are dropped wholesale. Capitalised single words on
      chat data are dominated by sentence-starter imperatives
      (``Remember``, ``However``, ``Can``, ``Did``); a clean
      multi-word policy throws those away.
    * Multi-word phrases are then quality-filtered:
        - any 1-character token (``But I``, ``And I``, ``As I``) is
          rejected; the regex is greedy across capitalised tokens and
          ``I`` qualifies, producing pseudo-phrases that are pure
          noise;
        - the first or last token being in the stop-word list rejects
          ``But Mark`` / ``Mark But`` style sentence fragments. This
          forfeits a few legitimate ``The X`` phrases; on chat data
          the trade is net positive, and curated knowledge can still
          be supplied via explicit ``[[wikilinks]]``.
    * Loose mode (``multiword_only=False``): only the single-word
      stop-word filter applies. Useful on curated text.
    """
    if not body:
        return []
    raw = _fallback_extract(body)
    out: list[str] = []
    for ent in raw:
        cleaned = ent.strip()
        if len(cleaned) < min_length:
            continue
        tokens = cleaned.split()
        is_multiword = len(tokens) > 1
        if multiword_only and not is_multiword:
            continue
        if not is_multiword and cleaned.casefold() in _INGEST_STOPWORDS:
            continue
        if is_multiword:
            if any(len(t) < 2 for t in tokens):
                continue
            if tokens[0].casefold() in _INGEST_STOPWORDS:
                continue
            if tokens[-1].casefold() in _INGEST_STOPWORDS:
                continue
        out.append(cleaned)
    return out


# FTS5 reserves these characters for query syntax (phrase, prefix, boolean,
# column filter, grouping, NEAR). Stripping them keeps end-user queries safe
# without requiring callers to escape — we only want bag-of-words matching.
_FTS_SPECIAL = set('"*+-():^?!.,;')


def _sanitize_fts_query(query: str) -> str:
    """Turn a free-form query into a safe FTS5 OR-of-phrases expression.

    Default FTS5 uses AND semantics, which is far too strict for
    natural-language questions ("how do I log in Python?" would only match
    documents containing every word). We split on whitespace, drop FTS5
    operators, quote each token (so it survives reserved keywords like
    ``AND``/``OR``/``NEAR``), and join with ``OR`` so the ranker scores by
    how many query terms each document contains.
    """
    chars = [" " if ch in _FTS_SPECIAL else ch for ch in query]
    tokens = [tok for tok in "".join(chars).split() if tok]
    if not tokens:
        return ""
    quoted = [f'"{tok}"' for tok in tokens]
    return " OR ".join(quoted)
