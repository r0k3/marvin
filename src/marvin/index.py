from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import sqlite_vec

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
    def __init__(self, db_path: Path, dimensions: int) -> None:
        self.db_path = db_path
        self.dimensions = dimensions
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
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
        vec_hits = self._vector_hits(
            query_embedding=query_embedding, limit=max(limit * 5, 20), kind=kind
        )
        fts_hits = self._fts_hits(query=query, limit=max(limit * 5, 20), kind=kind)

        scores: dict[int, float] = defaultdict(float)
        details: dict[int, sqlite3.Row] = {}
        rrf_k = 60.0

        for rank, row in enumerate(vec_hits, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] += 1.0 / (rrf_k + rank)
            details[chunk_id] = row

        for rank, row in enumerate(fts_hits, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] += 1.0 / (rrf_k + rank)
            details[chunk_id] = row

        note_scores: dict[str, float] = defaultdict(float)
        note_best_chunk: dict[str, sqlite3.Row] = {}
        for chunk_id, score in scores.items():
            row = details[chunk_id]
            relative_path = row["relative_path"]
            note_scores[relative_path] = max(note_scores[relative_path], score)
            existing = note_best_chunk.get(relative_path)
            if existing is None or score > note_scores.get(relative_path, 0.0):
                note_best_chunk[relative_path] = row

        sorted_paths = sorted(note_scores, key=lambda path: note_scores[path], reverse=True)[:limit]
        hits: list[SearchHit] = []
        for path in sorted_paths:
            row = note_best_chunk[path]
            hits.append(
                SearchHit(
                    title=row["title"],
                    kind=MemoryKind(row["kind"]),
                    path=row["relative_path"],
                    score=round(note_scores[path], 6),
                    excerpt=_excerpt(row["text"]),
                    tags=json.loads(row["tags_json"] or "[]"),
                    links=json.loads(row["links_json"] or "[]"),
                    chunk_text=row["text"] if include_chunk_text else None,
                )
            )
        return hits

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
