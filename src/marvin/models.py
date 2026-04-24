from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFLECTIVE = "reflective"

    @property
    def folder_name(self) -> str:
        return self.value.capitalize()


class NoteMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: MemoryKind
    title: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)


class NoteRecord(BaseModel):
    path: Path
    metadata: NoteMetadata
    body: str
    raw_text: str

    @property
    def title(self) -> str:
        return self.metadata.title

    @property
    def kind(self) -> MemoryKind:
        return self.metadata.kind


class ChunkRecord(BaseModel):
    note_path: str
    note_title: str
    note_kind: MemoryKind
    chunk_index: int
    heading: str
    tags_text: str
    text: str


class SearchHit(BaseModel):
    title: str
    kind: MemoryKind
    path: str
    score: float
    excerpt: str
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    # Full text of the best-matching chunk. Optional and populated only when
    # a caller sets ``include_chunk_text=True`` on a search; reranking needs
    # the full window (excerpts are truncated for display).
    chunk_text: str | None = None


class SyncReport(BaseModel):
    scanned: int = 0
    indexed: int = 0
    removed: int = 0


class MemoryWriteResult(BaseModel):
    title: str
    kind: MemoryKind
    path: str
    created: bool
    message: str


class SessionContext(BaseModel):
    task: str
    procedural: list[SearchHit] = Field(default_factory=list)
    semantic: list[SearchHit] = Field(default_factory=list)
    reflective: list[SearchHit] = Field(default_factory=list)
    recent_episodes: list[SearchHit] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


class SessionClosureResult(BaseModel):
    episode: MemoryWriteResult
    stored_semantic: list[MemoryWriteResult] = Field(default_factory=list)
    stored_procedures: list[MemoryWriteResult] = Field(default_factory=list)
    stored_reflections: list[MemoryWriteResult] = Field(default_factory=list)
