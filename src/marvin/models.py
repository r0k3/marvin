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


class FactAspect(StrEnum):
    KNOWLEDGE = "knowledge"
    PREFERENCE = "preference"
    DECISION = "decision"
    GOAL = "goal"
    PROBLEM = "problem"
    BELIEF = "belief"
    DIRECTIVE = "directive"


class SemanticFact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str
    predicate: str
    value: str
    aspect: FactAspect = FactAspect.KNOWLEDGE
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    source: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    deprecated: bool = False
    deprecated_at: datetime | None = None
    deprecated_reason: str | None = None
    replaced_by: str | None = None

    def deprecate(self, *, reason: str, replaced_by: str | None = None) -> None:
        self.deprecated = True
        self.deprecated_at = utc_now()
        self.deprecated_reason = reason
        self.replaced_by = replaced_by


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
    facts: list[SemanticFact] = Field(default_factory=list)
    # Episodic notes are marked consolidated once the offline consolidation
    # pass has extracted semantic facts from them, so the next pass skips them
    # instead of re-extracting. Meaningless (and absent from frontmatter) for
    # the other kinds.
    consolidated: bool = False
    # K-line procedural templates carry adaptive utility metadata: how often the
    # template was selected (usage_count) and an effectiveness EMA in [0,1].
    # Persisted (and present in frontmatter) only once non-zero.
    usage_count: int = 0
    effectiveness: float = 0.0


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


class ConsistencyReport(BaseModel):
    """Agreement between the authoritative vault and the derived index."""

    consistent: bool
    vault_notes: int
    indexed_notes: int
    missing_from_index: list[str] = Field(default_factory=list)
    orphaned_in_index: list[str] = Field(default_factory=list)


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
