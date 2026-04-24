from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .broker import MarvinBroker
from .config import MarvinSettings
from .embeddings import EmbeddingService
from .git import GitManager
from .index import MemoryIndex, chunk_markdown
from .models import (
    MemoryKind,
    MemoryWriteResult,
    NoteRecord,
    SearchHit,
    SessionClosureResult,
    SessionContext,
    SyncReport,
)
from .reranker import RerankerService
from .vault import VaultStore, normalize_links, normalize_tags


class MarvinService:
    def __init__(
        self,
        settings: MarvinSettings,
        broker: MarvinBroker | None = None,
        git_manager: GitManager | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.vault = VaultStore(self.settings.resolved_vault_path)
        self.broker = broker
        self.git_manager = git_manager
        self.embedder = EmbeddingService(
            provider=self.settings.embedding_provider,
            model_name=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )
        self.reranker = RerankerService(
            provider=(
                self.settings.rerank_provider
                if self.settings.rerank_enabled
                else "none"
            ),
            model_name=self.settings.rerank_model,
            max_chars=self.settings.rerank_max_chars,
        )
        self.index = MemoryIndex(
            self.settings.index_path, dimensions=self.settings.embedding_dimensions
        )

    def close(self) -> None:
        self.index.close()

    def sync(self) -> SyncReport:
        report = SyncReport()
        existing_paths: set[str] = set()
        notes = self.vault.list_notes()
        report.scanned = len(notes)

        for note in notes:
            relative_path = str(note.path.relative_to(self.settings.resolved_vault_path))
            existing_paths.add(relative_path)
            content_hash = self._content_hash(note)
            if self.index.note_is_current(relative_path, content_hash):
                continue
            chunks = chunk_markdown(note, self.settings.chunk_size, self.settings.chunk_overlap)
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            self.index.upsert_note(
                note, relative_path=relative_path, chunks=chunks, embeddings=embeddings
            )
            report.indexed += 1

        report.removed = self.index.prune_deleted_notes(existing_paths)
        return report

    def search(
        self, query: str, *, kind: MemoryKind | None = None, limit: int | None = None
    ) -> list[SearchHit]:
        self.sync()
        effective_limit = limit or self.settings.search_limit
        query_embedding = self.embedder.embed_text(query)
        if self.settings.rerank_enabled:
            # Over-fetch with the hybrid ranker, then let the cross-encoder
            # reorder the top pool before returning the final top-K. We ask
            # the index for the full chunk text (not just excerpt) so the
            # reranker has the real matched window to score, and strip it
            # on return to keep the wire payload compact.
            pool_size = max(effective_limit, self.settings.rerank_depth)
            pool = self.index.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                limit=pool_size,
                kind=kind,
                include_chunk_text=True,
            )
            docs = [(hit.chunk_text or hit.excerpt or hit.title) for hit in pool]
            scores = self.reranker.score(query, docs)
            order = sorted(
                range(len(pool)), key=lambda i: (-scores[i], i)
            )[:effective_limit]
            return [
                pool[i].model_copy(
                    update={"score": round(scores[i], 6), "chunk_text": None}
                )
                for i in order
            ]
        return self.index.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=effective_limit,
            kind=kind,
        )

    def recent(
        self, *, kind: MemoryKind | None = None, limit: int | None = None
    ) -> list[SearchHit]:
        self.sync()
        return self.index.recent(limit=limit or self.settings.recency_limit, kind=kind)

    def get_note(self, identifier: str) -> NoteRecord | None:
        return self.vault.get_note(identifier)

    def remember_semantic(
        self,
        *,
        concept: str,
        content: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        source: dict[str, str] | None = None,
    ) -> MemoryWriteResult:
        existing = self.vault.find_note(title=concept, kind=MemoryKind.SEMANTIC)
        facts = [content.strip()]
        if existing is not None:
            facts = self._extract_bullets(existing.body, section_name="Facts")
            if content.strip() not in facts:
                facts.append(content.strip())
        body = self._render_bullets_document(
            title=concept, intro="", section_name="Facts", items=facts
        )
        return self._write_note(
            kind=MemoryKind.SEMANTIC,
            title=concept,
            body=body,
            tags=tags,
            links=links,
            source=source,
            existing_path=existing.path if existing else None,
        )

    def store_procedure(
        self,
        *,
        title: str,
        steps: list[str],
        applicability: list[str] | None = None,
        anti_patterns: list[str] | None = None,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        source: dict[str, str] | None = None,
    ) -> MemoryWriteResult:
        existing = self.vault.find_note(title=title, kind=MemoryKind.PROCEDURAL)
        sections = [
            self._render_numbered_section("Procedure", steps),
            self._render_bullet_section("Applies When", applicability or []),
            self._render_bullet_section("Avoid", anti_patterns or []),
        ]
        body = "\n\n".join(section for section in sections if section).strip()
        return self._write_note(
            kind=MemoryKind.PROCEDURAL,
            title=title,
            body=body,
            tags=tags,
            links=links,
            source=source,
            existing_path=existing.path if existing else None,
        )

    def log_episode(
        self,
        *,
        title: str,
        summary: str,
        details: str = "",
        tags: list[str] | None = None,
        links: list[str] | None = None,
        source: dict[str, str] | None = None,
    ) -> MemoryWriteResult:
        sections = [
            f"## Summary\n{summary.strip()}" if summary.strip() else "",
            f"## Details\n{details.strip()}" if details.strip() else "",
        ]
        body = "\n\n".join(section for section in sections if section).strip()
        return self._write_note(
            kind=MemoryKind.EPISODIC,
            title=title,
            body=body,
            tags=tags,
            links=links,
            source=source,
            unique=True,
        )

    def reflect(
        self,
        *,
        title: str,
        insight: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        source: dict[str, str] | None = None,
    ) -> MemoryWriteResult:
        existing = self.vault.find_note(title=title, kind=MemoryKind.REFLECTIVE)
        body = f"## Insight\n{insight.strip()}"
        return self._write_note(
            kind=MemoryKind.REFLECTIVE,
            title=title,
            body=body,
            tags=tags,
            links=links,
            source=source,
            existing_path=existing.path if existing else None,
        )

    def prepare_session(
        self,
        *,
        task: str,
        repo_name: str | None = None,
        technologies: list[str] | None = None,
        limit: int = 8,
    ) -> SessionContext:
        query_terms = [task.strip()]
        if repo_name:
            query_terms.append(repo_name)
        if technologies:
            query_terms.extend([tech for tech in technologies if tech.strip()])
        query = " ".join(term for term in query_terms if term)

        procedural = self.search(query, kind=MemoryKind.PROCEDURAL, limit=max(2, limit // 2))
        semantic = self.search(query, kind=MemoryKind.SEMANTIC, limit=max(2, limit // 2))
        reflective = self.search(query, kind=MemoryKind.REFLECTIVE, limit=max(1, limit // 3))
        recent_episodes = self.recent(kind=MemoryKind.EPISODIC, limit=max(2, limit // 3))
        guidance = self._derive_guidance(procedural, semantic, reflective)

        return SessionContext(
            task=task,
            procedural=procedural,
            semantic=semantic,
            reflective=reflective,
            recent_episodes=recent_episodes,
            guidance=guidance,
        )

    def hook_session_end(
        self,
        *,
        title: str,
        summary: str,
        details: str = "",
        tags: list[str] | None = None,
        links: list[str] | None = None,
        semantic_facts: list[str] | None = None,
        procedures: list[dict[str, object]] | None = None,
        reflections: list[str] | None = None,
        source: dict[str, str] | None = None,
    ) -> SessionClosureResult:
        episode = self.log_episode(
            title=title,
            summary=summary,
            details=details,
            tags=tags,
            links=links,
            source=source,
        )

        stored_semantic: list[MemoryWriteResult] = []
        for fact in semantic_facts or []:
            concept, text = self._split_fact(fact)
            stored_semantic.append(
                self.remember_semantic(
                    concept=concept,
                    content=text,
                    tags=tags,
                    links=links,
                    source=source,
                )
            )

        stored_procedures: list[MemoryWriteResult] = []
        for procedure in procedures or []:
            title_value = str(procedure.get("title") or "Procedure")
            steps_value = [str(step) for step in procedure.get("steps") or []]
            applicability = [str(step) for step in procedure.get("applicability") or []]
            anti_patterns = [str(step) for step in procedure.get("anti_patterns") or []]
            stored_procedures.append(
                self.store_procedure(
                    title=title_value,
                    steps=steps_value,
                    applicability=applicability,
                    anti_patterns=anti_patterns,
                    tags=tags,
                    links=links,
                    source=source,
                )
            )

        stored_reflections: list[MemoryWriteResult] = []
        for insight in reflections or []:
            insight_title = f"Reflection - {title}"
            stored_reflections.append(
                self.reflect(
                    title=insight_title,
                    insight=insight,
                    tags=tags,
                    links=links,
                    source=source,
                )
            )

        return SessionClosureResult(
            episode=episode,
            stored_semantic=stored_semantic,
            stored_procedures=stored_procedures,
            stored_reflections=stored_reflections,
        )

    def _write_note(
        self,
        *,
        kind: MemoryKind,
        title: str,
        body: str,
        tags: list[str] | None,
        links: list[str] | None,
        source: dict[str, str] | None,
        existing_path: Path | None = None,
        unique: bool = False,
    ) -> MemoryWriteResult:
        path, created = self.vault.write_note(
            kind=kind,
            title=title,
            body=body,
            tags=normalize_tags(tags),
            links=normalize_links(links),
            source=source,
            existing_path=existing_path,
            unique=unique,
        )
        note = self.vault.read_note(path)
        relative_path = str(path.relative_to(self.settings.resolved_vault_path))
        chunks = chunk_markdown(note, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.index.upsert_note(
            note, relative_path=relative_path, chunks=chunks, embeddings=embeddings
        )

        # Git & NATS Hooks
        if self.git_manager:
            self.git_manager.commit(f"auto-save: {title}")
        if self.broker:
            import asyncio

            # Fire and forget publication
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.broker.publish(
                        "memory.created", {"path": relative_path, "kind": kind.value}
                    )
                )
            except RuntimeError:
                pass  # Not running in async context

        return MemoryWriteResult(
            title=title,
            kind=kind,
            path=relative_path,
            created=created,
            message=("created" if created else "updated"),
        )

    def _derive_guidance(
        self,
        procedural: list[SearchHit],
        semantic: list[SearchHit],
        reflective: list[SearchHit],
    ) -> list[str]:
        guidance: list[str] = []
        for hit in procedural[:3]:
            guidance.append(f"Follow procedure '{hit.title}' before coding changes.")
        for hit in semantic[:2]:
            guidance.append(f"Keep '{hit.title}' in view; it contains known project facts.")
        for hit in reflective[:1]:
            guidance.append(f"Apply reflection '{hit.title}' to avoid repeating prior mistakes.")
        return guidance

    def _extract_bullets(self, body: str, *, section_name: str) -> list[str]:
        marker = f"## {section_name}"
        if marker not in body:
            return []
        _, _, tail = body.partition(marker)
        lines = []
        for raw_line in tail.splitlines()[1:]:
            if raw_line.startswith("## "):
                break
            line = raw_line.strip()
            if line.startswith("- "):
                lines.append(line[2:].strip())
        return lines

    def _render_bullets_document(
        self, *, title: str, intro: str, section_name: str, items: Iterable[str]
    ) -> str:
        sections = []
        if intro.strip():
            sections.append(intro.strip())
        sections.append(self._render_bullet_section(section_name, list(items)))
        return "\n\n".join(section for section in sections if section).strip()

    def _render_bullet_section(self, heading: str, items: list[str]) -> str:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            return ""
        body = "\n".join(f"- {item}" for item in cleaned)
        return f"## {heading}\n{body}"

    def _render_numbered_section(self, heading: str, items: list[str]) -> str:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            return ""
        body = "\n".join(f"{index}. {item}" for index, item in enumerate(cleaned, start=1))
        return f"## {heading}\n{body}"

    def _split_fact(self, fact: str) -> tuple[str, str]:
        if ":" in fact:
            concept, text = fact.split(":", 1)
            if concept.strip() and text.strip():
                return concept.strip(), text.strip()
        return "Learned Fact", fact.strip()

    def _content_hash(self, note: NoteRecord) -> str:
        from hashlib import sha256

        return sha256(note.raw_text.encode("utf-8")).hexdigest()
