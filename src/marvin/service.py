from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .broker import MarvinBroker
from .config import MarvinSettings
from .consolidation import ConsolidationEngine
from .embeddings import EmbeddingService
from .git import GitManager
from .index import MemoryIndex, chunk_markdown
from .klines import (
    MatchSignal,
    TemplateMatch,
    next_effectiveness,
    parse_template_body,
    render_template_body,
    score_template,
)
from .models import (
    ConsistencyReport,
    FactAspect,
    MemoryKind,
    MemoryWriteResult,
    NoteRecord,
    SearchHit,
    SemanticFact,
    SessionClosureResult,
    SessionContext,
    SyncReport,
)
from .reranker import RerankerService
from .vault import VaultStore, normalize_links, normalize_tags


def _parse_decay_kinds(csv: str) -> frozenset[MemoryKind] | None:
    """Decode the ``decay_kinds_csv`` setting.

    Empty string => ``frozenset()`` (decay applies to no kind).
    The literal token ``all`` => ``None`` so the index falls back to
    its default (currently ``EPISODIC`` only) without enumerating
    each kind here.
    """
    text = csv.strip().lower()
    if not text:
        return frozenset()
    if text == "all":
        return frozenset(MemoryKind)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out: set[MemoryKind] = set()
    for token in parts:
        try:
            out.add(MemoryKind(token))
        except ValueError:
            continue
    return frozenset(out) if out else frozenset()


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
            provider=(self.settings.rerank_provider if self.settings.rerank_enabled else "none"),
            model_name=self.settings.rerank_model,
            max_chars=self.settings.rerank_max_chars,
        )
        self.index = MemoryIndex(
            self.settings.index_path,
            dimensions=self.settings.embedding_dimensions,
            first_stage_overfetch=self.settings.first_stage_overfetch,
            first_stage_overfetch_min=self.settings.first_stage_overfetch_min,
            kg_enabled=self.settings.kg_enabled,
            kg_rrf_k=self.settings.kg_rrf_k,
            kg_fusion_weight=self.settings.kg_fusion_weight,
            kg_extract_at_ingest=self.settings.kg_extract_at_ingest,
            kg_ingest_min_length=self.settings.kg_ingest_min_length,
            kg_ingest_multiword_only=self.settings.kg_ingest_multiword_only,
            decay_enabled=self.settings.decay_enabled,
            decay_half_life_days=self.settings.decay_half_life_days,
            decay_weight=self.settings.decay_weight,
            decay_kinds=_parse_decay_kinds(self.settings.decay_kinds_csv),
        )

    def close(self) -> None:
        self.index.close()

    def health(self) -> dict[str, object]:
        """Return a snapshot of the runtime configuration.

        Honest about lazy-loading: backends that have not yet been
        initialised are reported as ``"<provider> (not loaded)"`` so this
        method itself never triggers a model download.
        """
        embed_loaded = self.embedder.loaded_backend_name
        embedding_backend = (
            embed_loaded
            if embed_loaded is not None
            else f"{self.settings.embedding_provider} (not loaded)"
        )

        if not self.settings.rerank_enabled:
            reranker_backend = "disabled"
        else:
            rerank_loaded = self.reranker.loaded_backend_name
            reranker_backend = (
                rerank_loaded
                if rerank_loaded is not None
                else f"{self.settings.rerank_provider} (not loaded)"
            )

        from . import gpu as _gpu

        gpu_libs = _gpu.loaded_libs()

        return {
            "embedding_backend": embedding_backend,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
            "rerank_enabled": self.settings.rerank_enabled,
            "rerank_provider": self.settings.rerank_provider,
            "rerank_model": self.settings.rerank_model,
            "rerank_depth": self.settings.rerank_depth,
            "reranker_backend": reranker_backend,
            "gpu_active": bool(gpu_libs),
            "gpu_lib_count": len(gpu_libs),
            "kg_enabled": self.settings.kg_enabled,
            "kg_rrf_k": self.settings.kg_rrf_k,
            "kg_fusion_weight": self.settings.kg_fusion_weight,
            "kg_extract_at_ingest": self.settings.kg_extract_at_ingest,
            "kg_ingest_min_length": self.settings.kg_ingest_min_length,
            "kg_ingest_multiword_only": self.settings.kg_ingest_multiword_only,
            "decay_enabled": self.settings.decay_enabled,
            "decay_half_life_days": self.settings.decay_half_life_days,
            "decay_weight": self.settings.decay_weight,
            "decay_kinds_csv": self.settings.decay_kinds_csv,
            "vault_path": str(self.settings.resolved_vault_path),
            "index_path": str(self.settings.index_path),
        }

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

    def rebuild(self) -> SyncReport:
        """Drop all derived indexes and regenerate them from the vault.

        The human-readable Markdown files are the authoritative source of
        truth; the index is a cache. This clears that cache and re-derives it
        from scratch, recovering from index corruption or a schema change.
        """
        self.index.clear()
        return self.sync()

    def consistency_check(self) -> ConsistencyReport:
        """Compare the authoritative vault against the derived index.

        Flags notes present in the vault but missing from the index, and
        index rows orphaned from any vault file.
        """
        vault_paths = {
            str(note.path.relative_to(self.settings.resolved_vault_path))
            for note in self.vault.list_notes()
        }
        indexed = self.index.indexed_paths()
        missing_from_index = sorted(vault_paths - indexed)
        orphaned_in_index = sorted(indexed - vault_paths)
        return ConsistencyReport(
            consistent=not missing_from_index and not orphaned_in_index,
            vault_notes=len(vault_paths),
            indexed_notes=len(indexed),
            missing_from_index=missing_from_index,
            orphaned_in_index=orphaned_in_index,
        )

    def search(
        self, query: str, *, kind: MemoryKind | None = None, limit: int | None = None
    ) -> list[SearchHit]:
        self.sync()
        effective_limit = limit or self.settings.search_limit
        query_embedding = self.embedder.embed_text(query)
        return self._search_pool(
            query=query,
            query_embedding=query_embedding,
            kind=kind,
            limit=effective_limit,
        )

    def _search_pool(
        self,
        *,
        query: str,
        query_embedding: np.ndarray,
        kind: MemoryKind | None,
        limit: int,
    ) -> list[SearchHit]:
        """First-stage hybrid retrieval, optionally followed by reranking.

        Caller owns ``sync()`` and ``embed_text()``. This shared core lets
        :meth:`search` and :meth:`prepare_session` issue a single index
        round-trip (and a single rerank pass) instead of one per kind.
        """
        if self.settings.rerank_enabled:
            # Over-fetch with the hybrid ranker, then let the cross-encoder
            # reorder the top pool before returning the final top-K. We ask
            # the index for the full chunk text (not just excerpt) so the
            # reranker has the real matched window to score, and strip it
            # on return to keep the wire payload compact.
            pool_size = max(limit, self.settings.rerank_depth)
            pool = self.index.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                limit=pool_size,
                kind=kind,
                include_chunk_text=True,
            )
            if not pool:
                return []
            docs = [(hit.chunk_text or hit.excerpt or hit.title) for hit in pool]
            scores = self.reranker.score(query, docs)
            order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))[:limit]
            return [
                pool[i].model_copy(update={"score": round(scores[i], 6), "chunk_text": None})
                for i in order
            ]
        return self.index.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
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
        content: str | None = None,
        predicate: str | None = None,
        value: str | None = None,
        aspect: FactAspect | str = FactAspect.KNOWLEDGE,
        confidence: float = 0.6,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        source: dict[str, object] | None = None,
    ) -> MemoryWriteResult:
        value_text = (value if value is not None else content or "").strip()
        if not value_text:
            raise ValueError("remember_semantic requires content or value")
        predicate_text = (predicate or "fact").strip() or "fact"
        aspect_value = self._coerce_fact_aspect(aspect)

        existing = self.vault.find_note(title=concept, kind=MemoryKind.SEMANTIC)
        facts = self._semantic_facts_from_existing(existing, concept=concept)
        new_fact = SemanticFact(
            subject=concept,
            predicate=predicate_text,
            value=value_text,
            aspect=aspect_value,
            confidence=confidence,
            source=source or {},
        )

        duplicate = next(
            (
                fact
                for fact in facts
                if not fact.deprecated
                and self._normalize_fact_predicate(fact.predicate)
                == self._normalize_fact_predicate(new_fact.predicate)
                and self._normalize_fact_value(fact.value)
                == self._normalize_fact_value(new_fact.value)
            ),
            None,
        )
        if duplicate is None:
            for fact in facts:
                if not fact.deprecated and self._normalize_fact_predicate(
                    fact.predicate
                ) == self._normalize_fact_predicate(new_fact.predicate):
                    fact.deprecate(
                        reason=(f"Replaced by newer fact with the same predicate for {concept}."),
                        replaced_by=new_fact.id,
                    )
            facts.append(new_fact)

        body = self._render_semantic_document(facts)
        return self._write_note(
            kind=MemoryKind.SEMANTIC,
            title=concept,
            body=body,
            tags=tags,
            links=links,
            source=source,
            facts=facts,
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

    def register_template(
        self,
        *,
        title: str,
        plan: list[str],
        intents: list[str] | None = None,
        styles: list[str] | None = None,
        entity_types: list[str] | None = None,
        trigger_phrases: list[str] | None = None,
        slots: list[str] | None = None,
        failure_modes: list[str] | None = None,
        intro: str = "",
        tags: list[str] | None = None,
        links: list[str] | None = None,
        source: dict[str, object] | None = None,
    ) -> MemoryWriteResult:
        """Register a K-line procedural template.

        Stored as a procedural note whose body carries the template's trigger
        conditions (``## Intents`` / ``## Styles`` / ``## Entity Types`` /
        ``## Triggers`` keywords), plus ``## Slots`` / ``## Plan`` / ``##
        Failure Modes``. ``match_template`` parses these back and scores them.
        """

        body = render_template_body(
            intents=intents or (),
            styles=styles or (),
            entity_types=entity_types or (),
            trigger_phrases=trigger_phrases or (),
            slots=slots or (),
            plan=plan,
            failure_modes=failure_modes or (),
            intro=intro,
        )
        existing = self.vault.find_note(title=title, kind=MemoryKind.PROCEDURAL)
        return self._write_note(
            kind=MemoryKind.PROCEDURAL,
            title=title,
            body=body,
            tags=tags,
            links=links,
            source=source,
            existing_path=existing.path if existing else None,
        )

    def match_template(
        self,
        context: str = "",
        *,
        intent: str = "",
        styles: Iterable[str] = (),
        entity_types: Iterable[str] = (),
        top_k: int = 5,
    ) -> list[TemplateMatch]:
        """Select K-line templates for the current context (Minsky reactivation).

        Scores every template with the weighted trigger formula
        (``0.5*intent + 0.25*style + 0.25*entity_type + keyword bonus``), with
        intent as a hard gate. Non-template procedural notes and zero-scoring
        templates are dropped. Ties break by adaptive utility (effectiveness,
        then usage), then title.
        """

        signal = MatchSignal(
            intent=intent,
            styles=tuple(styles),
            entity_types=tuple(entity_types),
            text=context,
        )
        matches: list[TemplateMatch] = []
        for note in self.vault.list_notes(kind=MemoryKind.PROCEDURAL):
            template = parse_template_body(title=note.metadata.title, body=note.body)
            if template is None or not template.is_complete():
                continue
            score, matched = score_template(template, signal)
            if score <= 0.0:
                continue
            relative_path = str(note.path.relative_to(self.settings.resolved_vault_path))
            matches.append(
                TemplateMatch(
                    template=template,
                    score=score,
                    matched_phrases=matched,
                    note_path=relative_path,
                    usage_count=note.metadata.usage_count,
                    effectiveness=note.metadata.effectiveness,
                )
            )
        # Prefer higher score, then more-effective then more-used templates
        # (ACT-R-style utility), with title as a final deterministic tiebreak.
        matches.sort(
            key=lambda m: (-m.score, -m.effectiveness, -m.usage_count, m.template.title.casefold())
        )
        return matches[:top_k]

    def record_template_use(self, title: str, *, success: bool, alpha: float = 0.3) -> None:
        """Update a template's adaptive utility after it was applied.

        Increments the usage count and folds ``success`` into the effectiveness
        EMA, so frequently-selected, effective templates rank higher next time.
        """

        note = self.vault.find_note(title=title, kind=MemoryKind.PROCEDURAL)
        if note is None:
            return
        self.vault.write_note(
            kind=MemoryKind.PROCEDURAL,
            title=note.metadata.title,
            body=note.body,
            tags=note.metadata.tags,
            links=note.metadata.links,
            source=note.metadata.source,
            existing_path=note.path,
            usage_count=note.metadata.usage_count + 1,
            effectiveness=next_effectiveness(
                note.metadata.effectiveness, success=success, alpha=alpha
            ),
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

    def consolidate_semantic(
        self,
        *,
        engine: ConsolidationEngine | None = None,
        min_episodes: int = 3,
        max_episodes_per_entity: int = 10,
    ) -> list[MemoryWriteResult]:
        """Phase 1 consolidation: episodic -> semantic, entity-scoped.

        Groups unconsolidated episodes by the entities they mention; for each
        entity mentioned in at least ``min_episodes`` of them, extracts stable
        facts (deduplicated against the entity's known facts), persists them,
        and marks the consumed episodes consolidated. Episodes whose entities
        never cross the threshold stay unconsolidated and keep accumulating.
        """
        engine = engine or ConsolidationEngine()
        by_entity: dict[str, list[NoteRecord]] = {}
        for episode in self.vault.unconsolidated_episodes():
            for entity in episode.metadata.links:
                by_entity.setdefault(entity, []).append(episode)

        results: list[MemoryWriteResult] = []
        consumed: dict[str, Path] = {}
        for entity, episodes in by_entity.items():
            if len(episodes) < min_episodes:
                continue
            batch = episodes[:max_episodes_per_entity]
            known = [fact.value for fact in self._entity_facts(entity)]
            for item in engine.extract_entity_facts(entity, [ep.body for ep in batch], known):
                value = str(item.get("value") or "").strip()
                if not value:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.6))
                except (TypeError, ValueError):
                    confidence = 0.6
                results.append(
                    self.remember_semantic(
                        concept=entity,
                        predicate=str(item.get("predicate") or "fact"),
                        value=value,
                        aspect=str(item.get("aspect") or "knowledge"),
                        confidence=confidence,
                        source={"worker": "consolidation", "phase": "semantic"},
                    )
                )
            for episode in batch:
                consumed[str(episode.path)] = episode.path
        for path in consumed.values():
            self.vault.mark_consolidated(path)
        return results

    def consolidate_reflective(
        self,
        *,
        engine: ConsolidationEngine | None = None,
        min_facts: int = 3,
    ) -> list[MemoryWriteResult]:
        """Phase 2 consolidation: synthesize reflective insights from facts.

        Groups all non-deprecated semantic facts by aspect, asks the
        consolidation engine to synthesize cross-fact insights for each aspect
        with at least ``min_facts`` facts, deduplicates against existing
        reflections by title, and persists each insight as a reflective note
        linked back to the entities that sourced it (provenance).
        """
        engine = engine or ConsolidationEngine()
        seen = {
            note.metadata.title.casefold() for note in self.vault.list_notes(MemoryKind.REFLECTIVE)
        }
        results: list[MemoryWriteResult] = []
        for aspect, facts in self._facts_by_aspect().items():
            if len(facts) < min_facts:
                continue
            sources = sorted({fact.subject for fact in facts})
            for item in engine.synthesize_insights(aspect, [fact.value for fact in facts]):
                title = str(item.get("title") or "").strip()
                content = str(item.get("insight") or "").strip()
                if not title or not content or title.casefold() in seen:
                    continue
                seen.add(title.casefold())
                tags = [aspect]
                itype = str(item.get("type") or "").strip()
                if itype:
                    tags.append(itype)
                topics = item.get("topics")
                if isinstance(topics, list):
                    tags.extend(str(topic) for topic in topics)
                results.append(
                    self.reflect(
                        title=title,
                        insight=content,
                        tags=tags,
                        links=sources,
                        source={
                            "worker": "consolidation",
                            "phase": "reflective",
                            "aspect": aspect,
                        },
                    )
                )
        return results

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

        procedural_limit = max(2, limit // 2)
        semantic_limit = max(2, limit // 2)
        reflective_limit = max(1, limit // 3)
        # Pool size needs to cover all three kinds; over-fetch generously so
        # under-represented kinds still surface a few hits after partitioning.
        pool_limit = max(
            procedural_limit + semantic_limit + reflective_limit,
            3 * limit,
            20,
        )

        # Single sync, single query embedding, single first-stage + rerank pass.
        self.sync()
        query_embedding = self.embedder.embed_text(query)
        pool = self._search_pool(
            query=query,
            query_embedding=query_embedding,
            kind=None,
            limit=pool_limit,
        )

        procedural = [h for h in pool if h.kind == MemoryKind.PROCEDURAL][:procedural_limit]
        semantic = [h for h in pool if h.kind == MemoryKind.SEMANTIC][:semantic_limit]
        reflective = [h for h in pool if h.kind == MemoryKind.REFLECTIVE][:reflective_limit]
        # Bypass `self.recent()` so we don't pay for a second sync on the
        # same already-current vault.
        recent_episodes = self.index.recent(limit=max(2, limit // 3), kind=MemoryKind.EPISODIC)
        guidance = self._derive_guidance(procedural, semantic, reflective)
        # K-line activation: surface the best-matching template's plan so the
        # procedural strategy actually reaches the session, not just the notes.
        top_templates = self.match_template(query, top_k=1)
        if top_templates:
            top = top_templates[0]
            guidance.insert(0, f"Template '{top.template.title}': {' '.join(top.template.plan)}")

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
        source: dict[str, object] | None,
        facts: list[SemanticFact] | None = None,
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
            facts=facts,
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

    def _facts_by_aspect(self) -> dict[str, list[SemanticFact]]:
        """Non-deprecated semantic facts across the vault, grouped by aspect."""
        grouped: dict[str, list[SemanticFact]] = {}
        for note in self.vault.list_notes(MemoryKind.SEMANTIC):
            for fact in note.metadata.facts:
                if fact.deprecated:
                    continue
                grouped.setdefault(fact.aspect.value, []).append(fact)
        return grouped

    def _entity_facts(self, entity: str) -> list[SemanticFact]:
        """Non-deprecated facts stored about a single entity (matched by title)."""
        note = self.vault.find_note(title=entity, kind=MemoryKind.SEMANTIC)
        if note is None:
            return []
        return [fact for fact in note.metadata.facts if not fact.deprecated]

    def _semantic_facts_from_existing(
        self, existing: NoteRecord | None, *, concept: str
    ) -> list[SemanticFact]:
        if existing is None:
            return []
        if existing.metadata.facts:
            return [fact.model_copy(deep=True) for fact in existing.metadata.facts]
        migrated: list[SemanticFact] = []
        for bullet in self._extract_bullets(existing.body, section_name="Facts"):
            if bullet.strip():
                migrated.append(
                    SemanticFact(
                        subject=concept,
                        predicate="fact",
                        value=bullet.strip(),
                        source={"migration": "legacy-facts-section"},
                    )
                )
        return migrated

    def _render_semantic_document(self, facts: list[SemanticFact]) -> str:
        active = [fact for fact in facts if not fact.deprecated]
        deprecated = [fact for fact in facts if fact.deprecated]
        sections: list[str] = []
        active_lines = [f"- {self._format_fact(fact)}" for fact in active]
        if active_lines:
            sections.append("## Facts\n" + "\n".join(active_lines))
        deprecated_lines = [
            f"- ~~{self._format_fact(fact)}~~ {self._format_deprecation(fact)}"
            for fact in deprecated
        ]
        if deprecated_lines:
            sections.append("## Deprecated Facts\n" + "\n".join(deprecated_lines))
        return "\n\n".join(sections).strip()

    def _format_fact(self, fact: SemanticFact) -> str:
        if self._normalize_fact_predicate(fact.predicate) == "fact":
            return fact.value
        return f"{fact.predicate}: {fact.value}"

    def _format_deprecation(self, fact: SemanticFact) -> str:
        parts: list[str] = []
        if fact.deprecated_reason:
            parts.append(f"[DEPRECATED: {fact.deprecated_reason}]")
        else:
            parts.append("[DEPRECATED]")
        if fact.replaced_by:
            parts.append(f"(replaced_by: {fact.replaced_by})")
        return " ".join(parts)

    def _coerce_fact_aspect(self, aspect: FactAspect | str) -> FactAspect:
        if isinstance(aspect, FactAspect):
            return aspect
        try:
            return FactAspect(str(aspect).strip().lower())
        except ValueError:
            return FactAspect.KNOWLEDGE

    def _normalize_fact_predicate(self, value: str) -> str:
        return " ".join(value.casefold().strip().split())

    def _normalize_fact_value(self, value: str) -> str:
        return " ".join(value.casefold().strip().split())

    def _split_fact(self, fact: str) -> tuple[str, str]:
        if ":" in fact:
            concept, text = fact.split(":", 1)
            if concept.strip() and text.strip():
                return concept.strip(), text.strip()
        return "Learned Fact", fact.strip()

    def _content_hash(self, note: NoteRecord) -> str:
        from hashlib import sha256

        return sha256(note.raw_text.encode("utf-8")).hexdigest()
