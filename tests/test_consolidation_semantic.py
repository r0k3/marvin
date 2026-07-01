"""Phase 1 consolidation: episodic -> semantic, entity-scoped.

Covers the per-entity threshold, known-fact dedup passing, persistence, and that
only consumed episodes are marked consolidated. A stub engine stands in for the
LLM so the orchestration is deterministic.
"""

from __future__ import annotations

from pathlib import Path

from marvin.config import MarvinSettings
from marvin.models import MemoryKind
from marvin.service import MarvinService


class StubEngine:
    """Records calls and returns canned facts per entity."""

    def __init__(self, facts_by_entity: dict[str, list[dict]] | None = None):
        self.facts_by_entity = facts_by_entity or {}
        self.calls: list[tuple[str, list[str], list[str]]] = []

    def extract_entity_facts(
        self, entity: str, episodes: list[str], known_facts: list[str] | None = None
    ) -> list[dict]:
        self.calls.append((entity, list(episodes), list(known_facts or [])))
        return list(self.facts_by_entity.get(entity, []))


def _service(tmp_path: Path) -> MarvinService:
    return MarvinService(
        MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
    )


def test_entity_scoped_extraction_persists_and_marks(tmp_path: Path):
    service = _service(tmp_path)
    try:
        for i in range(3):
            service.log_episode(title=f"S{i}", summary=f"session {i} about [[XR-7291]]")
        stub = StubEngine(
            {
                "XR-7291": [
                    {
                        "predicate": "targets",
                        "value": "neural pathway regeneration",
                        "aspect": "knowledge",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        results = service.consolidate_semantic(engine=stub, min_episodes=3)

        assert len(results) == 1
        note = service.vault.find_note(title="XR-7291", kind=MemoryKind.SEMANTIC)
        assert note is not None
        assert any(f.value == "neural pathway regeneration" for f in note.metadata.facts)
        # all three episodes consumed -> consolidated
        episodes = service.vault.list_notes(MemoryKind.EPISODIC)
        assert episodes and all(ep.metadata.consolidated for ep in episodes)
        # engine called once for the entity with all three episode bodies
        assert len(stub.calls) == 1
        entity, bodies, _known = stub.calls[0]
        assert entity == "XR-7291"
        assert len(bodies) == 3
    finally:
        service.close()


def test_below_threshold_is_skipped_and_left_pending(tmp_path: Path):
    service = _service(tmp_path)
    try:
        for i in range(2):  # only two mentions -> below the default 3
            service.log_episode(title=f"S{i}", summary=f"about [[YB-1234]] {i}")
        stub = StubEngine({"YB-1234": [{"predicate": "p", "value": "v", "aspect": "knowledge"}]})

        results = service.consolidate_semantic(engine=stub, min_episodes=3)

        assert results == []
        assert stub.calls == []  # never invoked
        episodes = service.vault.list_notes(MemoryKind.EPISODIC)
        assert episodes and all(not ep.metadata.consolidated for ep in episodes)
    finally:
        service.close()


def test_known_facts_are_passed_for_dedup(tmp_path: Path):
    service = _service(tmp_path)
    try:
        service.remember_semantic(
            concept="XR-7291", predicate="targets", value="neural regeneration", aspect="knowledge"
        )
        for i in range(3):
            service.log_episode(title=f"S{i}", summary=f"more on [[XR-7291]] {i}")
        stub = StubEngine({})  # returns nothing

        service.consolidate_semantic(engine=stub, min_episodes=3)

        assert len(stub.calls) == 1
        entity, _bodies, known = stub.calls[0]
        assert entity == "XR-7291"
        assert "neural regeneration" in known
    finally:
        service.close()


def test_marks_only_consumed_episodes(tmp_path: Path):
    service = _service(tmp_path)
    try:
        for i in range(3):  # Alpha crosses the threshold
            service.log_episode(title=f"A{i}", summary=f"a {i} about [[Alpha]]")
        for i in range(2):  # Beta stays below it
            service.log_episode(title=f"B{i}", summary=f"b {i} about [[Beta]]")
        stub = StubEngine(
            {"Alpha": [{"predicate": "p", "value": "alpha fact", "aspect": "knowledge"}]}
        )

        service.consolidate_semantic(engine=stub, min_episodes=3)

        consolidated = {
            ep.metadata.title: ep.metadata.consolidated
            for ep in service.vault.list_notes(MemoryKind.EPISODIC)
        }
        assert consolidated["A0"] and consolidated["A1"] and consolidated["A2"]
        assert not consolidated["B0"] and not consolidated["B1"]
    finally:
        service.close()
