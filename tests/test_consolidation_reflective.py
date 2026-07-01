"""Phase 2 consolidation: semantic -> reflective synthesis.

Covers grouping facts by aspect, the threshold, dedup against existing
reflections, and provenance linking. A stub engine stands in for the LLM so the
orchestration is tested deterministically without a model.
"""

from __future__ import annotations

from pathlib import Path

from marvin.config import MarvinSettings
from marvin.consolidation import ConsolidationEngine
from marvin.models import MemoryKind
from marvin.service import MarvinService


class StubEngine:
    """Records calls and returns canned insights per aspect."""

    def __init__(self, insights_by_aspect: dict[str, list[dict]] | None = None):
        self.insights_by_aspect = insights_by_aspect or {}
        self.calls: list[tuple[str, list[str]]] = []

    def synthesize_insights(self, aspect: str, facts: list[str]) -> list[dict]:
        self.calls.append((aspect, list(facts)))
        return self.insights_by_aspect.get(aspect, [])


def _service(tmp_path: Path) -> MarvinService:
    return MarvinService(
        MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
    )


def _seed_problem_facts(service: MarvinService) -> None:
    """Three 'problem' facts across distinct entities (so none deprecate)."""
    service.remember_semantic(
        concept="Auth", predicate="issue", value="login fails on Safari", aspect="problem"
    )
    service.remember_semantic(
        concept="DB", predicate="issue", value="pool exhausts under load", aspect="problem"
    )
    service.remember_semantic(
        concept="API", predicate="issue", value="rate-limit errors at peak", aspect="problem"
    )


def test_facts_by_aspect_groups_and_excludes_deprecated(tmp_path: Path):
    service = _service(tmp_path)
    try:
        _seed_problem_facts(service)
        service.remember_semantic(
            concept="Style", predicate="pref", value="prefers dark mode", aspect="preference"
        )
        # Overwrite a fact on the same predicate -> old one deprecated, excluded.
        service.remember_semantic(
            concept="Auth", predicate="issue", value="login fails on all browsers", aspect="problem"
        )

        grouped = service._facts_by_aspect()
        assert len(grouped["problem"]) == 3  # 3 entities; Auth's old fact deprecated
        assert {f.subject for f in grouped["problem"]} == {"Auth", "DB", "API"}
        assert len(grouped["preference"]) == 1
        assert all(not f.deprecated for facts in grouped.values() for f in facts)
    finally:
        service.close()


def test_consolidate_reflective_synthesizes_links_and_tags(tmp_path: Path):
    service = _service(tmp_path)
    try:
        _seed_problem_facts(service)
        stub = StubEngine(
            {
                "problem": [
                    {
                        "title": "Recurring reliability issues",
                        "insight": "Multiple subsystems fail under load.",
                        "type": "pattern",
                        "topics": ["reliability"],
                    }
                ]
            }
        )
        results = service.consolidate_reflective(engine=stub, min_facts=3)

        assert len(results) == 1
        assert len(stub.calls) == 1
        aspect, values = stub.calls[0]
        assert aspect == "problem"
        assert set(values) == {
            "login fails on Safari",
            "pool exhausts under load",
            "rate-limit errors at peak",
        }
        note = service.vault.find_note(
            title="Recurring reliability issues", kind=MemoryKind.REFLECTIVE
        )
        assert note is not None
        # provenance: linked back to the entities that sourced the facts
        assert {"Auth", "DB", "API"} <= set(note.metadata.links)
        # classification carried as tags: aspect + type + topics
        assert {"problem", "pattern", "reliability"} <= set(note.metadata.tags)
        assert "Multiple subsystems fail under load." in note.body
    finally:
        service.close()


def test_consolidate_reflective_respects_threshold(tmp_path: Path):
    service = _service(tmp_path)
    try:
        # Only two 'goal' facts -> below the default threshold of 3.
        service.remember_semantic(concept="Q3", predicate="goal", value="ship beta", aspect="goal")
        service.remember_semantic(
            concept="Q4", predicate="goal", value="reach 1k users", aspect="goal"
        )
        stub = StubEngine({"goal": [{"title": "Roadmap", "insight": "x", "type": "pattern"}]})

        results = service.consolidate_reflective(engine=stub, min_facts=3)

        assert results == []
        assert "goal" not in [aspect for aspect, _ in stub.calls]  # skipped before calling
    finally:
        service.close()


def test_consolidate_reflective_dedups_existing_title(tmp_path: Path):
    service = _service(tmp_path)
    try:
        _seed_problem_facts(service)
        service.reflect(title="Recurring reliability issues", insight="ORIGINAL")
        stub = StubEngine(
            {
                "problem": [
                    {
                        "title": "Recurring reliability issues",
                        "insight": "NEW SYNTHESIZED",
                        "type": "pattern",
                    }
                ]
            }
        )

        results = service.consolidate_reflective(engine=stub, min_facts=3)

        assert results == []  # title already exists -> skipped, not overwritten
        note = service.vault.find_note(
            title="Recurring reliability issues", kind=MemoryKind.REFLECTIVE
        )
        assert "ORIGINAL" in note.body
        assert "NEW SYNTHESIZED" not in note.body
    finally:
        service.close()


def test_synthesize_insights_empty_returns_empty():
    # No LLM call for empty input -- safe to run without a model.
    assert ConsolidationEngine().synthesize_insights("knowledge", []) == []
