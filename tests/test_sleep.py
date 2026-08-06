"""The sleep pass: vault-as-queue extraction + two-phase composition.

The vault's ``extracted`` flag is the durable work queue — no broker or
worker involved. A stub engine stands in for the LLM throughout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marvin.config import MarvinSettings
from marvin.service import MarvinService


class NullEngine:
    def extract_entity_facts(self, entity, episodes, known_facts=None):
        return []

    def synthesize_insights(self, aspect, facts):
        return []


def _service(tmp_path: Path) -> MarvinService:
    return MarvinService(
        MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
    )


def test_extract_note_links_and_marks(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("marvin.extraction.extract_entities", lambda text: ["Postgres"])
    service = _service(tmp_path)
    try:
        service.log_episode(title="DB work", summary="Set up Postgres with asyncpg.")
        note = service.vault.unextracted_notes()[0]
        assert service.extract_note(note)

        again = service.vault.read_note(note.path)
        assert "[[Postgres]]" in again.body
        assert "Postgres" in again.metadata.links
        assert again.metadata.extracted
    finally:
        service.close()


def test_extract_pending_processes_each_note_once(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_extract(text: str) -> list[str]:
        calls.append(text)
        return []

    monkeypatch.setattr("marvin.extraction.extract_entities", fake_extract)
    service = _service(tmp_path)
    try:
        service.log_episode(title="A", summary="alpha")
        service.log_episode(title="B", summary="beta")

        assert service.extract_pending() == 0  # no entities -> no body changes
        assert len(calls) == 2
        assert service.vault.unextracted_notes() == []

        service.extract_pending()
        assert len(calls) == 2  # the queue drained; nothing re-processed
    finally:
        service.close()


def test_sleep_skips_extraction_without_langextract(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("marvin.extraction.LANGEXTRACT_AVAILABLE", False)
    service = _service(tmp_path)
    try:
        service.log_episode(title="A", summary="alpha")
        report = service.sleep(engine=NullEngine())

        assert report.extraction_skipped
        assert report.notes_linked == 0
        assert service.vault.unextracted_notes()  # queue untouched
    finally:
        service.close()


def test_sleep_runs_extraction_then_consolidation(tmp_path: Path, monkeypatch):
    """Extraction's injected wikilinks must drive phase 1's entity grouping."""
    monkeypatch.setattr("marvin.extraction.LANGEXTRACT_AVAILABLE", True)
    monkeypatch.setattr("marvin.extraction.extract_entities", lambda text: ["XR-9"])

    class Engine(NullEngine):
        def extract_entity_facts(self, entity, episodes, known_facts=None):
            if entity == "XR-9":
                return [
                    {
                        "predicate": "status",
                        "value": "ready",
                        "aspect": "knowledge",
                        "confidence": 0.9,
                    }
                ]
            return []

    service = _service(tmp_path)
    try:
        for i in range(3):
            service.log_episode(title=f"S{i}", summary=f"session {i} touching XR-9 again")

        report = service.sleep(engine=Engine())

        assert not report.extraction_skipped
        assert report.notes_linked == 3  # every episode body gained [[XR-9]]
        assert len(report.facts) == 1  # the injected links crossed the threshold
    finally:
        service.close()


def test_missing_consolidate_extra_raises_actionable_error(monkeypatch):
    import marvin.consolidation as consolidation

    monkeypatch.setattr(consolidation, "completion", None)
    engine = consolidation.ConsolidationEngine()
    with pytest.raises(RuntimeError, match=r"marvin-memory\[consolidate\]"):
        engine.synthesize_insights("decision", ["a"])
