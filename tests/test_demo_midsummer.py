"""Regression test for the Midsummer Night's Dream demo vault.

This guards the *illustrative* demo (``examples/demo_vault``). Note what it
asserts: that the right **notes are retrieved** from Marvin's memory for each
question -- i.e. the memory/retrieval system works -- NOT that a language model
*answers* correctly. The latter could lean on the LLM's baked-in knowledge of
Shakespeare; the rigorous, leakage-free claims come from synthetic-domain
experiments with invented entities. The demo is here to make the architecture
tangible, and this test keeps it honest and green.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from marvin.config import MarvinSettings
from marvin.eval.demo import DEMO_QUERIES, demo_vault_path
from marvin.models import MemoryKind
from marvin.service import MarvinService


def _demo_service(tmp_path: Path) -> MarvinService:
    """Index a hermetic copy of the demo vault under a deterministic embedder."""
    vault = tmp_path / "vault"
    shutil.copytree(demo_vault_path(), vault)
    service = MarvinService(
        MarvinSettings(
            vault_path=vault,
            state_dir=tmp_path / ".state",
            embedding_provider="hash",  # deterministic; queries have lexical overlap
        )
    )
    service.sync()
    return service


def test_demo_vault_spans_all_four_memory_kinds(tmp_path: Path):
    service = _demo_service(tmp_path)
    try:
        notes = service.vault.list_notes()
        kinds = {n.metadata.kind for n in notes}
        assert kinds == set(MemoryKind), f"missing kinds: {set(MemoryKind) - kinds}"
        assert len(notes) >= 14
    finally:
        service.close()


@pytest.mark.parametrize("question,expected_title,kind", DEMO_QUERIES)
def test_demo_query_retrieves_expected_note(
    tmp_path: Path, question: str, expected_title: str, kind: MemoryKind
):
    service = _demo_service(tmp_path)
    try:
        hits = service.search(question, limit=6)
        titles = [h.title for h in hits]
        assert expected_title in titles, f"{expected_title!r} not retrieved; got {titles}"
        # the question is answered from the correct memory kind, not just any hit
        assert kind.value in {h.kind.value for h in hits}
    finally:
        service.close()


def test_entity_graph_surfaces_connected_puck_cluster(tmp_path: Path):
    """Querying an entity surfaces its connected notes across memory kinds."""
    service = _demo_service(tmp_path)
    try:
        hits = service.search("Puck", limit=8)
        kinds = {h.kind for h in hits}
        assert len(hits) >= 3
        # Puck appears in both a recorded scene (episodic) and distilled facts (semantic)
        assert MemoryKind.EPISODIC in kinds
        assert MemoryKind.SEMANTIC in kinds
    finally:
        service.close()
