"""Tests for vault/index maintenance.

- episodic notes carry a ``consolidated`` flag so consolidation never
  re-extracts the same episodes;
- the derived index can be rebuilt from the authoritative Markdown vault, and
  a consistency check detects drift between the two.
"""

from __future__ import annotations

from pathlib import Path

from marvin.config import MarvinSettings
from marvin.models import MemoryKind
from marvin.service import MarvinService
from marvin.vault import VaultStore

# --------------------------------------------------------------------------
# consolidated flag (vault layer)
# --------------------------------------------------------------------------


def test_consolidated_flag_round_trips(tmp_path: Path):
    store = VaultStore(tmp_path)
    path, _ = store.write_note(kind=MemoryKind.EPISODIC, title="Session 1", body="user: hi")
    # Fresh episode is not consolidated, and the key isn't written until true.
    assert store.read_note(path).metadata.consolidated is False
    assert "consolidated" not in path.read_text(encoding="utf-8")

    store.mark_consolidated(path)
    note = store.read_note(path)
    assert note.metadata.consolidated is True
    assert "consolidated: true" in path.read_text(encoding="utf-8")


def test_mark_consolidated_preserves_body_and_tags(tmp_path: Path):
    store = VaultStore(tmp_path)
    path, _ = store.write_note(
        kind=MemoryKind.EPISODIC, title="E", body="user: detailed turn content", tags=["chat"]
    )
    store.mark_consolidated(path)
    note = store.read_note(path)
    assert note.metadata.consolidated is True
    assert "detailed turn content" in note.body
    assert "chat" in note.metadata.tags


def test_unconsolidated_episodes_filters(tmp_path: Path):
    store = VaultStore(tmp_path)
    p1, _ = store.write_note(kind=MemoryKind.EPISODIC, title="E1", body="a")
    store.write_note(kind=MemoryKind.EPISODIC, title="E2", body="b")
    store.mark_consolidated(p1)
    pending = {n.metadata.title for n in store.unconsolidated_episodes()}
    assert pending == {"E2"}


def _service(tmp_path: Path) -> MarvinService:
    return MarvinService(
        MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
    )


# --------------------------------------------------------------------------
# rebuild + consistency-check (service/index layer)
# --------------------------------------------------------------------------


def test_rebuild_reconstructs_index_from_vault(tmp_path: Path):
    service = _service(tmp_path)
    try:
        service.remember_semantic(concept="Apple", content="Apple makes the iPhone.")
        service.remember_semantic(concept="Banana", content="Bananas are yellow fruit.")
        service.sync()
        assert service.index.note_count() == 2
        assert any(h.title == "Apple" for h in service.search("iPhone", limit=5))

        # Simulate index corruption / loss.
        service.index.clear()
        assert service.index.note_count() == 0

        # Rebuild purely from the Markdown source of truth.
        service.rebuild()
        assert service.index.note_count() == 2
        assert any(h.title == "Apple" for h in service.search("iPhone", limit=5))
    finally:
        service.close()


def test_consistency_check_clean_then_detects_drift(tmp_path: Path):
    service = _service(tmp_path)
    try:
        service.remember_semantic(concept="Apple", content="Apple makes the iPhone.")
        service.sync()
        rep = service.consistency_check()
        assert rep.consistent
        assert rep.vault_notes == rep.indexed_notes == 1

        # Drift A: a vault note not yet indexed -> missing_from_index.
        service.vault.write_note(kind=MemoryKind.SEMANTIC, title="Orphan", body="not indexed")
        rep_a = service.consistency_check()
        assert not rep_a.consistent
        assert any("Orphan" in p for p in rep_a.missing_from_index)

        # Drift B: delete a vault file without pruning -> orphaned_in_index.
        service.sync()  # index Orphan so only the deletion drives the drift
        apple = service.vault.find_note(title="Apple", kind=MemoryKind.SEMANTIC)
        assert apple is not None
        apple.path.unlink()
        rep_b = service.consistency_check()
        assert not rep_b.consistent
        assert any("Apple" in p for p in rep_b.orphaned_in_index)
    finally:
        service.close()
