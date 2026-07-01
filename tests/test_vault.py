from pathlib import Path

from marvin.models import FactAspect, MemoryKind, SemanticFact
from marvin.vault import VaultStore


def test_vault_creates_directories(tmp_path: Path):
    store = VaultStore(tmp_path)
    assert store.vault_path.exists()
    assert (store.vault_path / "Episodic").exists()
    assert (store.vault_path / "Semantic").exists()
    assert (store.vault_path / "Procedural").exists()
    assert (store.vault_path / "Reflective").exists()


def test_write_and_read_note(tmp_path: Path):
    store = VaultStore(tmp_path)
    path, created = store.write_note(
        kind=MemoryKind.SEMANTIC,
        title="Test Concept",
        body="This is a test fact.",
        tags=["#test", "fact"],
        links=["Related Concept"],
    )
    assert created
    assert path.exists()

    note = store.read_note(path)
    assert note.metadata.title == "Test Concept"
    assert note.metadata.kind == MemoryKind.SEMANTIC
    assert "test" in note.metadata.tags
    assert "fact" in note.metadata.tags
    assert "Related Concept" in note.metadata.links
    assert "This is a test fact" in note.body
    assert "[[Related Concept]]" in note.raw_text


def test_semantic_facts_round_trip_in_frontmatter(tmp_path: Path):
    store = VaultStore(tmp_path)
    fact = SemanticFact(
        subject="Test Concept",
        predicate="storage",
        value="Facts are canonical in Markdown frontmatter.",
        aspect=FactAspect.DECISION,
        confidence=0.9,
        source={"test": "vault"},
    )
    path, created = store.write_note(
        kind=MemoryKind.SEMANTIC,
        title="Test Concept",
        body="## Facts\n- storage: Facts are canonical in Markdown frontmatter.",
        facts=[fact],
    )
    assert created

    note = store.read_note(path)
    assert len(note.metadata.facts) == 1
    restored = note.metadata.facts[0]
    assert restored.id == fact.id
    assert restored.aspect == FactAspect.DECISION
    assert restored.confidence == 0.9
    assert restored.source == {"test": "vault"}

    store.write_note(
        kind=MemoryKind.SEMANTIC,
        title="Test Concept",
        body=note.body,
        existing_path=path,
    )
    rewritten = store.read_note(path)
    assert [f.id for f in rewritten.metadata.facts] == [fact.id]


def test_list_and_find_notes(tmp_path: Path):
    store = VaultStore(tmp_path)
    store.write_note(kind=MemoryKind.SEMANTIC, title="Doc 1", body="1")
    store.write_note(kind=MemoryKind.EPISODIC, title="Doc 2", body="2")

    notes = store.list_notes()
    assert len(notes) == 2

    semantic_notes = store.list_notes(kind=MemoryKind.SEMANTIC)
    assert len(semantic_notes) == 1
    assert semantic_notes[0].metadata.title == "Doc 1"

    found = store.find_note(title="doc 2", kind=MemoryKind.EPISODIC)
    assert found is not None
    assert found.metadata.title == "Doc 2"
