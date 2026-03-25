from pathlib import Path
from marvin.config import MarvinSettings
from marvin.service import MarvinService
from marvin.models import MemoryKind


def test_service_basic_flow(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",  # use lightweight hashing for tests
    )
    service = MarvinService(settings)

    # Write a note
    res1 = service.remember_semantic(
        concept="Python Logging",
        content="Always use standard lib logging or loguru.",
        tags=["python", "best-practice"],
    )
    assert res1.created

    service.store_procedure(
        title="Logging Rule",
        steps=["Import logging", "Set level to INFO", "Log away"],
        tags=["python"],
    )

    # Writing already triggers an index update internally in `_write_note`
    # So a subsequent sync() will report 0 indexed items because they are already indexed
    # and their content hashes match. Let's write directly to vault to test sync.

    service.vault.write_note(
        kind=MemoryKind.SEMANTIC,
        title="Direct Note",
        body="This is directly written to the vault",
    )

    # Sync triggers index
    report = service.sync()
    assert report.indexed == 1
    assert report.scanned == 3

    # Search
    hits = service.search("how to log in python", limit=5)
    assert len(hits) >= 2

    # Verify RRF sorting (exact order may vary but both should be found)
    titles = [h.title for h in hits]
    assert "Python Logging" in titles
    assert "Logging Rule" in titles

    # Get recent
    recent = service.recent(kind=MemoryKind.SEMANTIC)
    assert len(recent) == 2
    assert "Python Logging" in [r.title for r in recent]

    service.close()
