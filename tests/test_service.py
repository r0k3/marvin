from pathlib import Path

from marvin.config import MarvinSettings
from marvin.models import MemoryKind, SearchHit
from marvin.reranker import RerankerService
from marvin.service import MarvinService


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


class _FakeReranker(RerankerService):
    """Deterministic reranker that scores by presence of a marker word.

    We use this to verify service-level wiring without loading a real
    cross-encoder in tests: the document containing ``marker`` always wins.
    """

    def __init__(self, marker: str) -> None:
        super().__init__(provider="none")
        self._marker = marker
        self._backend_name = "fake"

    def score(self, query: str, documents):
        return [1.0 if self._marker in doc else 0.0 for doc in documents]


def test_service_search_rerank_reorders_and_scrubs(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
        rerank_enabled=True,
        rerank_depth=5,
    )
    service = MarvinService(settings)
    service.reranker = _FakeReranker(marker="KANGAROO_MARKER")

    service.remember_semantic(
        concept="Marsupial Facts",
        content="A wombat is a burrowing marsupial KANGAROO_MARKER.",
    )
    service.remember_semantic(
        concept="Python Facts",
        content="Python is a popular language used worldwide.",
    )
    service.remember_semantic(
        concept="Paris Facts",
        content="Paris is the capital of France and a global city.",
    )

    hits = service.search("tell me about marsupials", limit=3)

    # The document carrying the marker must win under the fake reranker,
    # regardless of what the hybrid first stage thought.
    assert hits[0].title == "Marsupial Facts"
    # The score field must reflect the reranker's score (1.0 for winner).
    assert hits[0].score == 1.0
    # chunk_text is an internal plumbing field and must be scrubbed
    # before reaching API consumers.
    assert all(h.chunk_text is None for h in hits)

    service.close()


def test_service_search_rerank_disabled_by_default(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    service = MarvinService(settings)
    assert settings.rerank_enabled is False
    # Plain no-op service uses the noop backend.
    assert service.reranker.backend_name == "noop"
    service.close()


class TestServiceHealth:
    def test_reports_provider_before_load(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
        service = MarvinService(settings)
        try:
            payload = service.health()
            assert payload["embedding_backend"] == "hash (not loaded)"
            assert payload["embedding_provider"] == "hash"
            assert payload["rerank_enabled"] is False
            assert payload["reranker_backend"] == "disabled"
            assert payload["vault_path"].endswith("vault")
            assert payload["index_path"].endswith("marvin.db")
        finally:
            service.close()

    def test_reports_loaded_backend_after_use(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
        service = MarvinService(settings)
        try:
            service.embedder.embed_text("warm up")
            payload = service.health()
            assert payload["embedding_backend"] == "hash"
        finally:
            service.close()

    def test_reports_reranker_loaded_state(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
            rerank_enabled=True,
        )
        service = MarvinService(settings)
        try:
            payload_before = service.health()
            assert payload_before["reranker_backend"] == "auto (not loaded)"
            service.reranker._backend_name = "fake"
            payload_after = service.health()
            assert payload_after["reranker_backend"] == "fake"
        finally:
            service.close()

    def test_payload_keys_and_types(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
        service = MarvinService(settings)
        try:
            payload = service.health()
            for key in (
                "embedding_backend",
                "embedding_provider",
                "embedding_model",
                "embedding_dimensions",
                "rerank_enabled",
                "rerank_provider",
                "rerank_model",
                "rerank_depth",
                "reranker_backend",
                "vault_path",
                "index_path",
            ):
                assert key in payload, key
            assert isinstance(payload["embedding_dimensions"], int)
            assert isinstance(payload["rerank_enabled"], bool)
            assert isinstance(payload["rerank_depth"], int)
        finally:
            service.close()


def test_hybrid_search_include_chunk_text_flag(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    service = MarvinService(settings)
    service.remember_semantic(concept="Quokka", content="Small marsupial.")
    service.sync()

    query_embedding = service.embedder.embed_text("quokka")
    plain = service.index.hybrid_search(
        query="quokka", query_embedding=query_embedding, limit=3
    )
    rich = service.index.hybrid_search(
        query="quokka",
        query_embedding=query_embedding,
        limit=3,
        include_chunk_text=True,
    )
    assert plain and rich
    assert all(isinstance(h, SearchHit) for h in plain + rich)
    assert all(h.chunk_text is None for h in plain)
    assert all(h.chunk_text is not None and h.chunk_text for h in rich)

    service.close()
