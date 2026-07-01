from pathlib import Path
from unittest.mock import patch

from marvin.config import MarvinSettings
from marvin.models import FactAspect, MemoryKind, SearchHit
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


def test_remember_semantic_writes_structured_fact(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    service = MarvinService(settings)
    try:
        service.remember_semantic(
            concept="Database Configuration",
            predicate="storage",
            value="The project uses PostgreSQL with asyncpg.",
            aspect=FactAspect.DECISION,
            confidence=0.8,
            source={"test": "structured"},
        )

        note = service.vault.find_note(title="Database Configuration", kind=MemoryKind.SEMANTIC)
        assert note is not None
        assert len(note.metadata.facts) == 1
        fact = note.metadata.facts[0]
        assert fact.predicate == "storage"
        assert fact.value == "The project uses PostgreSQL with asyncpg."
        assert fact.aspect == FactAspect.DECISION
        assert fact.confidence == 0.8
        assert fact.source == {"test": "structured"}
        assert "- storage: The project uses PostgreSQL with asyncpg." in note.body
    finally:
        service.close()


def test_remember_semantic_deduplicates_exact_active_fact(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    service = MarvinService(settings)
    try:
        for _ in range(2):
            service.remember_semantic(
                concept="Runtime",
                predicate="python",
                value="Use Python 3.12 or newer.",
            )

        note = service.vault.find_note(title="Runtime", kind=MemoryKind.SEMANTIC)
        assert note is not None
        assert len(note.metadata.facts) == 1
        assert note.metadata.facts[0].value == "Use Python 3.12 or newer."
    finally:
        service.close()


def test_remember_semantic_deprecates_same_predicate_update(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    service = MarvinService(settings)
    try:
        service.remember_semantic(
            concept="Project Owner",
            predicate="owner",
            value="Alice owns the project.",
        )
        service.remember_semantic(
            concept="Project Owner",
            predicate="owner",
            value="Bob owns the project.",
        )

        note = service.vault.find_note(title="Project Owner", kind=MemoryKind.SEMANTIC)
        assert note is not None
        facts = note.metadata.facts
        assert len(facts) == 2
        old, new = facts
        assert old.deprecated is True
        assert old.replaced_by == new.id
        assert new.deprecated is False
        assert "## Deprecated Facts" in note.body
        assert "~~owner: Alice owns the project.~~" in note.body
    finally:
        service.close()


def test_remember_semantic_lazily_migrates_legacy_facts(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    service = MarvinService(settings)
    try:
        legacy_path, _ = service.vault.write_note(
            kind=MemoryKind.SEMANTIC,
            title="Legacy Note",
            body="## Facts\n- Existing legacy bullet.",
        )
        legacy = service.vault.read_note(legacy_path)
        assert legacy.metadata.facts == []

        service.remember_semantic(
            concept="Legacy Note",
            predicate="new_fact",
            value="Structured addition.",
        )

        note = service.vault.read_note(legacy_path)
        values = [fact.value for fact in note.metadata.facts]
        assert values == ["Existing legacy bullet.", "Structured addition."]
        assert note.metadata.facts[0].source == {"migration": "legacy-facts-section"}
    finally:
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


class TestPrepareSessionUnified:
    """``prepare_session`` must run one sync, one embed, one rerank pass."""

    def _seed(self, service: MarvinService) -> None:
        service.remember_semantic(
            concept="Logging",
            content="Use stdlib logging in Python projects.",
        )
        service.store_procedure(
            title="Bootstrap a Python repo",
            steps=["init git", "uv init", "add tests"],
        )
        service.reflect(
            title="Avoid ad-hoc print statements",
            insight="Switch to logging early; print breaks under MCP transports.",
        )
        service.log_episode(
            title="Recent debugging episode",
            summary="Tracked a flaky test for an hour.",
        )

    def test_returns_per_kind_buckets(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
        service = MarvinService(settings)
        try:
            self._seed(service)
            ctx = service.prepare_session(
                task="how do I structure logging in a new Python project?",
                limit=6,
            )
            kinds = {hit.kind for hit in ctx.procedural}
            assert kinds <= {MemoryKind.PROCEDURAL}
            assert {hit.kind for hit in ctx.semantic} <= {MemoryKind.SEMANTIC}
            assert {hit.kind for hit in ctx.reflective} <= {MemoryKind.REFLECTIVE}
            assert any(hit.kind == MemoryKind.EPISODIC for hit in ctx.recent_episodes)
            assert ctx.task.startswith("how do I structure logging")
        finally:
            service.close()

    def test_calls_sync_and_embed_exactly_once(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
        service = MarvinService(settings)
        try:
            self._seed(service)
            with (
                patch.object(service, "sync", wraps=service.sync) as sync_spy,
                patch.object(
                    service.embedder,
                    "embed_text",
                    wraps=service.embedder.embed_text,
                ) as embed_spy,
            ):
                service.prepare_session(task="anything", limit=6)
            assert sync_spy.call_count == 1
            assert embed_spy.call_count == 1
        finally:
            service.close()

    def test_reranker_invoked_once_when_enabled(self, tmp_path: Path) -> None:
        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
            rerank_enabled=True,
            rerank_depth=10,
        )
        service = MarvinService(settings)
        service.reranker = _FakeReranker(marker="LOG_MARKER")
        try:
            service.remember_semantic(
                concept="Logging",
                content="Use logging LOG_MARKER consistently.",
            )
            service.store_procedure(
                title="Setup logging",
                steps=["import logging", "set level"],
            )
            service.reflect(
                title="Reflection on logging",
                insight="Always log at module scope.",
            )

            with patch.object(service.reranker, "score", wraps=service.reranker.score) as score_spy:
                service.prepare_session(task="how to log in python", limit=6)
            # Single rerank pass over the merged pool, not three.
            assert score_spy.call_count == 1
        finally:
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
                "gpu_active",
                "gpu_lib_count",
                "kg_enabled",
                "kg_rrf_k",
                "vault_path",
                "index_path",
            ):
                assert key in payload, key
            assert isinstance(payload["embedding_dimensions"], int)
            assert isinstance(payload["rerank_enabled"], bool)
            assert isinstance(payload["rerank_depth"], int)
            assert isinstance(payload["gpu_active"], bool)
            assert isinstance(payload["gpu_lib_count"], int)
            assert isinstance(payload["kg_enabled"], bool)
            assert isinstance(payload["kg_rrf_k"], float)
        finally:
            service.close()

    def test_reports_gpu_inactive_without_bootstrap(self, tmp_path: Path) -> None:
        """Hash backend never preloads CUDA; ``gpu_active`` must be False.

        The bootstrap module caches state in process-wide globals so we
        defensively clear them. The True case (CUDA actually loaded) is
        covered by the manual smoke test in the worktree (eval CLI with
        ``--extra gpu``).
        """
        import marvin.gpu as gpu

        gpu._BOOTSTRAPPED = False
        gpu._LOADED_LIBS = []

        settings = MarvinSettings(
            vault_path=tmp_path / "vault",
            state_dir=tmp_path / ".state",
            embedding_provider="hash",
        )
        service = MarvinService(settings)
        try:
            payload = service.health()
            assert payload["gpu_active"] is False
            assert payload["gpu_lib_count"] == 0
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
    plain = service.index.hybrid_search(query="quokka", query_embedding=query_embedding, limit=3)
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
