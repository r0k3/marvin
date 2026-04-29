from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MarvinSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MARVIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vault_path: Path = Field(default=Path("marvin_vault"))
    state_dir: Path | None = Field(default=None)

    transport: Literal["http", "sse", "stdio"] = Field(default="http")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8421)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    embedding_provider: Literal["auto", "fastembed", "hash"] = Field(default="auto")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dimensions: int = Field(default=384)

    rerank_enabled: bool = Field(default=False)
    rerank_provider: Literal["auto", "fastembed", "none"] = Field(default="auto")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_depth: int = Field(default=50)
    rerank_max_chars: int = Field(default=1024)

    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    search_limit: int = Field(default=6)
    recency_limit: int = Field(default=6)

    # Each first-stage stream (FTS5 and sqlite-vec) pulls
    # ``max(limit * first_stage_overfetch, first_stage_overfetch_min)`` chunks
    # before they are fused with RRF. Higher values trade latency for recall;
    # the defaults match the legacy hardcoded ``max(limit * 5, 20)``.
    first_stage_overfetch: int = Field(default=5, ge=1)
    first_stage_overfetch_min: int = Field(default=20, ge=1)

    # K-Lines graph retrieval as a third RRF stream. Enabled by default
    # because hydration is cheap and the stream silently contributes
    # nothing on vaults that have not yet been wikilink-consolidated.
    # ``kg_rrf_k`` is the RRF damping constant used when fusing the
    # chunk-tier note ranking with the graph note ranking;
    # ``kg_fusion_weight`` scales the graph stream's RRF score (< 1
    # keeps strong chunk matches from being displaced by noisy entity
    # signal; 1.0 = symmetric fusion).
    kg_enabled: bool = Field(default=True)
    kg_rrf_k: float = Field(default=60.0, gt=0.0)
    kg_fusion_weight: float = Field(default=0.5, ge=0.0)

    # Time-aware re-ranking ("memory decay"). Off by default so any
    # behaviour change is opt-in. When ``decay_enabled`` is set, the
    # final note-level RRF score is multiplied by
    # ``1 + decay_weight * exp(-age_days / decay_half_life_days)`` where
    # ``age`` is computed from ``note.updated_at`` and a query-time
    # reference. This is a *freshness boost*, not a staleness penalty:
    # an old-but-relevant note keeps its full RRF score, but a
    # comparably-relevant fresh note is preferred. ``decay_weight`` is
    # the maximum boost (so 1.0 is at most a 2x score for an instant-old
    # note); ``decay_half_life_days`` controls how quickly that boost
    # decays. Sensible defaults assume episodic chat notes, where
    # roughly the last month dominates relevance for "what did I just
    # say" style queries.
    decay_enabled: bool = Field(default=False)
    decay_half_life_days: float = Field(default=30.0, gt=0.0)
    decay_weight: float = Field(default=0.5, ge=0.0)
    # Comma-separated list of note kinds the freshness boost applies
    # to. Default is ``"episodic"`` because semantic facts and
    # procedural how-tos are (mostly) timeless and shouldn't be
    # demoted purely for being old. Set to ``"episodic,procedural"``
    # for vaults that treat procedures as time-sensitive (deployment
    # runbooks, dependency notes), or ``""`` to apply decay to no
    # kind (effectively disabling it without flipping the toggle).
    decay_kinds_csv: str = Field(default="episodic")

    # At-ingest entity extraction. Augments ``metadata.links`` (which
    # only carries explicit ``[[wikilinks]]`` produced by the
    # consolidator) with capitalised noun phrases pulled from the body
    # via a regex extractor.
    #
    # Default: ``False``. The regex extractor is empirically a wash to
    # mildly harmful on chat-style benchmarks (LongMemEval-S 100q:
    # multi-session R@5 -7pp; single-session unchanged) because most
    # capitalised tokens in chat data are sentence-starter
    # imperatives, not entities, and the few real entities a query
    # references rarely line up with what the regex finds in the
    # haystack. Wikilinks-only Phase 1A behaviour is silent on such
    # data and matches the chunk-only baseline exactly.
    #
    # Enable when:
    #   * the vault has been wikilink-consolidated (the LLM extractor
    #     has run) and you want freshly-ingested notes to also
    #     contribute graph signal before the next consolidation pass,
    #     OR
    #   * the source text is curated (docs, research notes) where
    #     capitalisation reliably marks proper nouns.
    kg_extract_at_ingest: bool = Field(default=False)
    # Drop entities shorter than this many characters; raise to suppress
    # sentence-starter noise like ``The``, ``His`` (default keeps the
    # ``len > 2`` policy used by the existing LLM extraction path).
    kg_ingest_min_length: int = Field(default=3, ge=1)
    # Drop single-word at-ingest extractions (``True`` by default).
    # Capitalised single words on chat-style data are dominated by
    # sentence-starter verbs and modal auxiliaries (``Remember``,
    # ``However``, ``Can``, ``Did``); a multi-word policy keeps phrases
    # like ``Apple Card`` or ``Western Australia`` while throwing the
    # noise out wholesale. Set to ``false`` if you trust the source
    # text (curated docs, code) and want single-word proper nouns too.
    kg_ingest_multiword_only: bool = Field(default=True)

    @property
    def resolved_vault_path(self) -> Path:
        return self.vault_path.expanduser().resolve()

    @property
    def resolved_state_dir(self) -> Path:
        if self.state_dir is not None:
            return self.state_dir.expanduser().resolve()
        return self.resolved_vault_path / ".marvin"

    @property
    def index_path(self) -> Path:
        return self.resolved_state_dir / "marvin.db"

    def ensure_directories(self) -> None:
        self.resolved_vault_path.mkdir(parents=True, exist_ok=True)
        self.resolved_state_dir.mkdir(parents=True, exist_ok=True)
