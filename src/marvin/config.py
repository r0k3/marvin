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
    # chunk-tier note ranking with the graph note ranking.
    kg_enabled: bool = Field(default=True)
    kg_rrf_k: float = Field(default=60.0, gt=0.0)

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
