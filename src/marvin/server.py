from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .broker import MarvinBroker
from .config import MarvinSettings
from .git import GitManager
from .models import (
    MemoryKind,
    MemoryWriteResult,
    SearchHit,
    SessionClosureResult,
    SessionContext,
    SyncReport,
)
from .service import MarvinService


def parse_kind(value: str | None) -> MemoryKind | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return MemoryKind(normalized)


def create_app(settings: MarvinSettings) -> FastMCP:
    git_manager = GitManager(settings.resolved_vault_path)
    broker = MarvinBroker(os.environ.get("NATS_URL", "nats://127.0.0.1:4222"))
    service = MarvinService(settings, broker=broker, git_manager=git_manager)

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        service.sync()
        try:
            yield
        finally:
            service.close()

    app = FastMCP(
        name="marvin",
        instructions=(
            "Marvin is an Obsidian-native long-term memory system for coding and chat agents. "
            "Use search before guessing, store durable facts and procedures "
            "when the user reveals them, "
            "and log meaningful completed work as episodic memory."
        ),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        streamable_http_path="/mcp",
        sse_path="/sse",
        message_path="/messages/",
        lifespan=lifespan,
    )

    @app.tool(description="Sync the Obsidian vault into Marvin's local hybrid index.")
    def marvin_sync() -> SyncReport:
        return service.sync()

    @app.tool(description="Search Marvin memory using hybrid semantic and keyword retrieval.")
    def marvin_search(query: str, kind: str | None = None, limit: int = 6) -> list[SearchHit]:
        return service.search(query=query, kind=parse_kind(kind), limit=limit)

    @app.tool(description="Fetch the most recent memories, optionally filtered by memory kind.")
    def marvin_recent_activity(kind: str | None = None, limit: int = 8) -> list[SearchHit]:
        return service.recent(kind=parse_kind(kind), limit=limit)

    @app.tool(description="Read a single memory note by title, alias, or vault-relative path.")
    def marvin_read_memory(identifier: str) -> dict[str, Any]:
        note = service.get_note(identifier)
        if note is None:
            return {"found": False, "identifier": identifier}
        return {
            "found": True,
            "title": note.metadata.title,
            "kind": note.metadata.kind.value,
            "path": str(note.path.relative_to(settings.resolved_vault_path)),
            "tags": note.metadata.tags,
            "links": note.metadata.links,
            "body": note.body,
        }

    @app.tool(description="Store durable factual knowledge in semantic memory.")
    def marvin_remember_semantic(
        concept: str,
        content: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
    ) -> MemoryWriteResult:
        return service.remember_semantic(
            concept=concept,
            content=content,
            tags=tags,
            links=links,
            source={"tool": "marvin_remember_semantic"},
        )

    @app.tool(
        description="Store a reusable procedure, convention, or playbook for future sessions."
    )
    def marvin_store_procedure(
        title: str,
        steps: list[str],
        applicability: list[str] | None = None,
        anti_patterns: list[str] | None = None,
        tags: list[str] | None = None,
        links: list[str] | None = None,
    ) -> MemoryWriteResult:
        return service.store_procedure(
            title=title,
            steps=steps,
            applicability=applicability,
            anti_patterns=anti_patterns,
            tags=tags,
            links=links,
            source={"tool": "marvin_store_procedure"},
        )

    @app.tool(description="Log a completed task, event, or session into episodic memory.")
    def marvin_log_episode(
        title: str,
        summary: str,
        details: str = "",
        tags: list[str] | None = None,
        links: list[str] | None = None,
    ) -> MemoryWriteResult:
        return service.log_episode(
            title=title,
            summary=summary,
            details=details,
            tags=tags,
            links=links,
            source={"tool": "marvin_log_episode"},
        )

    @app.tool(description="Store a reflection, design principle, or lesson learned.")
    def marvin_reflect(
        title: str,
        insight: str,
        tags: list[str] | None = None,
        links: list[str] | None = None,
    ) -> MemoryWriteResult:
        return service.reflect(
            title=title,
            insight=insight,
            tags=tags,
            links=links,
            source={"tool": "marvin_reflect"},
        )

    @app.tool(
        description=(
            "Hook-friendly session bootstrap that pulls relevant"
            " procedures, facts, and recent episodes."
        )
    )
    def marvin_prepare_session(
        task: str,
        repo_name: str | None = None,
        technologies: list[str] | None = None,
        limit: int = 8,
    ) -> SessionContext:
        return service.prepare_session(
            task=task, repo_name=repo_name, technologies=technologies, limit=limit
        )

    @app.tool(
        description=(
            "Hook-friendly session finalizer that logs an episode and"
            " optionally extracts facts, procedures, and reflections."
        )
    )
    def marvin_finalize_session(
        title: str,
        summary: str,
        details: str = "",
        tags: list[str] | None = None,
        links: list[str] | None = None,
        semantic_facts: list[str] | None = None,
        procedures: list[dict[str, object]] | None = None,
        reflections: list[str] | None = None,
    ) -> SessionClosureResult:
        return service.hook_session_end(
            title=title,
            summary=summary,
            details=details,
            tags=tags,
            links=links,
            semantic_facts=semantic_facts,
            procedures=procedures,
            reflections=reflections,
            source={"tool": "marvin_finalize_session"},
        )

    @app.tool(description="Start a new agentic worktree (Git branch) for a specific feature.")
    def marvin_start_worktree(branch_name: str) -> str:
        return git_manager.create_worktree(branch_name)

    @app.tool(description="Merge a completed worktree back into main.")
    def marvin_merge_worktree(branch_name: str) -> dict[str, str]:
        return git_manager.merge_worktree(branch_name)

    @app.tool(description="Trigger background consolidation and graphing via the Brain Worker.")
    async def marvin_trigger_sleep() -> str:
        await broker.publish("memory.sleep", {"trigger": "agent"})
        return "Consolidation requested. The brain worker is now processing."

    return app


def build_settings_from_args(args: argparse.Namespace) -> MarvinSettings:
    settings = MarvinSettings()
    if args.vault_path:
        settings.vault_path = Path(args.vault_path)
    if args.state_dir:
        settings.state_dir = Path(args.state_dir)
    if args.transport:
        settings.transport = args.transport
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.log_level:
        settings.log_level = args.log_level
    if args.embedding_provider:
        settings.embedding_provider = args.embedding_provider
    if args.embedding_model:
        settings.embedding_model = args.embedding_model
    return settings


async def run_server(settings: MarvinSettings) -> None:
    app = create_app(settings)
    if settings.transport == "stdio":
        await app.run_stdio_async()
        return
    if settings.transport == "sse":
        await app.run_sse_async()
        return
    await app.run_streamable_http_async()


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Marvin MCP server.")
    parser.add_argument("--vault-path", help="Path to the Obsidian-compatible vault")
    parser.add_argument("--state-dir", help="Optional path for Marvin's SQLite index and state")
    parser.add_argument("--transport", choices=["http", "sse", "stdio"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument("--embedding-provider", choices=["auto", "fastembed", "hash"], default=None)
    parser.add_argument("--embedding-model", default=None)
    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()
    settings = build_settings_from_args(args)
    asyncio.run(run_server(settings))


if __name__ == "__main__":
    main()
