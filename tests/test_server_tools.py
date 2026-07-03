"""The MCP app exposes the full service surface (20 tools) and the new
wrappers delegate correctly."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from marvin.config import MarvinSettings
from marvin.server import create_app

EXPECTED_TOOLS = {
    "marvin_sync",
    "marvin_search",
    "marvin_recent_activity",
    "marvin_read_memory",
    "marvin_remember_semantic",
    "marvin_store_procedure",
    "marvin_register_template",
    "marvin_record_template_use",
    "marvin_match_template",
    "marvin_log_episode",
    "marvin_reflect",
    "marvin_prepare_session",
    "marvin_finalize_session",
    "marvin_start_worktree",
    "marvin_merge_worktree",
    "marvin_trigger_sleep",
    "marvin_consolidate",
    "marvin_rebuild",
    "marvin_consistency_check",
    "marvin_health",
}


@pytest.fixture()
def app(tmp_path: Path):
    settings = MarvinSettings(
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / ".state",
        embedding_provider="hash",
    )
    return create_app(settings)


def test_all_tools_registered(app):
    tools = asyncio.run(app.list_tools())
    names = {tool.name for tool in tools}
    assert names == EXPECTED_TOOLS


def test_health_tool_returns_snapshot(app):
    content = asyncio.run(app.call_tool("marvin_health", {}))
    text = str(content)
    assert "vault_path" in text


def test_consistency_check_tool_runs(app):
    content = asyncio.run(app.call_tool("marvin_consistency_check", {}))
    assert "consistent" in str(content)


def test_match_template_tool_empty_vault(app):
    content = asyncio.run(app.call_tool("marvin_match_template", {"context": "anything"}))
    # No templates registered: an empty (but well-formed) result, not an error.
    assert "error" not in str(content).lower()
