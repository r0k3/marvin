"""The Claude Code plugin (plugins/marvin-memory) stays valid and in sync."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / "plugins" / "marvin-memory"

KNOWN_HOOK_EVENTS = {
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "SessionStart",
    "SessionEnd",
    "PreCompact",
    "Notification",
}


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} missing frontmatter"
    _, fm, _ = text.split("---\n", 2)
    return yaml.safe_load(fm)


def test_manifest_matches_package_version():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "marvin-memory"
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert manifest["version"] == pyproject["project"]["version"]


def test_marketplace_points_at_existing_plugin():
    marketplace = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    entries = {p["name"]: p for p in marketplace["plugins"]}
    assert "marvin-memory" in entries
    source = entries["marvin-memory"]["source"]
    assert source.startswith("./")
    assert (REPO / source / ".claude-plugin" / "plugin.json").exists()


def test_hooks_config_is_valid():
    config = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())
    assert "hooks" in config, "plugin hooks.json requires the wrapper format"
    events = config["hooks"]
    assert set(events) <= KNOWN_HOOK_EVENTS
    assert {"SessionStart", "UserPromptSubmit"} <= set(events)
    for entries in events.values():
        for entry in entries:
            for hook in entry["hooks"]:
                assert hook["type"] == "command"
                assert "marvin hook" in hook["command"]
                # a missing marvin binary must no-op, never error
                assert "command -v marvin" in hook["command"]
                assert isinstance(hook["timeout"], int)


def test_mcp_config_serves_stdio():
    config = json.loads((PLUGIN / ".mcp.json").read_text())
    marvin = config["mcpServers"]["marvin"]
    assert marvin["command"] == "marvin"
    assert "stdio" in marvin["args"]


def test_every_skill_has_name_and_description():
    skill_files = sorted((PLUGIN / "skills").glob("*/SKILL.md"))
    assert len(skill_files) == 6
    for path in skill_files:
        fm = _frontmatter(path)
        assert fm.get("name"), path
        assert fm.get("description"), path


def test_knowledge_skill_matches_canonical():
    canonical = (REPO / "src" / "marvin" / "skill" / "SKILL.md").read_text()
    plugin_copy = (PLUGIN / "skills" / "marvin-memory" / "SKILL.md").read_text()
    assert plugin_copy == canonical, "run `just sync-plugin` after editing the canonical skill"


def test_curator_agent_frontmatter():
    fm = _frontmatter(PLUGIN / "agents" / "memory-curator.md")
    assert fm["name"] == "memory-curator"
    assert "curate" in fm["description"].lower()
    assert "Bash" in fm["tools"]
