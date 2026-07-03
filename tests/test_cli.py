"""Tests for the AXI-style CLI (marvin.cli).

Each test drives ``main()`` exactly as the console script would, on a
temporary vault with the deterministic hash embedder, and asserts on the
TOON output contract the AXI spec cares about: declared counts, definitive
empty states, truncation hints, help[] suggestions, and exit codes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marvin import cli


@pytest.fixture()
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MARVIN_EMBEDDING_PROVIDER", "hash")
    monkeypatch.delenv("MARVIN_VAULT_PATH", raising=False)
    return tmp_path / "vault"


def run(vault: Path, *argv: str) -> int:
    return cli.main(["--vault-path", str(vault), "--state-dir", str(vault / ".state"), *argv])


class TestDashboard:
    def test_empty_vault_is_definitive(self, vault, capsys):
        assert run(vault) == 0
        out = capsys.readouterr().out
        assert "vault:" in out
        assert "notes: 0" in out
        assert "recent[0]: (vault is empty" in out
        assert "help[" in out

    def test_counts_and_consolidate_hint(self, vault, capsys):
        for i in range(3):
            run(vault, "episode", f"E{i}", "--summary", f"did thing {i}")
        capsys.readouterr()
        assert run(vault) == 0
        out = capsys.readouterr().out
        assert "episodic: 3" in out
        assert "unconsolidated_episodes: 3" in out
        assert "marvin consolidate" in out


class TestSearchAndRead:
    def test_remember_search_read_roundtrip(self, vault, capsys):
        assert (
            run(
                vault,
                "remember",
                "Database Configuration",
                "--predicate",
                "storage",
                "--value",
                "PostgreSQL with asyncpg",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert out.startswith("stored[1]{title,kind,path,created}:")
        assert "Database Configuration,semantic" in out

        assert run(vault, "search", "postgresql") == 0
        out = capsys.readouterr().out
        assert out.startswith("hits[1]{title,kind,path}:")
        assert "help[" in out

        path = "Semantic/Database Configuration.md"
        assert run(vault, "read", path) == 0
        out = capsys.readouterr().out
        assert "note:" in out
        assert "facts[1]{predicate,value,aspect,deprecated}:" in out
        assert "storage,PostgreSQL with asyncpg,knowledge,false" in out

    def test_empty_search_is_definitive(self, vault, capsys):
        # Empty vault: the empty state must be explicit, never silent.
        # (On a non-empty vault dense retrieval always returns the nearest
        # neighbours, so "no matches" is the empty-vault contract.)
        assert run(vault, "search", "zzz-not-there") == 0
        out = capsys.readouterr().out
        assert 'hits[0]: (no matches for "zzz-not-there")' in out

    def test_read_truncates_with_size_hint(self, vault, capsys):
        long_summary = "w " * 800
        run(vault, "episode", "Long One", "--summary", long_summary)
        capsys.readouterr()
        assert run(vault, "read", "Episodic/Long One.md") == 0
        out = capsys.readouterr().out
        assert "(truncated," in out and "use --full" in out

        assert run(vault, "read", "Episodic/Long One.md", "--full") == 0
        out = capsys.readouterr().out
        assert "(truncated," not in out

    def test_fields_flag_widens_schema(self, vault, capsys):
        run(vault, "remember", "Y", "--predicate", "p", "--value", "unique-marker-value")
        capsys.readouterr()
        assert run(vault, "search", "unique-marker-value", "--fields", "title,kind,excerpt") == 0
        out = capsys.readouterr().out
        assert out.startswith("hits[1]{title,kind,excerpt}:")

    def test_unknown_field_exits_2(self, vault, capsys):
        run(vault, "remember", "Z", "--predicate", "p", "--value", "v")
        capsys.readouterr()
        with pytest.raises(SystemExit) as excinfo:
            run(vault, "search", "v", "--fields", "title,nope")
        assert excinfo.value.code == 2
        assert "unknown field(s): nope" in capsys.readouterr().out

    def test_unknown_flag_fails_loud(self, vault):
        with pytest.raises(SystemExit) as excinfo:
            run(vault, "search", "x", "--definitely-not-a-flag")
        assert excinfo.value.code == 2


class TestTemplates:
    def test_register_match_used_cycle(self, vault, capsys):
        assert (
            run(
                vault,
                "template",
                "register",
                "Bug Triage",
                "--plan",
                "reproduce first",
                "--plan",
                "bisect the boundary",
                "--intent",
                "debug",
                "--trigger",
                "flaky",
            )
            == 0
        )
        capsys.readouterr()

        assert run(vault, "template", "match", "flaky test", "--intent", "debug") == 0
        out = capsys.readouterr().out
        assert out.startswith("templates[1]{title,score,effectiveness,plan}:")
        assert "Bug Triage" in out

        assert run(vault, "template", "used", "Bug Triage") == 0
        out = capsys.readouterr().out
        assert "outcome: success" in out

    def test_match_miss_is_definitive(self, vault, capsys):
        assert run(vault, "template", "match", "anything", "--intent", "unknown") == 0
        out = capsys.readouterr().out
        assert "templates[0]: (no template's triggers match this context)" in out


class TestMaintenance:
    def test_sync_check_rebuild_health(self, vault, capsys):
        run(vault, "remember", "C", "--predicate", "p", "--value", "v")
        capsys.readouterr()
        assert run(vault, "sync") == 0
        assert "sync:" in capsys.readouterr().out
        assert run(vault, "check") == 0
        assert "consistent: true" in capsys.readouterr().out
        assert run(vault, "rebuild") == 0
        assert "rebuild:" in capsys.readouterr().out
        assert run(vault, "health") == 0
        assert "health:" in capsys.readouterr().out

    def test_session_prepare_sections(self, vault, capsys):
        run(vault, "remember", "AuthService", "--predicate", "owner", "--value", "team-x")
        capsys.readouterr()
        assert run(vault, "session", "prepare", "work on AuthService") == 0
        out = capsys.readouterr().out
        for section in ("procedural", "semantic", "reflective", "recent_episodes"):
            assert section in out


class TestErrorsAndServe:
    def test_runtime_error_is_structured_exit_1(self, vault, capsys):
        # remember with neither content nor --value raises in the service.
        assert run(vault, "remember", "OnlyConcept") == 1
        out = capsys.readouterr().out
        assert out.startswith("error[1]{code,message}:")
        assert "runtime" in out

    def test_legacy_transport_invocation_forwards_to_server(self, vault, capsys, monkeypatch):
        called: dict[str, list[str]] = {}
        monkeypatch.setattr(cli._server, "main", lambda argv: called.setdefault("argv", argv))
        assert cli.main(["--vault-path", str(vault), "--transport", "stdio"]) == 0
        assert "--transport" in called["argv"]
        assert "deprecated" in capsys.readouterr().err

    def test_serve_subcommand_forwards_args(self, vault, monkeypatch):
        called: dict[str, list[str]] = {}
        monkeypatch.setattr(cli._server, "main", lambda argv: called.setdefault("argv", argv))
        assert cli.main(["serve", "--transport", "sse", "--port", "9000"]) == 0
        assert called["argv"] == ["--transport", "sse", "--port", "9000"]
