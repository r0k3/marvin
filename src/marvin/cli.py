"""marvin — an AXI-style CLI over the Marvin memory system.

Follows the Agent eXperience Interface conventions (https://axi.md):

- **Content first**: bare ``marvin`` prints a live, directory-scoped
  dashboard of the vault, not help text.
- **TOON output**: tabular results declare count + schema once
  (``hits[3]{title,kind,path}:``) and emit compact rows — roughly 40%
  fewer tokens than JSON.
- **Minimal default schemas**: list commands return 3–4 fields;
  ``--fields`` requests more.
- **Truncation with size hints**: large bodies are cut with an explicit
  ``(truncated, N chars total — use --full ...)`` marker.
- **Definitive empty states**: ``hits[0]: (no matches for "x")`` — never
  silent emptiness.
- **Structured errors, honest exit codes**: runtime failures print an
  ``error[1]{code,message}`` block on stdout and exit 1; usage errors
  (including unknown flags) fail loud with exit 2; nothing ever prompts.
- **Contextual disclosure**: outputs end with a ``help[]`` block of
  concrete next-step command templates (placeholders, never guesses).

The MCP server remains available as ``marvin serve``; the pre-0.3
invocation (``marvin --transport stdio ...``) is detected and forwarded
with a deprecation note.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import server as _server
from .config import MarvinSettings
from .models import MemoryKind, NoteRecord, SearchHit
from .server import parse_kind
from .service import MarvinService
from .toon import encode_error, encode_help, encode_kv, encode_table

BODY_PREVIEW_CHARS = 600
_HIT_DEFAULT_FIELDS = ["title", "kind", "path"]
_HIT_ALL_FIELDS = ["title", "kind", "path", "score", "excerpt", "tags", "links"]


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _hit_row(hit: SearchHit) -> dict[str, object]:
    return {
        "title": hit.title,
        "kind": hit.kind.value,
        "path": hit.path,
        "score": hit.score,
        "excerpt": hit.excerpt,
        "tags": hit.tags,
        "links": hit.links,
    }


def _resolve_fields(requested: str | None) -> list[str]:
    if not requested:
        return list(_HIT_DEFAULT_FIELDS)
    fields = [f.strip() for f in requested.split(",") if f.strip()]
    unknown = [f for f in fields if f not in _HIT_ALL_FIELDS]
    if unknown:
        raise SystemExit(_usage_error(f"unknown field(s): {', '.join(unknown)}"))
    return fields


def _usage_error(message: str) -> int:
    print(encode_error("usage", message))
    return 2


def _print(*blocks: str) -> None:
    print("\n".join(block for block in blocks if block))


def _write_result_row(result) -> dict[str, object]:
    return {
        "title": result.title,
        "kind": result.kind.value,
        "path": result.path,
        "created": result.created,
    }


def _truncate_body(body: str, *, full: bool) -> str:
    if full or len(body) <= BODY_PREVIEW_CHARS:
        return body
    return (
        body[:BODY_PREVIEW_CHARS]
        + f"\n(truncated, {len(body)} chars total — use --full to see complete body)"
    )


# ---------------------------------------------------------------------------
# Commands. Each takes (service, args) and returns an exit code.
# ---------------------------------------------------------------------------


def cmd_dashboard(service: MarvinService, args: argparse.Namespace) -> int:
    """Live vault dashboard (AXI: content first, pre-computed aggregates).

    Reads only the vault and the existing index — never loads an
    embedding model and never calls an LLM, so it is safe as an ambient
    session hook.
    """
    settings = service.settings
    notes = service.vault.list_notes()
    by_kind = {kind.value: 0 for kind in MemoryKind}
    unconsolidated = 0
    for note in notes:
        by_kind[note.metadata.kind.value] += 1
        if note.metadata.kind is MemoryKind.EPISODIC and not note.metadata.consolidated:
            unconsolidated += 1

    _print(
        encode_kv(
            "vault",
            {
                "path": str(settings.resolved_vault_path),
                "notes": len(notes),
                "episodic": by_kind["episodic"],
                "semantic": by_kind["semantic"],
                "procedural": by_kind["procedural"],
                "reflective": by_kind["reflective"],
                "unconsolidated_episodes": unconsolidated,
                "indexed": service.index.note_count(),
            },
        )
    )
    recent = service.recent(limit=3)
    _print(
        encode_table(
            "recent",
            [_hit_row(h) for h in recent],
            _HIT_DEFAULT_FIELDS,
            empty="vault is empty — nothing remembered yet",
        )
    )
    suggestions = [
        ("marvin search <query>", "hybrid recall across all four memory types"),
        ("marvin remember <concept> --predicate <p> --value <v>", "store a semantic fact"),
        ("marvin session prepare <task>", "pull context for the work you are starting"),
    ]
    if unconsolidated >= 3:
        suggestions.insert(
            1, ("marvin consolidate", f"distill {unconsolidated} unconsolidated episodes")
        )
    _print(encode_help(suggestions))
    return 0


def cmd_search(service: MarvinService, args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    hits = service.search(query=query, kind=parse_kind(args.kind), limit=args.limit)
    fields = _resolve_fields(args.fields)
    _print(
        encode_table(
            "hits",
            [_hit_row(h) for h in hits],
            fields,
            empty=f'no matches for "{query}"',
        ),
        encode_help(
            [
                ("marvin read <path>", "open a result in full"),
                ("marvin search <query> --kind semantic", "restrict to one memory type"),
                ("marvin search <query> --fields title,kind,path,excerpt", "wider schema"),
            ]
        ),
    )
    return 0


def cmd_recent(service: MarvinService, args: argparse.Namespace) -> int:
    hits = service.recent(kind=parse_kind(args.kind), limit=args.limit)
    fields = _resolve_fields(args.fields)
    _print(
        encode_table(
            "recent",
            [_hit_row(h) for h in hits],
            fields,
            empty="no memories recorded yet",
        ),
        encode_help([("marvin read <path>", "open a memory in full")]),
    )
    return 0


def cmd_read(service: MarvinService, args: argparse.Namespace) -> int:
    note: NoteRecord | None = service.get_note(args.identifier)
    if note is None:
        _print(
            encode_table("note", [], ["title"], empty=f'no note matches "{args.identifier}"'),
            encode_help([("marvin search <query>", "find the right identifier")]),
        )
        return 1
    relative = str(note.path.relative_to(service.settings.resolved_vault_path))
    _print(
        encode_kv(
            "note",
            {
                "title": note.metadata.title,
                "kind": note.metadata.kind.value,
                "path": relative,
                "tags": note.metadata.tags,
                "links": note.metadata.links,
                "facts": len(note.metadata.facts),
                "body_chars": len(note.body),
            },
        )
    )
    if note.metadata.facts:
        _print(
            encode_table(
                "facts",
                [
                    {
                        "predicate": f.predicate,
                        "value": f.value,
                        "aspect": f.aspect.value,
                        "deprecated": f.deprecated,
                    }
                    for f in note.metadata.facts
                ],
                ["predicate", "value", "aspect", "deprecated"],
            )
        )
    print("body:")
    print(_truncate_body(note.body, full=args.full))
    return 0


def cmd_remember(service: MarvinService, args: argparse.Namespace) -> int:
    result = service.remember_semantic(
        concept=args.concept,
        content=args.content,
        predicate=args.predicate,
        value=args.value,
        aspect=args.aspect,
        confidence=args.confidence,
        tags=_split_csv(args.tags),
        links=_split_csv(args.links),
        source={"tool": "cli"},
    )
    _print(
        encode_table("stored", [_write_result_row(result)], ["title", "kind", "path", "created"]),
        encode_help(
            [
                ("marvin read <path>", "inspect the stored fact"),
                (
                    "marvin remember <concept> --predicate <p> --value <v2>",
                    "update (soft-deprecates the old value)",
                ),
            ]
        ),
    )
    return 0


def cmd_procedure(service: MarvinService, args: argparse.Namespace) -> int:
    result = service.store_procedure(
        title=args.title,
        steps=args.step,
        applicability=args.applies or None,
        anti_patterns=args.avoid or None,
        tags=_split_csv(args.tags),
        links=_split_csv(args.links),
        source={"tool": "cli"},
    )
    _print(
        encode_table("stored", [_write_result_row(result)], ["title", "kind", "path", "created"])
    )
    return 0


def cmd_template_register(service: MarvinService, args: argparse.Namespace) -> int:
    result = service.register_template(
        title=args.title,
        plan=args.plan,
        intents=args.intent or None,
        styles=args.style or None,
        entity_types=args.entity_type or None,
        trigger_phrases=args.trigger or None,
        slots=args.slot or None,
        failure_modes=args.failure or None,
        tags=_split_csv(args.tags),
        source={"tool": "cli"},
    )
    _print(
        encode_table("stored", [_write_result_row(result)], ["title", "kind", "path", "created"]),
        encode_help([("marvin template match <context> --intent <intent>", "try selecting it")]),
    )
    return 0


def cmd_template_match(service: MarvinService, args: argparse.Namespace) -> int:
    matches = service.match_template(
        " ".join(args.context),
        intent=args.intent or "",
        styles=args.style or (),
        entity_types=args.entity_type or (),
        top_k=args.top_k,
    )
    rows = [
        {
            "title": m.template.title,
            "score": m.score,
            "effectiveness": m.effectiveness,
            "plan": list(m.template.plan),
        }
        for m in matches
    ]
    _print(
        encode_table(
            "templates",
            rows,
            ["title", "score", "effectiveness", "plan"],
            empty="no template's triggers match this context",
        ),
        encode_help(
            [
                (
                    "marvin template used <title> [--failure]",
                    "record the outcome after applying one",
                ),
                (
                    "marvin template register <title> --plan <step> --intent <intent>",
                    "add a strategy",
                ),
            ]
        ),
    )
    return 0


def cmd_template_used(service: MarvinService, args: argparse.Namespace) -> int:
    service.record_template_use(args.title, success=not args.failure)
    outcome = "failure" if args.failure else "success"
    _print(encode_kv("recorded", {"title": args.title, "outcome": outcome}))
    return 0


def cmd_episode(service: MarvinService, args: argparse.Namespace) -> int:
    result = service.log_episode(
        title=args.title,
        summary=args.summary,
        details=args.details,
        tags=_split_csv(args.tags),
        links=_split_csv(args.links),
        source={"tool": "cli"},
    )
    _print(
        encode_table("stored", [_write_result_row(result)], ["title", "kind", "path", "created"])
    )
    return 0


def cmd_reflect(service: MarvinService, args: argparse.Namespace) -> int:
    result = service.reflect(
        title=args.title,
        insight=args.insight,
        tags=_split_csv(args.tags),
        links=_split_csv(args.links),
        source={"tool": "cli"},
    )
    _print(
        encode_table("stored", [_write_result_row(result)], ["title", "kind", "path", "created"])
    )
    return 0


def cmd_session_prepare(service: MarvinService, args: argparse.Namespace) -> int:
    context = service.prepare_session(
        task=" ".join(args.task),
        repo_name=args.repo,
        technologies=args.tech or None,
        limit=args.limit,
    )
    for name, hits in (
        ("procedural", context.procedural),
        ("semantic", context.semantic),
        ("reflective", context.reflective),
        ("recent_episodes", context.recent_episodes),
    ):
        _print(
            encode_table(
                name, [_hit_row(h) for h in hits], _HIT_DEFAULT_FIELDS, empty="none relevant"
            )
        )
    if context.guidance:
        _print(encode_table("guidance", [{"line": g} for g in context.guidance], ["line"]))
    _print(encode_help([("marvin read <path>", "open any of the surfaced notes")]))
    return 0


def cmd_session_finalize(service: MarvinService, args: argparse.Namespace) -> int:
    result = service.hook_session_end(
        title=args.title,
        summary=args.summary,
        details=args.details,
        semantic_facts=args.fact or None,
        reflections=args.reflection or None,
        source={"tool": "cli"},
    )
    rows = [_write_result_row(result.episode)]
    rows += [_write_result_row(r) for r in result.stored_semantic]
    rows += [_write_result_row(r) for r in result.stored_reflections]
    _print(encode_table("stored", rows, ["title", "kind", "path", "created"]))
    return 0


def cmd_sync(service: MarvinService, args: argparse.Namespace) -> int:
    report = service.sync()
    _print(
        encode_kv(
            "sync",
            {"scanned": report.scanned, "indexed": report.indexed, "removed": report.removed},
        )
    )
    return 0


def cmd_rebuild(service: MarvinService, args: argparse.Namespace) -> int:
    report = service.rebuild()
    _print(
        encode_kv(
            "rebuild",
            {"scanned": report.scanned, "indexed": report.indexed, "removed": report.removed},
        )
    )
    return 0


def cmd_check(service: MarvinService, args: argparse.Namespace) -> int:
    report = service.consistency_check()
    _print(
        encode_kv(
            "consistency",
            {
                "consistent": report.consistent,
                "vault_notes": report.vault_notes,
                "indexed_notes": report.indexed_notes,
            },
        )
    )
    if report.missing_from_index:
        _print(
            encode_table(
                "missing_from_index",
                [{"path": p} for p in report.missing_from_index],
                ["path"],
            )
        )
    if report.orphaned_in_index:
        _print(
            encode_table(
                "orphaned_in_index",
                [{"path": p} for p in report.orphaned_in_index],
                ["path"],
            )
        )
    if not report.consistent:
        _print(encode_help([("marvin rebuild", "regenerate the index from the vault")]))
        return 1
    return 0


def cmd_health(service: MarvinService, args: argparse.Namespace) -> int:
    _print(encode_kv("health", service.health()))
    return 0


def cmd_consolidate(service: MarvinService, args: argparse.Namespace) -> int:
    from .consolidation import ConsolidationEngine

    engine_kwargs: dict[str, str] = {}
    if args.model:
        engine_kwargs["model"] = args.model
    if args.api_base:
        engine_kwargs["api_base"] = args.api_base
    engine = ConsolidationEngine(**engine_kwargs)
    facts = service.consolidate_semantic(engine=engine, min_episodes=args.min_episodes)
    insights = service.consolidate_reflective(engine=engine, min_facts=args.min_facts)
    _print(
        encode_kv(
            "consolidation",
            {"facts_extracted": len(facts), "insights_created": len(insights)},
        ),
        encode_table(
            "facts",
            [_write_result_row(r) for r in facts],
            ["title", "kind", "path", "created"],
            empty="no entity crossed the episode threshold",
        ),
        encode_table(
            "insights",
            [_write_result_row(r) for r in insights],
            ["title", "kind", "path", "created"],
            empty="no aspect group had enough facts",
        ),
    )
    return 0


def cmd_worktree_start(service: MarvinService, args: argparse.Namespace) -> int:
    from .git import GitManager

    manager = GitManager(service.settings.resolved_vault_path)
    message = manager.create_worktree(args.branch)
    _print(encode_kv("worktree", {"branch": args.branch, "status": message}))
    return 0


def cmd_worktree_merge(service: MarvinService, args: argparse.Namespace) -> int:
    from .git import GitManager

    manager = GitManager(service.settings.resolved_vault_path)
    result = manager.merge_worktree(args.branch)
    _print(encode_kv("worktree", dict(result)))
    return 0


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------


def _add_hit_flags(parser: argparse.ArgumentParser, *, default_limit: int) -> None:
    parser.add_argument("--kind", choices=["episodic", "semantic", "procedural", "reflective"])
    parser.add_argument("--limit", type=int, default=default_limit)
    parser.add_argument(
        "--fields",
        help=f"comma-separated columns (default: {','.join(_HIT_DEFAULT_FIELDS)}; "
        f"available: {','.join(_HIT_ALL_FIELDS)})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marvin",
        description="Obsidian-native long-term memory for agents — AXI-style CLI. "
        "Run with no arguments for a live vault dashboard.",
    )
    parser.add_argument("--vault-path", help="Path to the vault (default: $MARVIN_VAULT_PATH)")
    parser.add_argument("--state-dir", help="Path for the derived index/state")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("search", help="Hybrid recall across all memory types")
    p.add_argument("query", nargs="+")
    _add_hit_flags(p, default_limit=6)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("recent", help="Most recent memories")
    _add_hit_flags(p, default_limit=8)
    p.set_defaults(func=cmd_recent)

    p = sub.add_parser("read", help="Read one note by title, alias, or path")
    p.add_argument("identifier")
    p.add_argument("--full", action="store_true", help="do not truncate the body")
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("remember", help="Store a semantic fact (soft-deprecates on update)")
    p.add_argument("concept")
    p.add_argument("content", nargs="?", help="unstructured fallback when no --predicate/--value")
    p.add_argument("--predicate")
    p.add_argument("--value")
    p.add_argument(
        "--aspect",
        default="knowledge",
        choices=["knowledge", "preference", "decision", "goal", "problem", "belief", "directive"],
    )
    p.add_argument("--confidence", type=float, default=0.6)
    p.add_argument("--tags", help="comma-separated")
    p.add_argument("--links", help="comma-separated")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("procedure", help="Store a reusable procedure or rule")
    p.add_argument("title")
    p.add_argument("--step", action="append", required=True, help="repeatable, ordered")
    p.add_argument("--applies", action="append", help="when this applies (repeatable)")
    p.add_argument("--avoid", action="append", help="anti-pattern (repeatable)")
    p.add_argument("--tags")
    p.add_argument("--links")
    p.set_defaults(func=cmd_procedure)

    tpl = sub.add_parser("template", help="K-line templates: register / match / used")
    tpl_sub = tpl.add_subparsers(dest="template_command", required=True)

    p = tpl_sub.add_parser("register", help="Register a K-line response strategy")
    p.add_argument("title")
    p.add_argument("--plan", action="append", required=True, help="ordered plan step (repeatable)")
    p.add_argument("--intent", action="append")
    p.add_argument("--style", action="append")
    p.add_argument("--entity-type", action="append", dest="entity_type")
    p.add_argument("--trigger", action="append", help="keyword trigger phrase")
    p.add_argument("--slot", action="append")
    p.add_argument("--failure", action="append", help="failure mode to avoid")
    p.add_argument("--tags")
    p.set_defaults(func=cmd_template_register)

    p = tpl_sub.add_parser("match", help="Select templates for a context")
    p.add_argument("context", nargs="*", default=[])
    p.add_argument("--intent")
    p.add_argument("--style", action="append")
    p.add_argument("--entity-type", action="append", dest="entity_type")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")
    p.set_defaults(func=cmd_template_match)

    p = tpl_sub.add_parser("used", help="Record a template outcome (default: success)")
    p.add_argument("title")
    p.add_argument("--failure", action="store_true")
    p.set_defaults(func=cmd_template_used)

    p = sub.add_parser("episode", help="Log a completed task or event")
    p.add_argument("title")
    p.add_argument("--summary", required=True)
    p.add_argument("--details", default="")
    p.add_argument("--tags")
    p.add_argument("--links")
    p.set_defaults(func=cmd_episode)

    p = sub.add_parser("reflect", help="Store a lesson or insight")
    p.add_argument("title")
    p.add_argument("--insight", required=True)
    p.add_argument("--tags")
    p.add_argument("--links")
    p.set_defaults(func=cmd_reflect)

    ses = sub.add_parser("session", help="Session lifecycle: prepare / finalize")
    ses_sub = ses.add_subparsers(dest="session_command", required=True)

    p = ses_sub.add_parser("prepare", help="Pull relevant context for a task")
    p.add_argument("task", nargs="+")
    p.add_argument("--repo")
    p.add_argument("--tech", action="append")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_session_prepare)

    p = ses_sub.add_parser("finalize", help="Log a closing episode (+ optional extractions)")
    p.add_argument("title")
    p.add_argument("--summary", required=True)
    p.add_argument("--details", default="")
    p.add_argument("--fact", action="append", help="semantic fact to extract (repeatable)")
    p.add_argument("--reflection", action="append", help="reflection to store (repeatable)")
    p.set_defaults(func=cmd_session_finalize)

    p = sub.add_parser("sync", help="Index vault changes")
    p.set_defaults(func=cmd_sync)
    p = sub.add_parser("rebuild", help="Regenerate all derived indexes from the vault")
    p.set_defaults(func=cmd_rebuild)
    p = sub.add_parser("check", help="Vault/index consistency check")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("health", help="Runtime health snapshot")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("consolidate", help="Run two-phase consolidation now (uses the local LLM)")
    p.add_argument("--model", help="LiteLLM model id override")
    p.add_argument("--api-base", dest="api_base")
    p.add_argument("--min-episodes", type=int, default=3, dest="min_episodes")
    p.add_argument("--min-facts", type=int, default=3, dest="min_facts")
    p.set_defaults(func=cmd_consolidate)

    wt = sub.add_parser("worktree", help="Branch memory for risky work: start / merge")
    wt_sub = wt.add_subparsers(dest="worktree_command", required=True)
    p = wt_sub.add_parser("start", help="Create an isolated memory branch")
    p.add_argument("branch")
    p.set_defaults(func=cmd_worktree_start)
    p = wt_sub.add_parser("merge", help="Merge a memory branch back to main")
    p.add_argument("branch")
    p.set_defaults(func=cmd_worktree_merge)

    # `serve` is dispatched before argument parsing (see main()) so every
    # remaining token reaches the server parser untouched; registered here
    # only so it appears in `marvin --help`.
    sub.add_parser("serve", help="Run the Marvin MCP server (all remaining args forwarded)")

    return parser


def _build_settings(args: argparse.Namespace) -> MarvinSettings:
    settings = MarvinSettings()
    if args.vault_path:
        settings.vault_path = Path(args.vault_path)
    if args.state_dir:
        settings.state_dir = Path(args.state_dir)
    return settings


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Pre-0.3 compatibility: `marvin --transport stdio ...` started the MCP
    # server. Detect the legacy flags-only invocation (no subcommand) and
    # forward it so existing agent configurations keep working.
    if argv and argv[0].startswith("--") and "--transport" in argv:
        print(
            "note: `marvin --transport ...` is deprecated; use `marvin serve --transport ...`",
            file=sys.stderr,
        )
        _server.main(argv)
        return 0

    # Dispatch `serve` before parsing so its flags reach the server parser
    # verbatim (argparse REMAINDER mishandles leading option tokens on 3.12).
    if argv and argv[0] == "serve":
        _server.main(argv[1:])
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    settings = _build_settings(args)
    service = MarvinService(settings)
    try:
        if args.command is None:
            return cmd_dashboard(service, args)
        return args.func(service, args)
    except Exception as exc:  # structured error on stdout, exit 1 (AXI)
        print(encode_error("runtime", f"{type(exc).__name__}: {exc}"))
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
