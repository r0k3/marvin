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

from .config import MarvinSettings
from .models import MemoryKind, NoteRecord, SearchHit, parse_kind
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
        reason=args.reason,
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


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    from urllib.request import urlopen

    try:
        with urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _tcp_ok(host: str, port: int, timeout: float = 1.5) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def cmd_doctor(service: MarvinService | None, args: argparse.Namespace) -> int:
    """Install-state checkup: every component, its status, the exact fix.

    Read-only by design: creates no directories, loads no models, calls
    no LLM. Missing optional components are reported with install
    commands, not treated as failures.
    """
    from importlib import metadata
    from importlib.util import find_spec
    from urllib.parse import urlparse

    settings = _build_settings(args)
    fixes: list[tuple[str, str]] = []

    try:
        version = metadata.version("marvin-memory")
    except metadata.PackageNotFoundError:
        version = "dev"

    # Core: vault + index. A missing vault is "not initialized", not broken.
    vault_path = settings.resolved_vault_path
    if vault_path.is_dir():
        note_count = sum(
            1 for kind in MemoryKind for _ in (vault_path / kind.folder_name).glob("*.md")
        )
        vault_status = f"{vault_path} ({note_count} notes)"
        git_status = "yes" if (vault_path / ".git").is_dir() else "no"
        if git_status == "no":
            fixes.append(("git -C <vault> init", "version the vault (memory writes as commits)"))
    else:
        vault_status = f"{vault_path} (missing — created on first write)"
        git_status = "n/a"
    index_status = "present" if settings.index_path.exists() else "absent (built on first sync)"

    _print(
        encode_kv(
            "doctor",
            {
                "version": version,
                "vault": vault_status,
                "vault_git": git_status,
                "index": index_status,
            },
        )
    )

    # Retrieval: embedder + reranker + GPU extra.
    fastembed_ok = find_spec("fastembed") is not None
    if settings.embedding_provider == "hash":
        embedding = "hash (configured)"
    elif fastembed_ok:
        embedding = f"fastembed {settings.embedding_model}"
    else:
        embedding = "hash fallback (fastembed unavailable)"
        fixes.append(("uv pip install fastembed", "restore dense-vector retrieval"))
    try:
        gpu_extra = f"onnxruntime-gpu {metadata.version('onnxruntime-gpu')}"
    except metadata.PackageNotFoundError:
        gpu_extra = "not installed (CPU inference)"
    _print(
        encode_kv(
            "retrieval",
            {
                "embedding": embedding,
                "rerank": "on" if settings.rerank_enabled else "off (MARVIN_RERANK_ENABLED=1)",
                "gpu_extra": gpu_extra,
            },
        )
    )

    # Sleep pass: [consolidate] extra + model endpoint.
    consolidate_ok = find_spec("litellm") is not None and find_spec("langextract") is not None
    sleep: dict[str, object] = {
        "extra_installed": "yes" if consolidate_ok else "no",
        "model": settings.sleep_model,
    }
    if not consolidate_ok:
        fixes.append(
            (
                "uv tool install 'marvin-memory[consolidate]'",
                "enable the sleep pass (entity extraction + consolidation)",
            )
        )
    elif settings.sleep_model.startswith("ollama/"):
        base = settings.sleep_api_base or "http://127.0.0.1:11434"
        reachable = _http_ok(f"{base.rstrip('/')}/api/version")
        sleep["endpoint"] = f"{base} ({'reachable' if reachable else 'unreachable'})"
        if not reachable:
            fixes.append(
                ("ollama serve", f"the sleep model {settings.sleep_model} needs this endpoint")
            )
    _print(encode_kv("sleep", sleep))

    # Bus: single-process default vs the [cluster] profile.
    cluster: dict[str, object] = {
        "bus": "memory (single-process)" if settings.bus == "memory" else "nats",
        "extra_installed": "yes" if find_spec("nats") is not None else "no",
    }
    if settings.bus == "nats":
        if find_spec("nats") is None:
            fixes.append(
                (
                    "uv tool install 'marvin-memory[cluster]'",
                    "MARVIN_BUS=nats requires the NATS client",
                )
            )
        parsed = urlparse(settings.nats_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 4222
        reachable = _tcp_ok(host, port)
        cluster["nats"] = f"{host}:{port} ({'reachable' if reachable else 'unreachable'})"
        if not reachable:
            fixes.append(("docker compose up -d nats", "start the broker MARVIN_BUS=nats expects"))
    _print(encode_kv("cluster", cluster))

    # The bundled agent skill.
    project_skill = Path.cwd() / ".claude" / "skills" / "marvin-memory"
    user_skill = Path.home() / ".claude" / "skills" / "marvin-memory"
    if not project_skill.is_dir() and not user_skill.is_dir():
        fixes.append(("marvin skill install", "teach your agent when to use memory"))
    _print(
        encode_kv(
            "skill",
            {
                "project": "installed" if project_skill.is_dir() else "absent",
                "user": "installed" if user_skill.is_dir() else "absent",
            },
        )
    )

    if fixes:
        _print(encode_help(fixes))
    return 0


# ---------------------------------------------------------------------------
# Auto-recall hooks (`marvin hook ...`): fast, budgeted, never blocking.
# Stdout is injected into the host agent's context (exit 0); errors go to
# stderr and the hook still exits 0 so a broken vault never breaks a session.
# ---------------------------------------------------------------------------


def _read_hook_stdin() -> dict[str, object]:
    """Parse the host's hook payload from stdin, tolerating anything.

    Returns ``{}`` on a TTY or empty input. Non-JSON input is preserved as
    ``{"text": ...}`` so hosts that pipe the raw prompt text still work.
    """
    import json

    if sys.stdin.isatty():
        return {}
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {"text": raw.strip()}
    return data if isinstance(data, dict) else {"text": str(data)}


def _clip_lines(lines: list[str], budget: int) -> list[str]:
    """Keep whole leading lines within a character budget (newlines counted)."""
    out: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + 1
        if used + cost > budget:
            break
        out.append(line)
        used += cost
    return out


def cmd_hook_session_start(service: MarvinService, args: argparse.Namespace) -> int:
    try:
        payload = _read_hook_stdin()
        if payload.get("source") == "resume":
            return 0  # context is intact on resume; re-injecting duplicates
        budget = args.budget_chars or service.settings.hook_session_budget_chars
        project = service.project_tag

        lines = ["marvin-memory auto-recall (session start):"]
        if project:
            lines.append(f"  project: {project}")

        # Embedder-free by design: vault listings only, so this hook stays
        # fast even on a cold cache. Project-tagged facts first, then by
        # confidence and recency.
        entries: list[tuple[tuple[int, float, float], str]] = []
        for note in service.vault.list_notes(MemoryKind.SEMANTIC):
            tagged = 0 if (project and project in note.metadata.tags) else 1
            for fact in note.metadata.facts:
                if fact.deprecated:
                    continue
                entries.append(
                    (
                        (tagged, -fact.confidence, -fact.created_at.timestamp()),
                        f"  fact: {fact.subject} {fact.predicate}: {fact.value}",
                    )
                )
        lines += [line for _, line in sorted(entries)[:24]]
        lines += [f"  recent: {hit.title} ({hit.kind.value})" for hit in service.recent(limit=3)]

        use = (
            "  use: marvin search <q> to recall; marvin remember <concept>"
            " --predicate <p> --value <v> to store"
        )
        clipped = _clip_lines(lines, max(budget - len(use) - 1, 0))
        clipped.append(use)
        print("\n".join(clipped))
        return 0
    except Exception as exc:  # never break the host session
        print(f"marvin hook error: {exc}", file=sys.stderr)
        return 0


def cmd_hook_user_prompt(service: MarvinService, args: argparse.Namespace) -> int:
    try:
        payload = _read_hook_stdin()
        prompt = str(payload.get("prompt") or payload.get("text") or " ".join(args.query or []))
        prompt = prompt.strip()
        if len(prompt) < 16 or prompt.startswith("/"):
            return 0
        budget = args.budget_chars or service.settings.hook_prompt_budget_chars

        from .promotion import detect_correction

        lines: list[str] = []
        signals = detect_correction(prompt)
        if signals:
            lines.append(
                f"memory: possible correction ({', '.join(signals)}) — once resolved, "
                'persist it: marvin remember "<concept>" --predicate <p> --value <new> '
                '--reason "user correction" (the old value is auto-deprecated)'
            )

        # Lexical-only recall: FTS + entity graph, no embedding-model load —
        # this runs on every prompt, so latency is the contract.
        hits = service.search(prompt, lexical_only=True, limit=4)
        if hits:
            lines.append("marvin-memory recall:")
            for hit in hits:
                excerpt = (hit.excerpt or "").replace("\n", " ").strip()[:160]
                lines.append(f"  {hit.title} ({hit.kind.value}): {excerpt}")

        if lines:
            print("\n".join(_clip_lines(lines, budget)))
        return 0
    except Exception as exc:  # never break the host session
        print(f"marvin hook error: {exc}", file=sys.stderr)
        return 0


# Hook wiring per host (verified against mid-2026 docs). Claude Code, Codex
# CLI, and Grok Build share the same 3-level schema (event -> matcher-group ->
# handler array) and the same contract: hook reads JSON on stdin, plain stdout
# on exit 0 is injected into model context. OpenCode and Amp have no shell
# hooks — only TypeScript plugin APIs — so `install` defers to `show` there.
_HOOK_EVENTS: tuple[tuple[str, str, int], ...] = (
    ("SessionStart", "marvin hook session-start", 30),
    ("UserPromptSubmit", "marvin hook user-prompt", 10),
)


def _hook_config() -> dict:
    return {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": command, "timeout": timeout}]}]
            for event, command, timeout in _HOOK_EVENTS
        }
    }


def _merge_hook_config(path: Path) -> list[str]:
    """Idempotently merge the marvin hook entries into a shared JSON config.

    Never touches unrelated keys or other tools' hooks; raises (and leaves the
    file untouched) if the existing JSON does not parse.
    """
    import json

    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8") or "{}")

    hooks = data.setdefault("hooks", {})
    added: list[str] = []
    for event, command, timeout in _HOOK_EVENTS:
        entries = hooks.setdefault(event, [])
        already = any(
            hook.get("command", "").startswith("marvin hook")
            for entry in entries
            if isinstance(entry, dict)
            for hook in entry.get("hooks", [])
            if isinstance(hook, dict)
        )
        if not already:
            entries.append({"hooks": [{"type": "command", "command": command, "timeout": timeout}]})
            added.append(event)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return added


def cmd_hooks_install(service: MarvinService | None, args: argparse.Namespace) -> int:
    import json

    host = args.host
    hints: list[tuple[str, str]] = [
        ("marvin hook session-start", "dry-run what gets injected at session start"),
        ('echo \'{"prompt":"..."}\' | marvin hook user-prompt', "dry-run prompt recall"),
    ]

    if host == "claude":
        base = Path.home() / ".claude" if args.user else Path.cwd() / ".claude"
        path = base / "settings.json"
        added = _merge_hook_config(path)
    elif host == "codex":
        base = Path.home() / ".codex" if args.user else Path.cwd() / ".codex"
        path = base / "hooks.json"
        added = _merge_hook_config(path)
        if not args.user:
            hints.append(("/hooks", "trust these project hooks inside Codex CLI (one-time)"))
    elif host == "grok":
        # Grok reads every JSON file under .grok/hooks/ — marvin owns its own
        # file there, so install is a plain (re)write, naturally idempotent.
        base = Path.home() / ".grok" if args.user else Path.cwd() / ".grok"
        path = base / "hooks" / "marvin.json"
        rendered = json.dumps(_hook_config(), indent=2) + "\n"
        existed = path.exists() and path.read_text(encoding="utf-8") == rendered
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        added = [] if existed else [event for event, _, _ in _HOOK_EVENTS]
        if not args.user:
            hints.append(("/hooks-trust", "trust these project hooks inside Grok Build (one-time)"))
    else:
        _print(
            encode_error(
                "usage",
                f"'{host}' has no shell-command hook mechanism (TypeScript plugin API only); "
                f"run `marvin hooks show --host {host}` for the plugin template",
            )
        )
        return 2

    _print(
        encode_kv(
            "hooks",
            {
                "host": host,
                "path": str(path),
                "added": ", ".join(added) if added else "(already installed)",
            },
        ),
        encode_help(hints),
    )
    return 0


_OPENCODE_PLUGIN = """\
# opencode has no shell-command hooks; per-prompt injection needs a TS plugin.
# WARNING: `experimental.chat.system.transform` is experimental and has been
# version-fragile (mutations silently discarded in some releases) — verify
# after opencode upgrades. Save as .opencode/plugins/marvin-memory.ts:

export const MarvinMemoryPlugin = async ({ $ }) => {
  return {
    "experimental.chat.system.transform": async (input, output) => {
      const result = await $`marvin hook session-start`.quiet()
      const text = result.text().trim()
      if (text) output.system.push(`<marvin-memory>\\n${text}\\n</marvin-memory>`)
    },
  }
}

# Static alternative (robust): pre-generate context into a file the config
# loads at session init, e.g. via your shell profile or a cron:
#   marvin hook session-start > .opencode/marvin-context.md
# and list it under the `instructions` key in opencode.json."""

_AMP_PLUGIN = """\
# Amp's declarative amp.hooks cannot run shell commands; context injection
# needs a TypeScript plugin. Save as .amp/plugins/marvin-memory.ts:

import type { PluginAPI } from '@ampcode/plugin'

export default function (amp: PluginAPI) {
  let sessionContext: string | null = null

  amp.on('session.start', async () => {
    const result = await amp.$`marvin hook session-start`
    sessionContext = result.stdout.trim() || null
  })

  amp.on('agent.start', async (event, ctx) => {
    const result = await ctx.$`marvin hook user-prompt ${event.prompt ?? ''}`
    let content = result.stdout.trim()
    if (sessionContext) {
      content = [sessionContext, content].filter(Boolean).join('\\n')
      sessionContext = null
    }
    if (content) return { message: { content, display: false } }
  })
}"""


def cmd_hooks_show(service: MarvinService | None, args: argparse.Namespace) -> int:
    import json

    host = args.host
    if host in ("claude", "codex", "grok"):
        target = {
            "claude": ".claude/settings.json (or ~/.claude/settings.json)",
            "codex": ".codex/hooks.json (or ~/.codex/hooks.json; TOML config.toml also works)",
            "grok": ".grok/hooks/marvin.json (or ~/.grok/hooks/marvin.json)",
        }[host]
        print(f"# {host}: merge into {target}")
        print(f"# (or just run: marvin hooks install --host {host})")
        print(json.dumps(_hook_config(), indent=2))
        return 0
    if host == "opencode":
        print(_OPENCODE_PLUGIN)
        return 0
    if host == "amp":
        print(_AMP_PLUGIN)
        return 0
    return _usage_error(f"unknown host: {host}")


def cmd_consolidate(service: MarvinService, args: argparse.Namespace) -> int:
    from .consolidation import ConsolidationEngine

    engine = ConsolidationEngine(
        model=args.model or service.settings.sleep_model,
        api_base=args.api_base or service.settings.sleep_api_base,
    )
    report = service.sleep(engine=engine, min_episodes=args.min_episodes, min_facts=args.min_facts)
    summary: dict[str, object] = {
        "notes_linked": report.notes_linked,
        "facts_extracted": len(report.facts),
        "insights_created": len(report.insights),
    }
    if report.extraction_skipped:
        summary["extraction_skipped"] = True
    _print(
        encode_kv("sleep", summary),
        encode_table(
            "facts",
            [_write_result_row(r) for r in report.facts],
            ["title", "kind", "path", "created"],
            empty="no entity crossed the episode threshold",
        ),
        encode_table(
            "insights",
            [_write_result_row(r) for r in report.insights],
            ["title", "kind", "path", "created"],
            empty="no aspect group had enough facts",
        ),
    )
    return 0


def _packaged_skill_dir() -> Path:
    from importlib.resources import files

    return Path(str(files("marvin") / "skill"))


def cmd_skill_show(service: MarvinService | None, args: argparse.Namespace) -> int:
    print((_packaged_skill_dir() / "SKILL.md").read_text(encoding="utf-8"))
    return 0


def cmd_skill_install(service: MarvinService | None, args: argparse.Namespace) -> int:
    import shutil

    if args.target:
        base = Path(args.target).expanduser()
    elif args.user:
        base = Path.home() / ".claude" / "skills"
    else:
        base = Path.cwd() / ".claude" / "skills"
    dest = base / "marvin-memory"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_packaged_skill_dir(), dest, dirs_exist_ok=True)
    _print(
        encode_kv("installed", {"skill": "marvin-memory", "path": str(dest)}),
        encode_help(
            [
                ("marvin skill show", "print the skill for pasting into other harnesses"),
            ]
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
    p.add_argument(
        "--reason",
        help="why the previous value is being replaced (recorded on the deprecated fact)",
    )
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
    p = sub.add_parser("doctor", help="Install-state checkup with exact fix commands")
    p.set_defaults(func=cmd_doctor, needs_service=False)

    hk = sub.add_parser(
        "hook",
        help="Auto-recall hook commands (fast, budgeted; wire via `marvin hooks install`)",
    )
    hk_sub = hk.add_subparsers(dest="hook_command", required=True)
    p = hk_sub.add_parser("session-start", help="Print session-start memory context (stdout)")
    p.add_argument("--budget-chars", type=int, default=None, dest="budget_chars")
    p.set_defaults(func=cmd_hook_session_start)
    p = hk_sub.add_parser(
        "user-prompt", help="Print prompt-relevant recall (stdout; lexical-only, no model load)"
    )
    p.add_argument("query", nargs="*", help="prompt text (default: hook JSON on stdin)")
    p.add_argument("--budget-chars", type=int, default=None, dest="budget_chars")
    p.set_defaults(func=cmd_hook_user_prompt)

    hks = sub.add_parser("hooks", help="Install or show host hook configuration")
    hks_sub = hks.add_subparsers(dest="hooks_command", required=True)
    p = hks_sub.add_parser("install", help="Wire the auto-recall hooks into a host's config")
    p.add_argument(
        "--host",
        default="claude",
        choices=["claude", "codex", "grok", "opencode", "amp"],
    )
    p.add_argument("--user", action="store_true", help="user-level config instead of project")
    p.set_defaults(func=cmd_hooks_install, needs_service=False)
    p = hks_sub.add_parser("show", help="Print the hook config snippet for manual setup")
    p.add_argument(
        "--host",
        default="claude",
        choices=["claude", "codex", "grok", "opencode", "amp"],
    )
    p.set_defaults(func=cmd_hooks_show, needs_service=False)

    p = sub.add_parser(
        "consolidate",
        help="Run the sleep pass now: entity extraction + two-phase consolidation",
    )
    p.add_argument("--model", help="LiteLLM model id override")
    p.add_argument("--api-base", dest="api_base")
    p.add_argument("--min-episodes", type=int, default=3, dest="min_episodes")
    p.add_argument("--min-facts", type=int, default=3, dest="min_facts")
    p.set_defaults(func=cmd_consolidate)

    sk = sub.add_parser("skill", help="The bundled agent skill: show / install")
    sk_sub = sk.add_subparsers(dest="skill_command", required=True)
    p = sk_sub.add_parser("show", help="Print SKILL.md (paste into any harness)")
    p.set_defaults(func=cmd_skill_show, needs_service=False)
    p = sk_sub.add_parser("install", help="Copy the skill into a skills directory")
    p.add_argument(
        "--user",
        action="store_true",
        help="install to ~/.claude/skills (default: ./.claude/skills)",
    )
    p.add_argument("--target", help="explicit skills directory")
    p.set_defaults(func=cmd_skill_install, needs_service=False)

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
    # forward it so existing agent configurations keep working. The server
    # module (and the mcp stack under it) is imported only on these two
    # dispatch paths — every other command, hooks especially, skips it.
    if argv and argv[0].startswith("--") and "--transport" in argv:
        print(
            "note: `marvin --transport ...` is deprecated; use `marvin serve --transport ...`",
            file=sys.stderr,
        )
        from . import server as _server

        _server.main(argv)
        return 0

    # Dispatch `serve` before parsing so its flags reach the server parser
    # verbatim (argparse REMAINDER mishandles leading option tokens on 3.12).
    if argv and argv[0] == "serve":
        from . import server as _server

        _server.main(argv[1:])
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    # Commands like `skill show|install` never touch the vault; don't create
    # a service (which would create vault directories) for them.
    if not getattr(args, "needs_service", True):
        try:
            return args.func(None, args)
        except Exception as exc:
            print(encode_error("runtime", f"{type(exc).__name__}: {exc}"))
            return 1

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
