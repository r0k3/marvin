# CLI Reference (AXI)

`marvin` is an [AXI-style](https://axi.md/) command-line interface over the
memory system — designed so that an **agent** driving it through a shell
spends as few tokens as possible, while staying pleasant for humans.

The AXI conventions it follows:

- **Content first** — bare `marvin` prints a live vault dashboard, not help.
- **TOON output** — tables declare count and schema once
  (`hits[3]{title,kind,path}:`) then emit compact comma-joined rows: ~40%
  fewer tokens than JSON.
- **Minimal default schemas** — list output carries 3 fields by default;
  `--fields title,kind,path,excerpt` widens it.
- **Truncation with size hints** — long bodies end with
  `(truncated, N chars total — use --full to see complete body)`.
- **Definitive empty states** — `hits[0]: (no matches for "x")`, never a
  silent blank.
- **Structured errors and honest exit codes** — runtime failures print an
  `error[1]{code,message}` block on stdout and exit `1`; usage errors
  (including unknown flags) exit `2`; nothing ever prompts interactively.
- **Contextual disclosure** — output ends with a `help[]` block of concrete
  next-step command templates (`<placeholders>`, never guessed values).

Global flags: `--vault-path <dir>` and `--state-dir <dir>` (defaults come
from `MARVIN_VAULT_PATH` / `MARVIN_STATE_DIR` or `.env`).

## The dashboard

```text
$ marvin
vault:
  path: /home/you/vault
  notes: 42
  episodic: 17
  semantic: 18
  procedural: 4
  reflective: 3
  unconsolidated_episodes: 5
  indexed: 42
recent[3]{title,kind,path}:
  Fixed race in worker,episodic,Episodic/Fixed race in worker.md
  ...
help[4]:
  marvin search <query>       # hybrid recall across all four memory types
  marvin consolidate          # distill 5 unconsolidated episodes
  ...
```

The dashboard reads only the vault and the existing index — it never loads
an embedding model and never calls an LLM, so it is cheap enough to run as
**ambient context**: hook it into your agent's session start so the state of
memory is on screen before the first action. Pair it with the bundled
[agent skill](../guide/skills.md) (`marvin skill install`), which teaches the
agent *when* to act on what the dashboard shows. For Claude Code:

```json
{
  "hooks": {
    "SessionStart": [
      {"hooks": [{"type": "command", "command": "marvin"}]}
    ]
  }
}
```

## Commands

| Command | What it does |
|---|---|
| `marvin` | Live vault dashboard (see above) |
| `marvin search <query...> [--kind k] [--limit n] [--fields ...]` | Hybrid recall (keyword + vector + entity graph, rank-fused) |
| `marvin recent [--kind k] [--limit n]` | Most recent memories |
| `marvin read <identifier> [--full]` | One note by title, alias, or path — structured facts + body |
| `marvin remember <concept> [content] --predicate p --value v [--aspect a] [--confidence c]` | Store a semantic fact; same-predicate updates soft-deprecate the old value |
| `marvin procedure <title> --step s [--step s2 ...] [--applies ...] [--avoid ...]` | Store a procedure/rule |
| `marvin template register <title> --plan s [--intent i] [--style ...] [--entity-type ...] [--trigger ...] [--slot ...] [--failure ...]` | Register a K-line response strategy |
| `marvin template match [context...] [--intent i] [--top-k n]` | Select templates (weighted trigger scoring; intent is a hard gate) |
| `marvin template used <title> [--failure]` | Record an outcome; updates the effectiveness EMA |
| `marvin episode <title> --summary s [--details d]` | Log episodic memory |
| `marvin reflect <title> --insight i` | Store a reflection |
| `marvin session prepare <task...> [--repo r] [--tech t]` | Pull relevant context for the work you are starting |
| `marvin session finalize <title> --summary s [--fact f ...] [--reflection r ...]` | Closing episode + optional extractions |
| `marvin sync` / `marvin rebuild` / `marvin check` | Index maintenance (vault is authoritative; `check` exits 1 on drift) |
| `marvin health` | Runtime snapshot: backends, GPU, paths |
| `marvin consolidate [--model m] [--api-base u]` | Two-phase consolidation now, synchronously (episodic → semantic → reflective) |
| `marvin worktree start <branch>` / `marvin worktree merge <branch>` | Branch memory for risky work |
| `marvin skill show` | Print the bundled `marvin-memory` agent skill (paste into any harness) |
| `marvin skill install [--user\|--target <dir>]` | Install the skill into a skills directory (default: `./.claude/skills`) |
| `marvin serve [--transport stdio\|sse\|http] [--host] [--port]` | Run the MCP server (everything after `serve` is forwarded) |

Every subcommand answers `--help` concisely.

## Shell composability

TOON rows are line-oriented, so standard tools compose:

```bash
marvin search "database migrations" --limit 20 | grep semantic
marvin recent --kind episodic | head -5
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (including definitive empty results) |
| 1 | Runtime failure (structured `error[...]` block on stdout) or failed `check` |
| 2 | Usage error — unknown command, flag, or field (fail loud, never ignore) |

## Compatibility

The pre-0.3 server invocation `marvin --transport stdio ...` still works —
it is detected, a deprecation note goes to stderr, and the call is forwarded
to the server. New configurations should use `marvin serve --transport stdio`.
