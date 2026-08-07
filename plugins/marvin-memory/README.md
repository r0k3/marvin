# marvin-memory — Claude Code plugin

Durable, Obsidian-native, Git-backed memory for Claude Code, in one install:

| Component | What it gives you |
|---|---|
| **Hooks** | Auto-recall: session-start memory context + per-prompt lexical recall (~0.4 s, char-budgeted, never breaks a session) |
| **MCP server** | The full 20-tool `marvin` surface (`marvin serve --transport stdio`) |
| **Skill** `marvin-memory` | Teaches Claude *when* to store, recall, correct, and close the loop — pressure-tested against baseline behavior |
| **Slash commands** | `/marvin-memory:remember`, `:recall`, `:sleep`, `:status`, `:export` |
| **Agent** `memory-curator` | On-demand vault maintenance through the CLI (audit, consolidate, supersede with reasons) |

## Prerequisites

The plugin wraps the **`marvin` CLI**, which must be on `PATH`:

```bash
uv tool install marvin-memory          # or: git+https://github.com/r0k3/marvin
marvin doctor                          # verify the install
```

Recommended: pin one global vault so every project shares memory (the vault is
global by design; project auto-tags narrow recall):

```json
// ~/.claude/settings.json
{ "env": { "MARVIN_VAULT_PATH": "~/.marvin_vault" } }
```

Without it, Marvin resolves `./marvin_vault` relative to each project.

## Install

```
/plugin marketplace add r0k3/marvin
/plugin install marvin-memory@marvin
```

Restart Claude Code (hooks and MCP servers load at session start).

## Notes

- **Don't double-inject:** the plugin's hooks replace `marvin hooks install --host claude`. Use one or the other.
- The optional sleep pass (consolidation) needs `marvin-memory[consolidate]` and a litellm endpoint (`MARVIN_SLEEP_MODEL` / `MARVIN_SLEEP_API_BASE`, default local Ollama). `/marvin-memory:status` tells you what's missing.
- Every memory is a Markdown file in the vault — point Obsidian at it, `git log` it, audit it. Docs: <https://r0k3.github.io/marvin>.
