# Marvin

<img src="docs/assets/logo.svg" width="140" alt="Marvin logo" align="right">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-r0k3.github.io%2Fmarvin-blue)](https://r0k3.github.io/marvin/)

Long-term memory for AI agents, kept in plain Markdown.

Marvin stores what your agent learns as Markdown files with YAML frontmatter, in a vault you can open in Obsidian, grep, diff, and keep under Git. Recall runs on a local hybrid index: SQLite full-text search, vector similarity, and a wikilink entity graph, fused and optionally reranked. Storing a memory never calls an LLM, and neither does recalling one. Both work offline and cost nothing per call.

The name honors Marvin Minsky. The design follows the paper [K-Lines: A Cognitively-Grounded Four-Memory Architecture for Persistent Conversational AI](https://ssrn.com/abstract=6234218), and this repository is its reference implementation.

## Install

```bash
uv tool install git+https://github.com/r0k3/marvin
```

Try it:

```bash
marvin remember "DB" --predicate storage --value "PostgreSQL with asyncpg"
marvin search "postgres"
marvin            # dashboard: counts per memory type, recent notes, suggested next steps
marvin doctor     # what's installed, what's missing, and the command that fixes it
```

`marvin --help` lists every command. Output is TOON, a compact tabular format that agents parse cheaply and humans read fine.

## Using it from an agent

**Claude Code.** Install the plugin. It brings the MCP server (20 tools), hooks that inject relevant memories at session start and on each prompt, a skill that teaches the agent when to store and recall, and `/marvin-memory:*` commands:

```
/plugin marketplace add r0k3/marvin
/plugin install marvin-memory@marvin
```

Set one global vault so all projects share memory: add `"env": {"MARVIN_VAULT_PATH": "~/.marvin_vault"}` to `~/.claude/settings.json`. See [plugins/marvin-memory](plugins/marvin-memory/) for details.

**Any MCP client.** Run the server over stdio:

```json
{
  "mcpServers": {
    "marvin": {
      "command": "marvin",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

`marvin serve --transport sse` serves HTTP on port 8421 instead, for clients like Goose and Cursor that connect by URL.

**Codex CLI and Grok Build.** `marvin hooks install --host codex` (or `grok`) wires the same auto-recall hooks into those tools. One caveat we found in testing: Grok up to 0.2.118 runs the hooks but ignores their output, so use the skill there instead (`marvin skill install --host grok`).

**Anything else.** `marvin skill show` prints the skill for pasting into other harnesses, and `marvin skill export` compiles your vault into a self-contained skill bundle that needs no server at all.

## How memory is organized

| Kind | Holds | Example |
|---|---|---|
| Episodic | events and completed work | "Fixed the double-ack race in the worker" |
| Semantic | durable facts | DB storage: PostgreSQL with asyncpg |
| Procedural | playbooks and response strategies | a release checklist; a debugging template |
| Reflective | lessons that cut across facts | "test host contracts against the binary, not the docs" |

Facts are structured: predicate, value, aspect, confidence. Storing a new value for the same predicate deprecates the old one instead of overwriting it, so corrections keep an audit trail (`--reason "user correction: ..."` records why). Writes made inside a git repository are tagged with the project automatically. The vault stays global on purpose, so what the agent learns in one project is available in the next.

## The sleep pass

Raw episodes accumulate fast and noisy. The sleep pass distills them offline: a local LLM extracts entities and links them as `[[wikilinks]]`, pulls stable facts out of recurring episodes, and synthesizes higher-level insights from accumulated facts. Your agent's write path never waits for any of this.

```bash
uv tool install 'marvin-memory[consolidate] @ git+https://github.com/r0k3/marvin'
marvin consolidate
```

The default model is a local one via Ollama (`qwen3.6:35b-a3b-q4_K_M`); any litellm-compatible endpoint works (`MARVIN_SLEEP_MODEL`, `MARVIN_SLEEP_API_BASE`). Progress lives in the vault itself as frontmatter flags, so the pass picks up where it left off no matter where it runs: on demand, inside the serve process, or in the optional NATS worker cluster (`docker compose up -d`) for always-on setups.

## Recall quality

LongMemEval-S, all 500 questions, ~115k-token haystacks:

| Metric | Score |
|---|---|
| recall_any@5 | 99.6% |
| NDCG@10 | 95.3% |
| End-to-end QA, fully local reader | 82.8% |

The retrieval layer is what makes a local model viable here: handed the full history instead, the same reader scores 45.8%. Reproduction commands and the judge protocol are in the [evaluation guide](https://r0k3.github.io/marvin/guide/evaluation/).

## Safety

Risky work can branch its memory: `marvin worktree start <name>` gives the agent an isolated Git branch of the vault, merged on success or discarded on failure. Every write is stripped of invisible Unicode (the bidi-control smuggling class of tricks), and text produced by the sleep-pass LLM is scanned for prompt-injection patterns before it may enter the vault. Notes you or your agent write directly are never content-filtered.

## Documentation

The full docs are at [r0k3.github.io/marvin](https://r0k3.github.io/marvin/): [getting started](https://r0k3.github.io/marvin/guide/getting-started/), [CLI reference](https://r0k3.github.io/marvin/reference/cli/), [MCP tools](https://r0k3.github.io/marvin/reference/mcp-tools/), [the agent skill](https://r0k3.github.io/marvin/guide/skills/), and [evaluation](https://r0k3.github.io/marvin/guide/evaluation/). How it works inside is in [ARCHITECTURE.md](ARCHITECTURE.md).

There is a small demo: `uv run python -m marvin.eval.demo` runs four-memory retrieval over a bundled *A Midsummer Night's Dream* vault.

## Citation

If you use Marvin in research, please cite:

> Kende, Robert. *K-Lines: A Cognitively-Grounded Four-Memory Architecture for Persistent Conversational AI.* SSRN, February 2026. <https://ssrn.com/abstract=6234218>

<details>
<summary>BibTeX</summary>

```bibtex
@misc{kende2026klines,
  author       = {Kende, Robert},
  title        = {K-Lines: A Cognitively-Grounded Four-Memory Architecture for Persistent Conversational AI},
  year         = {2026},
  month        = feb,
  howpublished = {SSRN},
  doi          = {10.2139/ssrn.6234218},
  url          = {https://ssrn.com/abstract=6234218}
}
```

</details>

## License

MIT. See [LICENSE](LICENSE).
