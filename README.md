# Marvin

<p align="center">
  <img src="docs/assets/logo.svg" width="180" alt="Marvin">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Docs](https://img.shields.io/badge/docs-r0k3.github.io%2Fmarvin-blueviolet)](https://r0k3.github.io/marvin/)

**An active, Obsidian-native, Git-backed memory system for AI agents.**

Named in homage to **Marvin Minsky** and his foundational book [*The Society of Mind*](https://amzn.to/3PuFQ3K), Marvin turns ephemeral LLM context windows into a durable, topologically connected knowledge base. It is the reference implementation of the research paper [*K-Lines: A Cognitively-Grounded Four-Memory Architecture for Persistent Conversational AI*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6234218) (and its [companion repository](https://github.com/r0k3/k-lines)).

**Headline results on [LongMemEval-S](https://arxiv.org/abs/2410.10813)** (500 questions, ~115k-token haystacks): **99.6% `recall_any@5`** retrieval, and **82.8% end-to-end QA accuracy with a fully local, quantized reader** — within ~10 points of frontier-cloud-reader systems, with zero LLM calls on the write or query path and no cloud dependency. [Full numbers →](https://r0k3.github.io/marvin/guide/evaluation/)

---

## Why Marvin?

Most agentic "memory skills" dump chat logs into a hidden SQLite database or a black-box vector store. That works for casual chatbots, but breaks down in professional workflows. When you take your agent's memories seriously, you need **ergonomics, interpretability, and safety**.

### 1. The Ergonomics of Obsidian (Markdown)
Marvin writes every memory as a clean, human-readable **Markdown file with YAML frontmatter**, organized into `Episodic/`, `Semantic/`, `Procedural/`, and `Reflective/` folders. Point [Obsidian](https://obsidian.md/) at the vault and you get a visual graph of everything your agent has learned. You can edit, co-author, and browse the knowledge base alongside your AI.

### 2. Safety through Git (Agentic Worktrees)
Agents hallucinate and explore dead ends. Because the vault is natively backed by **Git**, agents check out isolated worktree branches for risky tasks: merge on success, discard on failure. Every memory write is a commit — `git blame`, `git diff`, and `git revert` work on your agent's thoughts. A polluted memory is one revert away from clean, not a database surgery project.

### 3. Asynchronous Consolidation (Computational Sleep)
Biological memory is *consolidated* during sleep. Marvin's background **Brain Worker** (driven by a **NATS** broker) extracts entities via [`langextract`](https://github.com/google/langextract), injects `[[Wikilinks]]`, and runs a two-phase consolidation with a local LLM: entity-scoped **semantic facts** are distilled from raw episodic logs, then cross-fact **reflective insights** are synthesized per aspect. Your agent's write path stays fast; the thinking happens offline.

## The four memories

| Kind | Holds | Mechanism |
|---|---|---|
| **Episodic** | Raw events: tasks, bugs, sessions | Logged fast, marked `consolidated` once distilled |
| **Semantic** | Durable facts | Structured facts (predicate / value / aspect / confidence); a new value for the same predicate **soft-deprecates** the old one — auditable, never silently overwritten, excluded from retrieval |
| **Procedural** | Playbooks and rules | Plain procedures, plus **K-line templates**: response strategies with trigger conditions (intents / styles / entity types / keywords), selected by weighted partial-match scoring and ranked by adaptive effectiveness |
| **Reflective** | Cross-cutting insights | Synthesized from accumulated facts during sleep, with provenance links |

## Features

- **Obsidian-Native Vault** — clean Markdown + YAML frontmatter; full graph visualization in Obsidian.
- **Git-Backed Worktrees** — branch memory for risky tasks; merge on success, discard on failure.
- **Structured Semantic Facts** — stable fact IDs, predicates, aspects, confidence, and soft deprecation with `replaced_by` linking.
- **K-Line Procedural Templates** — Minsky-style partial reactivation: weighted trigger scoring plus an ACT-R-style utility (usage count + effectiveness EMA), surfaced automatically by `marvin_prepare_session`.
- **Deep Semantic Graphing** — zero-shot entity extraction with automatic `[[Wikilink]]` injection; the wikilink graph feeds retrieval.
- **Computational Sleep** — two-phase background consolidation using local open-weight models (default `qwen3.6:35b-a3b-q4_K_M` via Ollama).
- **Hybrid Retrieval** — three-stream Reciprocal Rank Fusion (SQLite FTS5 + `sqlite-vec` dense vectors + IDF-weighted entity graph), optional `bge-reranker-v2-m3` cross-encoder (int8 on CPU, fp16 on GPU), and opt-in time-aware freshness decay.
- **MCP Gateway** — 20 tools (the full service surface) over SSE (port 8421) or stdio; plugs into any MCP-compatible agent.
- **AXI CLI** — the same functionality as an [axi.md](https://axi.md/)-style command line: token-efficient TOON output, a live dashboard on bare `marvin`, `help[]` next-step hints, structured errors. Built for agents driving a shell, pleasant for humans.
- **Reproducible Benchmark** — built-in LongMemEval-S harness for retrieval *and* end-to-end QA, so memory changes are measured, not vibed.

## Benchmarks

LongMemEval-S, cleaned release, all 500 questions. Retrieval: hybrid three-stream RRF + cross-encoder rerank. QA: top-10 retrieved sessions read by a **local** `qwen3.6:35b-a3b` (q4_K_M) with JSON + Chain-of-Note, graded under the official per-question-type judge protocol.

| Metric | Score |
|---|---|
| `recall_any@5` | **99.6%** |
| NDCG@10 | 95.3% |
| MRR | 95.5% |
| **End-to-end QA accuracy (local reader)** | **82.8%** |

Two facts worth knowing: the published SOTA cluster (90–93%) is driven by frontier **cloud** readers, and in our controlled ablation the same local reader scores **45.8%** when handed the full ~126k-token history versus **81.2%** with Marvin's retrieved top-10 (~16k tokens) — the memory layer is what makes a local model accurate at all. Reproduction commands, per-type breakdowns, and the judge protocol are in the [evaluation guide](https://r0k3.github.io/marvin/guide/evaluation/).

## Architecture

Marvin runs as four lightweight services orchestrated by Docker Compose:

| Service | Role |
|---|---|
| **MCP Gateway** | FastMCP server exposing 20 tools via SSE on port `8421`. The low-latency edge your agent talks to. |
| **NATS** | High-performance message broker with JetStream. Streams `memory.created` and `memory.sleep` events. |
| **Brain Worker** | Background daemon subscribing to NATS. Runs entity extraction (langextract) and two-phase sleep consolidation. |
| **Ollama** | Bundled local LLM container. Cost-free inference for the Brain Worker's consolidation phases. |

```text
+-----------+        MCP (SSE / stdio)         +--------------------+
|           | -------------------------------->|                    |
|   Agent   |                                  |   MCP Gateway      | ---> (Writes) ---> [ Git-Backed Vault ]
|           | <------------------------------- |   (FastMCP API)    | <--- (Reads)  <--- [ SQLite-Vec Index ]
|           |        (Search / Tools)          +--------------------+
+-----------+                                           |
                                              [ Publish 'memory.created' ]
                                              [ Publish 'memory.sleep'   ]
                                                        v
                                              +--------------------+
                                              |    NATS Broker     |
                                              +--------------------+
                                                        |
                                              [ Consume Events ]
                                                        v
                                              +--------------------+
                                              |                    | ---> (Extract Entities) -> [ LangExtract ]
[ Local Ollama ] <--- (Consolidate 2-phase) --|    Brain Worker    |
                                              |                    | ---> (Update Links & Commit to Vault)
                                              +--------------------+
```

More detail in [ARCHITECTURE.md](ARCHITECTURE.md) and the [docs site](https://r0k3.github.io/marvin/architecture/).

## Quick Start

### Lightweight (CLI + MCP over stdio)

```bash
uv tool install git+https://github.com/r0k3/marvin
```

You immediately have the AXI command line:

```bash
marvin remember "DB" --predicate storage --value "PostgreSQL with asyncpg"
marvin search "postgres"                 # TOON output: hits[1]{title,kind,path}: ...
marvin skill install                     # teach your agent when to use all this
```

Bare `marvin` is a live dashboard, not help text — token-efficient TOON your agent (or you) can read at a glance:

```text
$ marvin
vault:
  path: ~/.marvin_vault
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
  marvin search <query>   # hybrid recall across all four memory types
  marvin consolidate      # distill 5 unconsolidated episodes
  ...
```

And the MCP server for your agent:

```json
{
  "mcpServers": {
    "marvin": {
      "command": "marvin",
      "args": ["serve", "--vault-path", "~/.marvin_vault", "--transport", "stdio"]
    }
  }
}
```

This gives you the vault, hybrid retrieval, and all 20 tools — without the background worker.

### Full cluster (Docker, with the Brain Worker)

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

```bash
git clone https://github.com/r0k3/marvin.git
cd marvin

# Start the Marvin cluster
docker compose up -d

# Pull the local consolidation model (one-time; any litellm-supported model works)
docker exec -it marvin-ollama-1 ollama pull qwen3.6:35b-a3b-q4_K_M
```

The MCP Gateway is now listening on `http://localhost:8421/sse`.

### GPU acceleration (optional)

```bash
uv pip install 'marvin[gpu]'   # Linux + NVIDIA (CUDA 12.x): embeddings + fp16 reranker on GPU
```

## Agent Configuration

### Goose

Add to `~/.config/goose/config.yaml`:

```yaml
extensions:
  marvin:
    enabled: true
    type: sse
    name: marvin
    uri: http://localhost:8421/sse
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "marvin": {
      "url": "http://localhost:8421/sse"
    }
  }
}
```

### Claude Code / Copilot CLI

Add to your MCP config:

```json
{
  "mcpServers": {
    "marvin": {
      "type": "sse",
      "url": "http://localhost:8421/sse"
    }
  }
}
```

## The Agent Skill

Marvin ships with a first-class **agent skill** (`marvin-memory`) that teaches the agent *when* to use memory without being told: which signals to store (and as which memory type), recall-before-answering, closing the template feedback loop after "that worked", session lifecycle, and what never to store. It covers both surfaces — the `marvin_*` MCP tools and the CLI — and was pressure-tested against baseline agent behavior.

```bash
marvin skill install          # → ./.claude/skills/marvin-memory (Claude Code, project)
marvin skill install --user   # → ~/.claude/skills/marvin-memory
marvin skill show             # print SKILL.md to paste into any other harness
```

Source: [`src/marvin/skill/SKILL.md`](src/marvin/skill/SKILL.md).

## Try the demo

```bash
uv run python -m marvin.eval.demo
```

Runs annotated four-memory retrieval over the bundled *A Midsummer Night's Dream* vault ([`examples/demo_vault`](examples/demo_vault)) — episodic scenes, semantic facts, a procedural analysis strategy, and a reflective insight, each answering the query type it should.

## Documentation

Full documentation is at **[r0k3.github.io/marvin](https://r0k3.github.io/marvin/)**:

- [Getting started](https://r0k3.github.io/marvin/guide/getting-started/) — install, Docker cluster, agent configs
- [MCP tools reference](https://r0k3.github.io/marvin/reference/mcp-tools/) — all 20 tools
- [CLI reference (AXI)](https://r0k3.github.io/marvin/reference/cli/) — every command, TOON format, exit codes, session-hook pattern
- [The agent skill](https://r0k3.github.io/marvin/guide/skills/) — what `marvin-memory` teaches and the test-first evidence behind it
- [Architecture](https://r0k3.github.io/marvin/architecture/) and [evaluation methodology](https://r0k3.github.io/marvin/guide/evaluation/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/r0k3/marvin).

## License

MIT — see [LICENSE](LICENSE) for details.
