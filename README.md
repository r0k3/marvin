# Marvin

<p align="center">
  <img src="docs/assets/logo.svg" width="180" alt="Marvin">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Docs](https://img.shields.io/badge/docs-r0k3.github.io%2Fmarvin-blueviolet)](https://r0k3.github.io/marvin/)

**An active, Obsidian-native, Git-backed memory cluster for professional AI agents.**

Named in homage to **Marvin Minsky** and his foundational book [*The Society of Mind*](https://amzn.to/3PuFQ3K), Marvin transforms ephemeral LLM context windows into a durable, topologically connected knowledge base. It is a practical implementation of the cognitive architectures detailed in the research paper [*K-Lines: A Framework for LLM Agent Memory*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6234218) (and its [companion repository](https://github.com/r0k3/k-lines)).

---

## Why Marvin?

Most agentic "memory skills" today simply dump chat logs into a hidden SQLite database or a black-box vector store. This is sufficient for casual chatbots, but it breaks down rapidly in professional software engineering workflows. When you take your agent's memories seriously, you need **ergonomics, interpretability, and safety**.

Marvin is built around three core philosophies:

### 1. The Ergonomics of Obsidian (Markdown)
Instead of locking agent memories inside a proprietary database, Marvin writes everything as clean, human-readable **Markdown files with YAML frontmatter** — organized into `Semantic/`, `Procedural/`, `Episodic/`, and `Reflective/` folders. Point your [Obsidian](https://obsidian.md/) vault at Marvin's storage directory and you instantly get a visual graph of everything your agent has learned. You can manually edit, co-author, and browse the knowledge base alongside your AI.

### 2. Safety through Git (Agentic Worktrees)
Agents hallucinate and explore dead ends. Because Marvin's vault is natively backed by **Git**, agents can check out isolated worktree branches. If a task succeeds, the memory branch is merged. If it fails, the branch is discarded. Ground truth stays clean, with full `git blame` for your agent's thoughts.

### 3. Asynchronous Consolidation (Computational Sleep)
Biological memory isn't just stored; it is *consolidated* while we sleep. Marvin uses a **NATS** message broker and a background **Brain Worker** — while your agent rapidly logs noisy Episodic events, the Brain Worker asynchronously extracts entities via [`langextract`](https://github.com/google/langextract), injects `[[Wikilinks]]`, and distills raw logs into permanent Semantic facts and Procedural rules using a local LLM (`qwen3.5:9b`).

## Features

- **Obsidian-Native Vault** — Clean Markdown with YAML frontmatter, full graph visualization in Obsidian.
- **Git-Backed Worktrees** — Branch memory for risky tasks; merge on success, discard on failure.
- **Deep Semantic Graphing** — Zero-shot entity extraction with automatic `[[Wikilink]]` injection.
- **Computational Sleep** — Background consolidation via local open-weight models (Ollama).
- **Hybrid Retrieval** — SQLite-vec (ONNX embeddings) + FTS5 keyword search, combined via Reciprocal Rank Fusion.
- **Cross-Encoder Reranking** — Optional `bge-reranker-v2-m3` pass (int8 ONNX, ~570 MB) that lifts LongMemEval-S `NDCG@10` from 88.9% → 94.6% and `MRR` from 89.5% → 95.3% (+13 pp / +11 pp on the hardest multi-session slice).
- **MCP Gateway** — 13 tools over SSE (port 8421) or stdio. Plug into any MCP-compatible agent.
- **Reproducible Benchmark** — Built-in [LongMemEval-S](https://arxiv.org/abs/2410.10813) harness so retrieval changes are measured, not vibes (BM25 baseline: `recall_any@5 = 95.6%`).

## Architecture

Marvin runs as four lightweight services orchestrated by Docker Compose:

| Service | Role |
|---|---|
| **MCP Gateway** | FastMCP server exposing 13 tools via SSE on port `8421`. The low-latency edge your agent talks to. |
| **NATS** | High-performance message broker with JetStream. Streams `memory.created` and `memory.sleep` events. |
| **Brain Worker** | Background daemon subscribing to NATS. Runs entity extraction (langextract) and sleep consolidation. |
| **Ollama** | Bundled local LLM container. Provides cost-free inference for the Brain Worker's consolidation phase. |

```text
+-----------+        MCP (SSE / stdio)         +--------------------+
|           | -------------------------------->|                    |
|   Agent   |                                  |   MCP Gateway      | ---> (Writes) ---> [ Git-Backed Vault ]
|  (Goose)  | <------------------------------- |   (FastMCP API)    | <--- (Reads)  <--- [ SQLite-Vec Index ]
|           |        (Search / Tools)          +--------------------+
+-----------+                                           |
                                                        |
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
[ Local Ollama ] <--- (Distill Facts/Rules) ---   Brain Worker     |
                                              |                    | ---> (Update Links & Commit to Vault)
                                              +--------------------+
```

## Quick Start

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

```bash
git clone https://github.com/r0k3/marvin.git
cd marvin

# Start the Marvin cluster
docker compose up -d

# Pull the local LLM model (one-time setup)
docker exec -it marvin-ollama-1 ollama pull qwen3.5:9b
```

The MCP Gateway is now listening on `http://localhost:8421/sse`.

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

For best results, load the **Marvin Skill** into your agent's system prompt. It teaches the agent to proactively use memory — searching before acting, logging episodes, branching for risky tasks, and triggering sleep consolidation.

The skill is at [`src/marvin/skill.md`](src/marvin/skill.md). In Goose, add it as a skill file. In other agents, paste its contents into your system prompt or custom instructions.

## Documentation

Full documentation — architecture deep-dives, getting started guide, agent skills reference, and case studies — is available at **[r0k3.github.io/marvin](https://r0k3.github.io/marvin/)**.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/r0k3/marvin).

## License

MIT — see [LICENSE](LICENSE) for details.
