# Marvin

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

**Markdown/Obsidian-native, Git-backed long-term memory for AI coding agents.**

Marvin gives AI agents (Claude, Goose, Cursor, etc.) durable memory that persists across sessions — stored as plain Markdown you can browse in [Obsidian](https://obsidian.md). It exposes 13 MCP tools over SSE/stdio, runs entirely local via Docker Compose, and consolidates noisy logs into permanent knowledge while you sleep.

> Named after [Marvin Minsky](https://en.wikipedia.org/wiki/Marvin_Minsky). Based on his *K-Lines* theory of memory — see the companion paper on [SSRN (Abstract #6234218)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6234218).

---

## Why Marvin?

AI coding agents are stateless by default. Every session starts from zero — context is lost, mistakes are repeated, and institutional knowledge never accumulates. Marvin solves this by giving agents a **structured, persistent memory system** inspired by Minsky's K-Lines paper:

- **Semantic** memories store durable facts ("the auth service uses JWT tokens")
- **Procedural** memories store reusable rules ("always run tests before committing")
- **Episodic** memories log what happened during a session
- **Reflective** memories capture design principles and lessons learned

A background worker continuously distills noisy episodes into permanent knowledge — the same way human memory consolidates during sleep.

## Features

- **🗄️ Obsidian-Native Vault** — All memories are clean Markdown files with YAML frontmatter, organized by type (`Semantic/`, `Procedural/`, `Episodic/`, `Reflective/`). Open the vault directly in Obsidian for full graph visualization.
- **🌿 Git-Backed Worktrees** — The vault is a Git repository. Agents can branch memory for risky tasks — if it fails, discard; if it succeeds, merge. Ground truth stays clean.
- **🔗 Deep Semantic Graphing** — The Brain Worker uses Google's [`langextract`](https://github.com/google/langextract) for zero-shot entity extraction, automatically injecting `[[Wikilinks]]` to build a dense, traversable knowledge graph.
- **😴 Computational Sleep** — A background worker reads noisy Episodic logs, distills them via a local LLM (Ollama), and writes permanent Semantic facts and Procedural rules — all while you're away.
- **🔍 Hybrid Retrieval** — SQLite-vec for vector search (local ONNX embeddings) + FTS5 for keyword search, combined via Reciprocal Rank Fusion (RRF).
- **🔌 MCP Gateway** — FastMCP server with 13 tools over SSE (port 8421) or stdio. Plug into any MCP-compatible agent.

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
docker exec -it marvin-ollama-1 ollama pull qwen3:8b
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

Full documentation is available at [**r0k3.github.io/marvin**](https://r0k3.github.io/marvin/).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/r0k3/marvin).

## License

MIT — see [LICENSE](LICENSE) for details.
