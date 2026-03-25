# Marvin V2 Architecture

Marvin is designed to bridge the gap between an LLM agent's ephemeral context window and the requirement for long-term, durable reasoning. It operates as an **Obsidian-native, Git-backed memory cluster**.

## Core Design Principles

1. **Interpretability & Portability:** All memories are plain-text Markdown files with YAML frontmatter. The storage directory is directly compatible with the Obsidian note-taking app. No proprietary binary blobs obscure the user's data.
2. **Computational Sleep:** Ephemeral agent actions (logs, bugs, conversations) are noisy. Marvin features a background worker that periodically "sleeps," using an LLM to distill noisy logs into permanent facts and rules.
3. **Agentic Worktrees:** Codebases use Git branches to isolate experiments. Marvin applies this to memory. When an agent attempts a risky task, it can branch its memory. If the task fails, the memory branch is abandoned, keeping the ground truth pure.
4. **Deep Semantic Graphing:** Vector search is good for fuzziness, but exact relationships matter. A background NLP worker extracts entities from text using LLM extraction (`langextract`) and hard-links them using `[[Wikilinks]]`, creating a mathematically traversable knowledge graph.

## System Components

Marvin V2 operates as a suite of lightweight services orchestrated via Docker Compose.

### 1. The MCP Gateway (`server.py` & `service.py`)
*   **Role:** The front-facing edge that integrates with the client agent (e.g., Goose, Cursor) using the Model Context Protocol (MCP).
*   **Transport:** Exposes endpoints via SSE (Server-Sent Events) over HTTP, with a fallback to `stdio` for basic CLI runners.
*   **Data Flow:** Receives memory write/search requests. Writes raw Markdown to the Vault. Queries the `sqlite-vec` index for fast retrieval. Publishes asynchronous events to the Message Broker.

### 2. The Hybrid Index (`index.py` & `embeddings.py`)
*   **Role:** Extremely fast retrieval engine that operates natively over the Markdown vault.
*   **Technology:** Uses `sqlite-vec` for embedded vector search and SQLite FTS5 for full-text keyword search.
*   **Mechanism:** Chunks Markdown files, generates ONNX-based local embeddings (via `fastembed`), and combines vector similarity with keyword ranking using **Reciprocal Rank Fusion (RRF)**.

### 3. Git-Backed Vault Management (`vault.py` & `git.py`)
*   **Role:** The single source of truth for all memory.
*   **Technology:** Local file system + `gitpython`.
*   **Mechanism:** Automatically initializes the `marvin_vault` as a Git repository. Exposes tools allowing the agent to create and merge branches (`marvin_start_worktree`).

### 4. The Message Broker (`broker.py`)
*   **Role:** Decouples the low-latency Gateway from the heavy NLP and LLM processing tasks.
*   **Technology:** **NATS** (with JetStream for durability).
*   **Events:** Streams events like `memory.created` (when an agent logs something new) and `memory.sleep` (when consolidation is triggered).

### 5. The Brain Worker (`worker.py`)
*   **Role:** The asynchronous daemon that makes Marvin "intelligent."
*   **Technology:** Python, subscribing to NATS.
*   **Task A (Entity Graphing):** Upon `memory.created`, it uses **Google's `langextract`** (`extraction.py`) to perform zero-shot Named Entity Recognition. It then safely modifies the source Markdown file to wrap discovered entities in `[[Obsidian Links]]`.
*   **Task B (Consolidation):** Upon `memory.sleep`, it fetches unconsolidated `Episodic` notes and passes them through an LLM.

### 6. The Consolidation Engine (`consolidation.py`)
*   **Role:** The cognitive backend used by the Brain Worker.
*   **Technology:** `litellm` bridging to a local **Ollama** container (e.g., running `qwen3.5:9b`).
*   **Mechanism:** Follows a strict JSON-enforced prompt to extract high-level `Semantic` facts (architectural truths) and `Procedural` rules (coding conventions) from a list of raw episodes.

## Data Flow Diagram

```text
+-----------+        MCP (SSE / stdio)         +--------------------+
|           | -------------------------------> |                    |
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
[ Local Ollama ] <--- (Distill Facts/Rules) ---    Brain Worker    |
                                              |                    | ---> (Update Links & Commit to Vault)
                                              +--------------------+
```
