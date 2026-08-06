# Marvin V2 Architecture

Marvin is designed to bridge the gap between an LLM agent's ephemeral context window and the requirement for long-term, durable reasoning. It operates as an **Obsidian-native, Git-backed memory system**.

## Deployment Profiles

Marvin ships one binary with two profiles:

* **Single process (default).** `marvin` (CLI) and `marvin serve` (MCP) run everything in one process: vault, hybrid index, and — when the `consolidate` extra is installed — the full sleep pass, either on demand (`marvin consolidate`, `marvin_consolidate`) or in the background of the serve process (`marvin_trigger_sleep` schedules it on the in-process event bus). Nothing else to deploy; the base install has no broker, no LLM client, no server framework.
* **Cluster (`[cluster]` extra + Docker Compose).** For always-on background processing across processes: the gateway publishes events to **NATS** and a separate **Brain Worker** consumes them, with a bundled Ollama container for local inference. Set `MARVIN_BUS=nats`.

Either way, **the vault itself is the work queue**: episodes carry a `consolidated` flag and every note an `extracted` flag, so a crash or missed event loses nothing — the next sleep pass picks up exactly where the vault says work is pending. Scaling follows Git, not a broker: replicate the vault via a Git remote and each node rebuilds its index locally.

## Core Design Principles

1. **Interpretability & Portability:** All memories are plain-text Markdown files with YAML frontmatter. The storage directory is directly compatible with the Obsidian note-taking app. No proprietary binary blobs obscure the user's data.
2. **Computational Sleep:** Ephemeral agent actions (logs, bugs, conversations) are noisy. Marvin "sleeps" — on demand, in the background of the serve process, or via the cluster worker — using an LLM to distill noisy logs into permanent facts and rules. The write and query paths never wait on it.
3. **Agentic Worktrees:** Codebases use Git branches to isolate experiments. Marvin applies this to memory. When an agent attempts a risky task, it can branch its memory. If the task fails, the memory branch is abandoned, keeping the ground truth pure.
4. **Deep Semantic Graphing:** Vector search is good for fuzziness, but exact relationships matter. The sleep pass extracts entities from text using LLM extraction (`langextract`) and hard-links them using `[[Wikilinks]]`, creating a mathematically traversable knowledge graph.

## System Components

### 1. The MCP Gateway & AXI CLI (`server.py`, `cli.py` & `service.py`)
*   **Role:** The front-facing edges that integrate with the client agent. The MCP gateway exposes the full service surface as **20 tools** over the Model Context Protocol; the **AXI CLI** (`marvin`) exposes the same service to shell-driving agents and humans, with token-efficient TOON output and a live dashboard on bare invocation.
*   **Transport:** MCP via SSE (Server-Sent Events) over HTTP or `stdio` (`marvin serve`); the CLI runs directly against the local vault.
*   **Data Flow:** Receives memory write/search requests. Writes raw Markdown to the Vault. Queries the `sqlite-vec` index for fast retrieval. Publishes asynchronous events on the event bus.

### 2. The Hybrid Index (`index.py`, `embeddings.py` & `reranker.py`)
*   **Role:** Extremely fast retrieval engine that operates natively over the Markdown vault.
*   **Technology:** Uses `sqlite-vec` for embedded vector search, SQLite FTS5 for full-text keyword search, and an `entity_edges` table hydrated from `[[Wikilinks]]`.
*   **Mechanism:** Chunks Markdown files, generates ONNX-based local embeddings (via `fastembed`), and fuses three streams with **Reciprocal Rank Fusion (RRF)**: vector similarity, keyword ranking, and an IDF-weighted entity-graph ranking. An optional cross-encoder reranker (`bge-reranker-v2-m3`, int8 on CPU / fp16 on GPU) re-scores the fused candidates, and an opt-in time-aware freshness boost favours recent episodic notes. Deprecated semantic facts are excluded at chunking time, so corrected knowledge never resurfaces in search.

### 3. Git-Backed Vault Management (`vault.py` & `git.py`)
*   **Role:** The single source of truth for all memory.
*   **Technology:** Local file system + `gitpython`.
*   **Mechanism:** Automatically initializes the `marvin_vault` as a Git repository. Exposes tools allowing the agent to create and merge branches (`marvin_start_worktree`).

### 4. The Event Bus (`bus.py`)
*   **Role:** Decouples the low-latency Gateway from the heavy NLP and LLM processing tasks. Events are a *nudge*, never the source of truth — pending work is durably marked in the vault itself (`consolidated` / `extracted` flags).
*   **Technology:** `InProcessBus` (default): handlers run as fire-and-forget asyncio tasks inside the serve process. `NatsBus` (`MARVIN_BUS=nats`, `[cluster]` extra): **NATS** with JetStream, for a separate worker process.
*   **Events:** `memory.created` (when an agent logs something new) and `memory.sleep` (when a sleep pass is requested).

### 5. The Sleep Pass (`service.sleep`) and the Brain Worker (`worker.py`)
*   **Role:** The offline processing that makes Marvin "intelligent": entity extraction first (its `[[Wikilinks]]` drive the entity-scoped grouping), then two-phase consolidation.
*   **Task A (Entity Graphing):** Uses **Google's `langextract`** (`extraction.py`) for zero-shot Named Entity Recognition over every note not yet marked `extracted`, safely rewriting the source Markdown to wrap discovered entities in `[[Obsidian Links]]`.
*   **Task B (Consolidation):** Fetches unconsolidated `Episodic` notes and passes them through an LLM (see §6).
*   **Where it runs:** in-process (`marvin consolidate`, the `marvin_consolidate` tool, or the in-process `memory.sleep` handler) — or, in the cluster profile, in the **Brain Worker**, an asynchronous daemon subscribing to NATS.

### 6. The Consolidation Engine (`consolidation.py`)
*   **Role:** The cognitive backend of the sleep pass. Requires the `consolidate` extra.
*   **Technology:** `litellm` bridging to any supported endpoint; defaults to local **Ollama** (`qwen3.6:35b-a3b-q4_K_M`), configurable via `MARVIN_SLEEP_MODEL` / `MARVIN_SLEEP_API_BASE`.
*   **Mechanism:** Two JSON-enforced phases. **Phase 1 (episodic → semantic):** episodes are grouped by the entities they mention; once an entity crosses a minimum-episode threshold, atomic facts (predicate / value / aspect / confidence) are extracted, deduplicated against the entity's known facts, and persisted — consumed episodes are marked `consolidated` so they are never re-processed. **Phase 2 (semantic → reflective):** accumulated facts are grouped by aspect and synthesized into higher-order reflective insights (patterns, gaps, anomalies, lessons) with provenance links back to the source entities.

### 7. The Cognitive Layer (`models.py`, `klines.py` & `service.py`)
*   **Role:** The paper-aligned memory semantics on top of storage and retrieval.
*   **Structured facts:** `SemanticFact` records (subject / predicate / value / aspect / confidence) live canonically in YAML frontmatter; a new fact with the same concept + predicate soft-deprecates the old value with a `replaced_by` link, keeping corrections auditable.
*   **K-line templates:** Procedural notes can carry trigger conditions (intents, styles, entity types, keyword phrases) plus a plan, slots, and failure modes. `match_template` scores them with weighted partial-matching (intent 0.5 as a hard gate, style 0.25, entity type 0.25, capped keyword bonus) and prefers templates with a higher adaptive utility (usage count + effectiveness EMA). `prepare_session` surfaces the winning template's plan directly into session guidance.

## Data Flow

### Single process (default)

```text
+-----------+     MCP (stdio / SSE) or CLI     +---------------------------+
|           | -------------------------------> |      marvin process       | ---> (Writes) ---> [ Git-Backed Vault ]
|   Agent   |                                  |  gateway + index + bus    | <--- (Reads)  <--- [ SQLite-Vec Index ]
|           | <------------------------------- |                           |
+-----------+        (Search / Tools)          |  in-process sleep pass    | ---> (Extract + Consolidate)
                                               +---------------------------+          |
                                                                                      v
                                                                       [ any litellm endpoint, e.g. Ollama ]
```

The vault's `extracted` / `consolidated` flags are the queue; `marvin consolidate` (or the `memory.sleep` event) drains it.

### Cluster profile (`MARVIN_BUS=nats`)

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
