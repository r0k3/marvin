# Marvin architecture

Marvin bridges an LLM agent's ephemeral context window and the need for durable, auditable knowledge. It is an Obsidian-native, Git-backed memory system: Markdown files are the single source of truth, everything else is derived.

## Design principles

1. **Interpretability and portability.** Every memory is a plain Markdown file with YAML frontmatter, laid out as an Obsidian vault. No proprietary blobs. The SQLite index is a cache and can always be rebuilt from the vault (`marvin rebuild`).
2. **Zero LLM calls on the write and query paths.** Storing and recalling memories is deterministic and local. LLMs appear in exactly one place: the offline sleep pass.
3. **The vault is the work queue.** Pending work is marked durably in frontmatter (`consolidated` on episodes, `extracted` on all notes). A crash, restart, or long pause loses nothing; the next sleep pass reads the flags and continues. No event log can drift from the source of truth, because the source of truth is the queue.
4. **Safety through Git.** Every memory write can be a commit. Risky work branches the vault (`marvin worktree start`), merging on success and discarding on failure. `git blame`, `diff`, and `revert` work on the agent's knowledge.

## Deployment profiles

One binary, two profiles:

**Single process (default).** `marvin` (CLI) and `marvin serve` (MCP) run everything in one process: vault, hybrid index, and, when the `consolidate` extra is installed, the sleep pass, either on demand (`marvin consolidate`, the `marvin_consolidate` tool) or in the background of the serve process (`marvin_trigger_sleep` schedules it on the in-process event bus). The base install has no broker, no LLM client, and no server framework.

**Cluster (`[cluster]` extra + Docker Compose).** For always-on background processing across processes: the gateway publishes events to NATS (`MARVIN_BUS=nats`) and a separate Brain Worker consumes them, with a bundled Ollama container for local inference.

Scaling follows Git, not a message bus: replicate the vault via a Git remote and each node rebuilds its index locally. The only central piece a team needs is the remote.

### Dependency extras

| Install | Adds | Enables |
|---|---|---|
| base | fastembed, sqlite-vec, mcp, gitpython, pydantic | vault, index, retrieval, CLI, MCP server, hooks |
| `[consolidate]` | litellm, langextract | the sleep pass (extraction + consolidation) |
| `[cluster]` | nats-py | the NATS-evented multi-process profile |
| `[gpu]` | CUDA onnxruntime | GPU embeddings and the fp16 reranker |

## Components

### Storage: vault and models (`vault.py`, `models.py`)

Four memory kinds as first-class folders: `Episodic/`, `Semantic/`, `Procedural/`, `Reflective/`. Semantic notes carry structured facts in frontmatter (stable id, predicate, value, aspect, confidence, source, deprecation metadata); active facts render into the body under `## Facts`. A new value for an existing concept + predicate soft-deprecates the old fact with `replaced_by` and a reason, rather than overwriting it.

### Retrieval: the hybrid index (`index.py`, `embeddings.py`, `reranker.py`)

SQLite holds three streams over chunked Markdown: FTS5 keyword search, dense vectors in `sqlite-vec` (bge-small via fastembed, ONNX, local), and an entity graph hydrated from `[[wikilinks]]`. Chunk-level reciprocal rank fusion combines FTS and vectors; a second note-level RRF folds in the IDF-weighted graph ranking, letting a note surface because it links to the query's entities even when no chunk matches lexically. An optional cross-encoder (`bge-reranker-v2-m3`) rescores the pool, and an opt-in freshness boost favors recent episodic notes. A `lexical_only` mode skips the dense stream entirely so latency-bound callers never pay an embedding-model load. Deprecated facts are excluded at chunking time, so corrected knowledge cannot resurface in search.

### Service layer (`service.py`)

`MarvinService` composes vault, index, and Git into the operation surface shared by every interface: remember/search/read, session prepare and finalize, K-line template registration and matching, consolidation phases, consistency checks. The 20 MCP tools (`server.py`, FastMCP over stdio or SSE), the CLI (`cli.py`), and the hook commands are all thin layers over it.

### The cognitive layer (`klines.py`)

Procedural notes can be K-line templates: response strategies with trigger conditions (intents, styles, entity types, keyword phrases), a plan, slots, and failure modes. `match_template` scores them with weighted partial matching (intent is a hard gate) and breaks ties with an ACT-R-style utility from usage count and an effectiveness EMA, updated by `marvin template used`. `prepare_session` surfaces the winning plan directly into session guidance.

### The sleep pass (`service.sleep`, `extraction.py`, `consolidation.py`)

Three stages, LLM-powered, strictly offline:

1. **Entity extraction.** langextract performs zero-shot extraction over every note not yet marked `extracted`, rewriting it with `[[wikilinks]]` that feed the graph stream. Falls back to a regex heuristic when langextract is unavailable.
2. **Episodic to semantic.** Unconsolidated episodes are grouped by the entities they link. Once an entity appears in enough episodes (default 3), atomic facts are extracted, deduplicated against the entity's known facts, and persisted; consumed episodes are marked `consolidated`.
3. **Semantic to reflective.** Accumulated facts are grouped by aspect and synthesized into cross-fact insights with provenance links back to their source entities.

The engine (`consolidation.py`) speaks litellm, defaulting to local Ollama (`MARVIN_SLEEP_MODEL`, `MARVIN_SLEEP_API_BASE`). The same pass runs from the CLI, the MCP tool, the in-process bus handler, or the cluster worker.

### The event bus (`bus.py`, `worker.py`)

Events are a nudge, never the source of truth (principle 3). `InProcessBus` (default) dispatches handlers as fire-and-forget tasks inside the serve process. `NatsBus` publishes `memory.created` and `memory.sleep` over JetStream for the cluster profile, where `worker.py` consumes them.

### Auto-recall hooks (`cli.py` hook commands, `promotion.py`)

Host agents run `marvin hook session-start` and `marvin hook user-prompt`; stdout is injected into the model's context. Session start emits the highest-confidence facts (current project's first) plus recent episodes, without loading any model. The per-prompt hook recalls lexically (`lexical_only` search) against the prompt text within a character budget, and runs a deterministic correction detector: when the prompt matches a correction cue ("no, actually...", "we no longer use..."), the hook nudges the host agent, which holds the full conversation, to persist the fix with `--reason`, closing the correction loop with zero Marvin-side LLM calls. Hooks always exit 0; a broken vault must never break a session. `--format json` wraps the output in the `hookSpecificOutput` envelope for hosts that require it. `marvin hooks install --host claude|codex|grok` wires the host configs; `marvin hooks show` prints templates for hosts with plugin-only APIs.

### Project tagging (`project.py`)

Writes made inside a git repository are auto-tagged `project/<owner>-<repo>`, derived from the origin remote so clones share a tag. The vault is deliberately global; tags are how recall narrows.

### Write-path hygiene (`sanitize.py`)

Two policies, deliberately different. Every write is stripped of invisible Unicode (zero-width characters, bidi controls, the tag block): those code points have no legitimate role in memory notes and are the classic vector for instructions a human auditor cannot see. Content scanning applies only to machine-generated text: sleep-pass output matching injection shapes (instruction overrides, fake transcript prefixes, chat-template tokens) is dropped and logged before it can enter future agent context. Notes authored by the user or agent are never content-filtered; a note *about* prompt injection must remain storable.

### Skill packaging (`export.py`, `src/marvin/skill/`)

Two directions. The bundled `marvin-memory` skill teaches an agent when to store, recall, correct, and close the feedback loop; `marvin skill install --host claude|grok|amp` places it in the host's skills directory. In reverse, `marvin skill export` deterministically compiles the vault into a portable Agent Skills bundle (facts to a glossary, procedures and templates to patterns, insights to a cheatsheet, episodes to a history, all character-budgeted) usable where no Marvin runs.

### The Claude Code plugin (`plugins/marvin-memory/`)

The repository doubles as a plugin marketplace (`.claude-plugin/marketplace.json`). The plugin bundles the hooks, the stdio MCP server, the skill (a verbatim copy of the canonical one, equality-enforced by test), five slash commands, and a memory-curator agent. It stays a thin wrapper over the host-neutral CLI; nothing Claude-specific lives in the core package.

## Latency budget

The CLI is on agent hot paths, so import cost is a design constraint: litellm, langextract, and the MCP server stack all load lazily, keeping cold start near one second and the hook commands around 0.4 s. The per-prompt hook additionally avoids the embedding model via `lexical_only` retrieval. Anything that would put a module-level heavy import back on the CLI path is a regression.

## Data flow

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
[ Local Ollama ] <--- (Distill Facts/Rules) --|    Brain Worker    |
                                              |                    | ---> (Update Links & Commit to Vault)
                                              +--------------------+
```
