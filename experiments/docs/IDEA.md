# Marvin V2: A State-of-the-Art Agentic Memory System

## The Vision
Marvin is designed to bridge the gap between an LLM agent's ephemeral context window and the requirement for long-term, durable reasoning in professional software engineering. While version 1 provided a fast, local SQLite-backed retrieval index for Obsidian markdown files, **Version 2** transforms Marvin from a passive database into an active, thinking memory cluster.

We are building a Dockerized, multi-container architecture that handles memory storage, semantic entity extraction, and continuous "computational sleep" (consolidation), all without blocking the primary agent (e.g., Goose).

## The "Why" (Core Philosophies)

### 1. The Necessity of Computational Sleep
In biological systems, memory is not just stored; it is processed, pruned, and consolidated during sleep. In agentic systems, dumping every raw terminal error or chat response into an index eventually poisons the search space with noise.
*   **The Solution:** An asynchronous worker that reads noisy *Episodic* memories, uses a dedicated LLM to extract universal truths (*Semantic*) and behavioral rules (*Procedural*), and deprecates outdated facts.

### 2. Deep Semantic Graphing
Retrieval-Augmented Generation (RAG) using purely dense vectors often fails to capture rigid structural relationships (e.g., "Service A depends on Library B").
*   **The Solution:** We utilize Google's `langextract` library in the background to perform zero-shot entity and relation extraction on all new memories. It automatically injects `[[Obsidian Wikilinks]]` into the markdown files, building a mathematically rigorous knowledge graph that the agent can traverse.

### 3. Memory as Code (Git-Backed Worktrees)
Agents often explore dead ends. If an agent tries a major refactor and fails, its memory of that failure should not permanently pollute the primary knowledge base as established "fact."
*   **The Solution:** The `marvin_vault` is a Git repository. We expose tools for the agent to spawn "memory worktrees" (Git branches) tied to specific tasks. If the task succeeds, the memory branch is merged. If it fails, the branch is discarded, keeping the `main` memory clean.

## The "What" (Architecture)

Marvin V2 operates as a cluster of lightweight services orchestrated by Docker Compose:

1.  **MCP Gateway (`mcp-gateway`):** A Python FastMCP server exposing tools via SSE (Server-Sent Events) on port 8421, with a fallback to standard `stdio`. This is the low-latency edge that Goose talks to. It reads from the `sqlite-vec` index and publishes events to the broker.
2.  **Message Broker (`nats`):** We use NATS, a hyper-fast, lightweight binary broker, to handle async events like `memory.created` and `memory.consolidate` without dropping messages.
3.  **Brain Worker (`brain-worker`):** A Python subscriber daemon. It listens to NATS, runs the heavy `langextract` models to update the markdown graph, and handles the Git merging/conflict resolution logic.
4.  **Local LLM (`ollama`):** A bundled container providing a truly local, cost-free LLM (e.g., Llama 3) for the Brain Worker to use during the consolidation phase. (Configurable to use external APIs via `litellm` if preferred).

## The "How" (Workflow Example)

1.  **Initialization:** Goose starts a session. The `marvin_prepare_session` tool fetches the current procedural rules.
2.  **Worktree Creation:** Goose starts a major task and calls `marvin_start_worktree("auth-refactor")`. Marvin creates a Git branch.
3.  **Episodic Logging:** Goose completes a sub-task and logs it. The MCP gateway writes the markdown to the branch and publishes a `memory.created` event to NATS.
4.  **Background Processing:** The Brain Worker picks up the event, runs `langextract`, discovers the entity "JWT Token", and rewrites the markdown file to link to `[[JWT Token]]`.
5.  **Consolidation:** At night (or triggered via `marvin_sleep()`), the Brain Worker reads all episodes, uses Ollama to deduce that "JWT Tokens require a 15-minute expiry in this codebase," and saves a new *Procedural* rule.
