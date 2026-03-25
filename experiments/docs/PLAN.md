# Implementation Plan: Marvin V2

## Phase 1: Environment & Orchestration Setup
1.  **Docker Compose Setup:**
    *   Create `docker-compose.yml` at the root of `marvin/`.
    *   Define services: `mcp-gateway` (builds from local Dockerfile), `brain-worker` (builds from same Dockerfile, different command), `nats` (official alpine image), and `ollama` (official image, with volume for model caching).
2.  **Dependencies:**
    *   Update `pyproject.toml` to include: `nats-py`, `litellm`, `gitpython`. *(Note: `google-langextract` might need to be installed from source/git if not readily available on PyPI, or we use a comparable entity extractor like `gliner` if `langextract` is unavailable, but we will target `langextract` first based on user preference).*
3.  **Dockerfile:**
    *   Create a multi-stage `Dockerfile` using `python:3.12-slim` and `uv` to install dependencies efficiently.

## Phase 2: Git-Backed Vault Management
1.  **GitManager Class (`src/marvin/git.py`):**
    *   Implement a wrapper around `gitpython` or standard `subprocess` git commands.
    *   Functions: `init_vault()`, `commit(message)`, `checkout_branch(name, create=True)`, `merge_branch(source, target)`.
2.  **Vault Integration (`src/marvin/vault.py`):**
    *   Update the `write_note` method to automatically trigger a `git add` and `git commit` to the currently active branch.

## Phase 3: The NATS Message Broker
1.  **NATS Client (`src/marvin/broker.py`):**
    *   Initialize async connection to `nats://nats:4222`.
    *   Create simple `publish_event(subject, payload)` and `subscribe(subject, handler)` methods.
2.  **Gateway Publisher:**
    *   Update `src/marvin/server.py` and `service.py`. Whenever a memory is written, publish a `memory.created` event containing the relative path and kind.

## Phase 4: The Brain Worker (Entity Extraction)
1.  **Worker Daemon (`src/marvin/worker.py`):**
    *   Create the main loop that connects to NATS and listens for events.
2.  **LangExtract Integration (`src/marvin/extraction.py`):**
    *   Implement the logic to receive raw markdown text, run it through the entity extractor.
    *   If new entities/relations are found, rewrite the original markdown file to inject `[[Entity]]` links.
    *   Commit the change via Git (`"chore: auto-linked entities"`).

## Phase 5: Computational Sleep (Consolidation)
1.  **LLM Integration (`src/marvin/llm.py`):**
    *   Configure `litellm` to point to the local `ollama` container by default (e.g., `ollama/qwen3.5:9b`).
2.  **Consolidation Logic (`src/marvin/consolidation.py`):**
    *   Fetch all `Episodic` notes missing a `consolidated: true` flag.
    *   Prompt the LLM: "Extract permanent facts (Semantic) and coding rules (Procedural) from these logs."
    *   Parse the LLM JSON output. Create new Semantic/Procedural markdown files. Update the old Episodic files with `consolidated: true`. Commit changes.
3.  **MCP Trigger:**
    *   Add `marvin_trigger_sleep()` to the MCP server, which simply publishes a `memory.consolidate` event to NATS.

## Phase 6: New MCP Tool Surface
1.  **Server Updates (`src/marvin/server.py`):**
    *   Add `marvin_start_worktree(branch_name)`
    *   Add `marvin_merge_worktree(branch_name)`
    *   Add `marvin_get_related(concept)` (Parses the markdown links and returns connected notes).
    *   Add `marvin_deprecate_memory(identifier, reason)` (Edits frontmatter to `deprecated: true`).

## Phase 7: Testing & Documentation
1.  **Skill Update:** Rewrite `src/marvin/skill.md` to instruct the agent on how to use worktrees for risky tasks and how to traverse the graph.
2.  **End-to-End Test:** Write a script to simulate Goose creating a worktree, logging an episode, triggering sleep, and merging.
