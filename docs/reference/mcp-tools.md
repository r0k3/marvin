# MCP Tools Reference

Marvin exposes 15 tools to Model Context Protocol (MCP) clients.

## Session Lifecycle

### `marvin_prepare_session`
Hook-friendly session bootstrap: pulls relevant procedures, semantic facts,
reflections, and recent episodes for the task at hand. If a K-line template
matches the task, its plan is surfaced at the top of the guidance.
**Arguments:**
* `task` (str): What the agent is about to work on.
* `repo_name` (str, optional), `technologies` (list[str], optional): Extra retrieval context.
* `limit` (int, optional): Defaults to 8.

### `marvin_finalize_session`
Hook-friendly session finalizer: logs a closing episode and optionally extracts
semantic facts, procedures, and reflections in the same call.
**Arguments:**
* `title` (str), `summary` (str), `details` (str, optional)
* `semantic_facts` (list[str], optional), `procedures` (list[dict], optional), `reflections` (list[str], optional)
* `tags`, `links` (list[str], optional)

### `marvin_trigger_sleep`
Trigger background consolidation ("computational sleep") via the Brain Worker:
phase 1 extracts entity-scoped semantic facts from unconsolidated episodes,
phase 2 synthesizes reflective insights across accumulated facts.

## Core Storage Tools

### `marvin_remember_semantic`
Stores a durable piece of architectural knowledge, user preference, or fact.
**Arguments:**
* `concept` (str): The name of the concept.
* `content` (str, optional): Easy path for an unstructured fact.
* `predicate` (str, optional): Stable property name for structured facts.
* `value` (str, optional): Structured fact value. If omitted, `content` is used.
* `aspect` (str, optional): One of `knowledge`, `preference`, `decision`, `goal`, `problem`, `belief`, or `directive` (default: `knowledge`).
* `confidence` (float, optional): 0.0–1.0 confidence score (default: 0.6).
* `tags` (list[str], optional): e.g. `["python", "best-practice"]`
* `links` (list[str], optional): Links to related concepts.

Semantic facts are stored with stable IDs in note frontmatter and rendered
under `## Facts` for Obsidian/search readability. A new active fact with the
same `concept` + `predicate` and a different value soft-deprecates the older
fact; deprecated facts remain auditable under `## Deprecated Facts` but are not
indexed for retrieval.

### `marvin_store_procedure`
Stores a reusable playbook or strict rule the agent should follow.
**Arguments:**
* `title` (str): Rule name.
* `steps` (list[str]): The ordered procedure.
* `applicability` (list[str], optional): When this applies.
* `anti_patterns` (list[str], optional): Things to strictly avoid.

### `marvin_register_template`
Registers a **K-line procedural template**: a response strategy with explicit
trigger conditions and an ordered plan, selectable by `marvin_prepare_session`
via weighted partial-match scoring (intent 0.5 as a hard gate, style 0.25,
entity type 0.25, plus a capped keyword bonus).
**Arguments:**
* `title` (str), `plan` (list[str]): The strategy and its ordered steps.
* `intents`, `styles`, `entity_types`, `trigger_phrases` (list[str], optional): Trigger conditions.
* `slots`, `failure_modes` (list[str], optional): Fill-in parameters and known pitfalls.
* `tags` (list[str], optional)

### `marvin_record_template_use`
Records whether a K-line template helped, updating its usage count and
effectiveness EMA (ACT-R-style utility) so effective templates rank higher
in future matches.
**Arguments:**
* `title` (str), `success` (bool)

### `marvin_log_episode`
Log a completed task, fixed bug, or event.
**Arguments:**
* `title` (str): Short summary.
* `summary` (str): Longer detail.
* `details` (str, optional): In-depth context.

### `marvin_reflect`
Store a lesson learned.
**Arguments:**
* `title` (str)
* `insight` (str)

## Advanced Retrieval

### `marvin_search`
Performs a hybrid search (Reciprocal Rank Fusion on Vector + Keyword FTS5).
**Arguments:**
* `query` (str): The query string.
* `kind` (str, optional): `semantic`, `procedural`, `episodic`, or `reflective`.
* `limit` (int, optional): Defaults to 6.

### `marvin_recent_activity`
Fetches chronologically recent memories. Useful for getting caught up after an agent restart.

### `marvin_read_memory`
Reads a single note by title, alias, or vault-relative path. The payload
includes the note's structured facts alongside tags, links, and the body.
**Arguments:**
* `identifier` (str)

### `marvin_sync`
Syncs the Obsidian vault into Marvin's local hybrid index (picks up notes
added or edited outside the MCP tools, e.g. by hand in Obsidian).

## Advanced Git Worktrees

### `marvin_start_worktree`
**Arguments:** `branch_name` (str)
Spawns a Git branch. Keeps the memory index clean during risky agent explorations.

### `marvin_merge_worktree`
**Arguments:** `branch_name` (str)
Merges the worktree back to `main`.
