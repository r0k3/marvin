# MCP Tools Reference

Marvin exposes a set of rich tools designed for Model Context Protocol (MCP) clients.

## Core Storage Tools

### `marvin_remember_semantic`
Stores a durable piece of architectural knowledge, user preference, or fact.
**Arguments:**
* `concept` (str): The name of the concept.
* `content` (str): The fact.
* `tags` (list[str], optional): e.g. `["python", "best-practice"]`
* `links` (list[str], optional): Links to related concepts.

### `marvin_store_procedure`
Stores a reusable playbook or strict rule the agent should follow.
**Arguments:**
* `title` (str): Rule name.
* `steps` (list[str]): The ordered procedure.
* `applicability` (list[str], optional): When this applies.
* `anti_patterns` (list[str], optional): Things to strictly avoid.

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

## Advanced Git Worktrees

### `marvin_start_worktree`
**Arguments:** `branch_name` (str)
Spawns a Git branch. Keeps the memory index clean during risky agent explorations.

### `marvin_merge_worktree`
**Arguments:** `branch_name` (str)
Merges the worktree back to `main`.
