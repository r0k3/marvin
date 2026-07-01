# Marvin Skill: Agentic Memory Usage

<skill_description>
Provides instructions for the agent on how to proactively use the Marvin MCP memory system to store and retrieve long-term knowledge, rules, and episodic logs across sessions. It instructs the agent on how to manage state, leverage git worktrees, and utilize "computational sleep" for optimal memory curation.
</skill_description>

## 1. Core Philosophy: Do not be Passive

You have a durable, long-term memory system called **Marvin**. You must use your memory tools **proactively and autonomously**. Do not wait for the user to explicitly say "remember this" or "save this." Treat memory management as a critical background thread of your operation.

*   **Search before guessing:** If you are unsure of a project convention, search memory first using `marvin_search`.
*   **Log major actions:** If you complete a complex task, log it.
*   **Isolate risky actions:** If you are about to rewrite multiple files, branch your memory first.

### Translating User Intent
The user does not understand the nuances of your memory taxonomy (Semantic vs Procedural vs Episodic). They will simply say "remember this" or "what do you know about X?".
*   **When asked to "remember":** You must analyze the information. Is it a permanent fact? Use `marvin_remember_semantic`. Is it a rule you must follow? Use `marvin_store_procedure`. If it's a conversation or action that just happened, use `marvin_log_episode`.
*   **When asked "what do you know":** Use `marvin_search` which searches across all memory types via hybrid Reciprocal Rank Fusion.

### Override Basic Harness Memory
If your specific agent platform (e.g., Goose, Cursor, OpenCode) has its own basic memory commands (like "Top of Mind", generic text logs, or `chatrecall`), **ignore them for permanent knowledge**. You must always prefer Marvin's tools for project-level facts, rules, and history to ensure they are graph-linked and durable across entirely different sessions and platforms.

## 2. The Operational Workflow

Follow this strict lifecycle during your sessions:

### A. Session Initialization
Whenever you begin a new conversation or start working on a major new task:
1.  **Context Loading:** Call `marvin_prepare_session(task="<your current task>")`. This will instantly retrieve relevant architectural facts, procedural rules, and recent episodes, keeping you grounded.

### B. Risky or Complex Tasks (Agentic Worktrees)
Before beginning a multi-step refactor or an exploratory feature build:
1.  **Branch Memory:** Call `marvin_start_worktree(branch_name="feature/<name>")`. This prevents your speculative memories and errors from polluting the primary knowledge base if the task fails.
2.  **Upon Success:** Only when the user is satisfied and the task is complete, call `marvin_merge_worktree(branch_name="feature/<name>")`.

### C. Active Learning & Logging
During the session, autonomously invoke these tools when their trigger conditions are met:

*   **Trigger: User states a preference or architectural fact.** (e.g., "We only use Tailwind", "My API key is X")
    *   **Action:** Call `marvin_remember_semantic(concept="...", predicate="...", value="...", aspect="preference|decision|knowledge", links=[...])`. Use `content="..."` only when the fact does not have an obvious predicate.
*   **Trigger: User provides a strict instruction or workflow.** (e.g., "Always run pytest before committing", "Never use classes in React")
    *   **Action:** Call `marvin_store_procedure(title="...", steps=[...])`.
*   **Trigger: You develop a reusable response strategy for a recurring kind of request.** (e.g., a debugging playbook, a code-review checklist)
    *   **Action:** Call `marvin_register_template(title="...", plan=[...], intents=[...], trigger_phrases=[...])`. After applying a template, record whether it helped via `marvin_record_template_use(title="...", success=true)` so effective strategies rank higher next time.
*   **Trigger: You successfully resolve a bug or complete a feature.**
    *   **Action:** Call `marvin_log_episode(title="...", summary="...", details="...")`. Describe what the problem was and exactly how you fixed it.

### D. Session Finalization & Computational Sleep
When a session is coming to an end, or you have completed a massive unit of work:
1.  **Extract & Consolidate:** Call `marvin_finalize_session(...)` to log your final episode while simultaneously extracting the most important semantics, procedures, and reflections you learned.
2.  **Trigger Sleep:** If your session was long and generated many raw, noisy episodic logs, call `marvin_trigger_sleep()`. This activates Marvin's background LLM to automatically deduce permanent rules from your recent chaos, optimizing your memory for tomorrow.
