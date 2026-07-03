---
name: marvin-memory
description: Use when Marvin memory is available (marvin_* MCP tools or the `marvin` CLI) and any of these occur — the user states a durable preference, fact, rule, or correction; asks what you know or how something is done here; confirms an approach worked or failed; you finish significant work; or a session starts or ends.
---

# Marvin Memory

Marvin is your durable four-type memory for this user — episodic, semantic,
procedural, reflective — stored as human-readable Markdown they can audit.
Memory work is a background thread of your operation: act on the signals
below without waiting to be told "remember this".

Tools are named `marvin_*` over MCP; the shell equivalent is the `marvin`
CLI (shown second). Parameter details: tool descriptions / `marvin --help`.

## Store on these signals

| Signal | Store as | MCP / CLI |
|---|---|---|
| Durable fact, preference, or decision ("we use X", "never do Y") | semantic fact | `marvin_remember_semantic` / `marvin remember <concept> --predicate <p> --value <v> --aspect <a> --confidence <c>` |
| Workflow rule ("always run X before Y") | procedure | `marvin_store_procedure` / `marvin procedure <title> --step <s> ...` |
| Correction of an earlier fact ("actually it's Z now") | the SAME concept + predicate again with the new value — the old one soft-deprecates automatically; never edit or delete | same as fact |
| Notable work completed (bug fixed, feature shipped, incident resolved) | episode | `marvin_log_episode` / `marvin episode <title> --summary <s>` |
| A transferable lesson inside that work (root cause, gotcha, principle) | reflection, in addition to the episode | `marvin_reflect` / `marvin reflect <title> --insight <i>` |
| A response strategy that proved reusable | K-line template | `marvin_register_template` / `marvin template register <title> --plan <step> --intent <i> --trigger <phrase>` |

Calibrate `aspect` (knowledge / preference / decision / directive / goal /
problem / belief) and `confidence`: categorical statements ("no exceptions",
"never") warrant 0.9+; hedged ones ("I think", "for now") 0.5–0.6.

Do not store: scheduling ephemera, secrets or credentials, or anything the
code and git history already record.

## Recall before you answer

For any question about this project's or user's past, conventions, or
"how do we ... here": search memory first — `marvin_search` /
`marvin search <query>` — read the top hit, and answer from it. If memory
has nothing, say you don't have it stored; don't invent.

When a request matches a recurring situation ("walk me through", "how
should I handle"), select the strategy through the matcher —
`marvin_match_template` / `marvin template match <context> --intent <i>` —
and follow the returned plan. Selecting through the matcher (rather than
reading the template note directly) is what lets effective strategies
outrank stale ones. Intents are short labels (`debug`, `biography`,
`recall`), not sentences; on zero matches, retry with a simpler intent or
none before falling back to reading procedures directly.

## Close the loop — the most-missed step

When a later user message reveals the outcome of a strategy you applied —
"that worked", "found it", "great process", or "no, that was wrong" —
record it immediately:

`marvin_record_template_use(title, success)` /
`marvin template used <title> [--failure]`

An unrecorded outcome freezes that template's effectiveness at zero, and
future selection can never learn from it. Success confirmations count, not
just failures.

## Session lifecycle

- **Start of a session or major task:** `marvin_prepare_session(task=...)` /
  `marvin session prepare <task>` — grounds you in relevant facts, rules,
  and recent episodes (a matching template's plan arrives in guidance).
- **End of a session or milestone:** `marvin_finalize_session` /
  `marvin session finalize <title> --summary <s> [--fact <f>] [--reflection <r>]`.
- **≥3 unconsolidated episodes** (the CLI dashboard shows the count):
  trigger consolidation — `marvin_trigger_sleep` (background cluster) or
  `marvin_consolidate` / `marvin consolidate` (synchronous).
- **Risky or exploratory multi-step work:** branch memory first —
  `marvin_start_worktree` / `marvin worktree start <branch>` — merge on
  success, discard on failure.
