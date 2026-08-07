---
name: remember
description: This skill should be used when the user runs /marvin-memory:remember or says "remember this", "store this fact", "save to memory", "note that we use X", or corrects a stored fact ("actually it's Y now — update memory"). Stores durable knowledge into the Marvin vault via the marvin CLI.
argument-hint: <fact, preference, decision, or correction in free text>
allowed-tools: Bash(marvin *)
---

# Store a memory

Turn the user's free-text statement into a structured semantic fact and store it.

Input: `$ARGUMENTS` (fall back to the most recent user statement when empty).

1. Extract the structure:
   - **concept** — the subject the fact is about (e.g. `DB`, `Deploy target`, `Code style`)
   - **predicate** — a short stable property name (`storage`, `hostname`, `preference`, `uses`)
   - **value** — the fact text itself
   - **aspect** — one of knowledge / preference / decision / directive / goal / problem / belief
   - **confidence** — 0.9+ for categorical statements ("always", "never"), 0.5–0.6 for hedged ones ("for now", "I think")
2. Determine whether this **corrects** an existing fact (wording like "actually", "no longer", "changed to", or a same-concept lookup via `marvin read <concept>` when unsure). For corrections, add `--reason "user correction: <what changed>"` — the old value soft-deprecates with an audit trail; never edit or delete it.
3. Store it:
   ```bash
   marvin remember "<concept>" --predicate <predicate> --value "<value>" --aspect <aspect> --confidence <c> [--reason "..."]
   ```
4. Confirm to the user what was stored (concept, predicate, value) and where (the path from the TOON output). If the input was a procedure ("always run X before Y") rather than a fact, use `marvin procedure` instead; if it was a lesson, use `marvin reflect`.

Do not store scheduling ephemera, secrets or credentials, or anything the code and git history already record — say so instead.
