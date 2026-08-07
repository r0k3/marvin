---
name: status
description: This skill should be used when the user runs /marvin-memory:status or asks "how's my memory vault", "memory status", "is marvin healthy", "check the memory setup". Shows the vault dashboard and the install-state checkup.
allowed-tools: Bash(marvin *)
---

# Memory status

Give the user a one-glance picture of their memory system.

1. Run both, in order:
   ```bash
   marvin            # live vault dashboard (counts per kind, recent, suggestions)
   marvin doctor     # install-state checkup with exact fix commands
   ```
2. Summarize in a few sentences: vault size and shape (episodic/semantic/procedural/reflective counts), how many episodes await consolidation, and whether every component is healthy.
3. Surface anything actionable verbatim from the outputs: the `marvin consolidate` suggestion when unconsolidated episodes have accumulated, and any `help[]` fix commands from doctor (missing extras, unreachable sleep endpoint, uninstalled skill).
4. When everything is healthy and the vault is active, say so briefly — no fabricated concerns.
