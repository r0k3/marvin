---
name: sleep
description: This skill should be used when the user runs /marvin-memory:sleep or says "consolidate memory", "run the sleep pass", "distill episodes", or the session-start context shows several unconsolidated episodes. Runs Marvin's computational sleep — entity extraction plus two-phase consolidation.
argument-hint: "[--model <litellm id>] [--api-base <url>]"
allowed-tools: Bash(marvin *)
---

# Run the sleep pass

Distill raw episodic memory into durable facts and insights.

1. Run the pass, forwarding any user-supplied flags:
   ```bash
   marvin consolidate $ARGUMENTS
   ```
   This performs entity extraction (wikilink injection) over unprocessed notes, then episodic → semantic fact extraction, then semantic → reflective synthesis. It uses the configured local LLM (`MARVIN_SLEEP_MODEL`, default Ollama) and can take a minute on a backlog.
2. Report the outcome from the TOON output: `notes_linked`, `facts_extracted`, `insights_created`, and list the newly written notes so the user can review them.
3. On failure:
   - A message mentioning `marvin-memory[consolidate]` means the LLM extra is not installed — relay the exact install command from the error.
   - Connection errors mean the sleep endpoint is down — run `marvin doctor` and relay its fix commands (e.g. `ollama serve`).
4. If `extraction_skipped` is true, mention that entity extraction needs the `consolidate` extra and only consolidation ran.
