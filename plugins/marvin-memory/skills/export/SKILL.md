---
name: export
description: This skill should be used when the user runs /marvin-memory:export or asks to "export memory as a skill", "compile the vault", "make my memories portable", "create a skill from what you know". Compiles the Marvin vault into a portable Agent Skills bundle.
argument-hint: "[--tag project/<owner>-<repo>] [--out <dir>] [--name <skill-name>]"
allowed-tools: Bash(marvin *)
---

# Export memory as a portable skill

Compile the vault into a self-contained skill bundle usable in any Agent-Skills-reading host — no Marvin server required where it lands.

1. Run the deterministic compiler (zero LLM calls), forwarding user flags:
   ```bash
   marvin skill export $ARGUMENTS
   ```
   `--tag project/<owner>-<repo>` narrows the bundle to one project's memories and derives the bundle name; bare export compiles the whole vault as `vault-memory`.
2. Report from the TOON output: the bundle path, the files produced (glossary / patterns / cheatsheet / history) with sizes, and anything truncated at the character budget.
3. Relay the install one-liners from the output's `help[]` block (copy into `./.claude/skills/` for Claude Code and Grok, `~/.agents/skills/` for Amp).
4. Remind the user the bundle is a snapshot: regenerate after significant memory changes, and prefer the live system where Marvin runs.
