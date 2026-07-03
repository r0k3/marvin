# The Agent Skill

Marvin ships with a first-class agent skill, **`marvin-memory`**, in the
standard `SKILL.md` format (YAML frontmatter + instructions) that Claude Code
and other skill-aware harnesses load natively. It teaches an agent *when* to
use memory without being told — turning Marvin from a set of tools into a
habit.

## Install

```bash
marvin skill install          # project-level → ./.claude/skills/marvin-memory
marvin skill install --user   # user-level    → ~/.claude/skills/marvin-memory
marvin skill install --target <dir>   # any skills directory
marvin skill show             # print SKILL.md — paste into any other harness
```

The skill is bundled inside the package
([source](https://github.com/r0k3/marvin/blob/main/src/marvin/skill/SKILL.md)),
so `install` works offline from any Marvin installation. It covers **both
surfaces**: the `marvin_*` MCP tools and the AXI CLI — whichever your agent
has, the same guidance applies.

## What it teaches (and why)

The skill was built test-first: we ran pressure scenarios with capable agents
that had full access to Marvin but **no instructions**, recorded what they
got wrong, and wrote the skill against those observed failures.

What baseline agents already did well (the CLI's own affordances carry a lot):
choosing sensible memory types, skipping ephemera, even updating a corrected
fact through the same concept + predicate so it soft-deprecated cleanly.

What they reliably missed — and what the skill therefore drills:

1. **Closing the feedback loop.** After the user confirmed a strategy worked
   ("that worked great, we found it"), no baseline agent recorded the outcome
   — the K-line template's effectiveness stayed frozen at zero forever. The
   skill makes outcome-recording an explicit trigger:
   `marvin_record_template_use` / `marvin template used <title> [--failure]`.
2. **Selecting strategies through the matcher.** Baselines read procedural
   notes directly instead of calling `marvin_match_template` — which means
   effectiveness-based ranking never gets a chance to work.
3. **Confidence calibration.** A categorical "no exceptions" directive was
   stored at the default confidence (0.6). The skill sets the rule:
   categorical ⇒ 0.9+, hedged ⇒ 0.5–0.6.
4. **Reflections alongside episodes.** War stories were logged as episodes,
   but the transferable lesson inside them (the root cause, the gotcha) was
   never lifted into reflective memory.

Plus the foundations: a signal → memory-type table, recall-before-answering,
the session lifecycle (`prepare` / `finalize` / consolidate), memory
worktrees for risky work, and a short do-not-store list (ephemera, secrets,
anything the code or git history already records).

## Ambient context (pairs well)

The skill tells the agent *when* to act; the CLI dashboard tells it *what
state memory is in*. Wiring bare `marvin` as a session-start hook gives the
agent both (see the [CLI reference](../reference/cli.md) for the hook
snippet).
