# Marvin

<p align="center">
  <img src="assets/logo.svg" width="160" alt="Marvin logo">
</p>

Long-term memory for AI agents, kept in plain Markdown.

Marvin stores what your agent learns as Markdown files with YAML frontmatter, in a vault you can open in [Obsidian](https://obsidian.md/), grep, diff, and keep under Git. Recall runs on a local hybrid index (full-text search, vectors, and a wikilink entity graph). Storing a memory never calls an LLM, and neither does recalling one.

The name honors Marvin Minsky. The design follows the paper [K-Lines: A Cognitively-Grounded Four-Memory Architecture for Persistent Conversational AI](https://ssrn.com/abstract=6234218), and this project is its reference implementation.

## What you get

- Four memory kinds as vault folders: episodic events, semantic facts, procedural playbooks, reflective insights.
- Structured facts with predicate, value, and confidence. Updates deprecate old values instead of overwriting them, so corrections keep an audit trail.
- Hybrid retrieval: FTS5, `sqlite-vec` vectors, and an entity-graph stream fused with reciprocal rank fusion, with optional cross-encoder reranking. 99.6% recall_any@5 on LongMemEval-S.
- An offline sleep pass that distills episodes into facts and insights with a local LLM, tracked by frontmatter flags so it can stop and resume anywhere.
- K-line templates: response strategies selected by trigger matching and ranked by measured effectiveness.
- Every interface an agent might want: a 20-tool MCP server, a TOON-output CLI built for shell-driving agents, auto-recall hooks, a Claude Code plugin, and portable skills.

## Where to start

- [Getting started](guide/getting-started.md) covers install, the Claude Code plugin, hooks, and agent configuration.
- [CLI reference](reference/cli.md) and [MCP tools](reference/mcp-tools.md) document the two main surfaces.
- [The agent skill](guide/skills.md) explains how agents learn when to use memory.
- [Evaluation](guide/evaluation.md) has the benchmark methodology and reproduction commands.
- [Architecture](architecture.md) explains how it works inside.

Source and issues: [github.com/r0k3/marvin](https://github.com/r0k3/marvin).
