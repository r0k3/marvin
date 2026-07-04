# Welcome to Marvin

<p align="center">
  <img src="assets/logo.svg" width="200" alt="Marvin Logo">
</p>

**Marvin** is an active, Obsidian-native, Git-backed memory cluster designed for professional AI agents.

Named in homage to **Marvin Minsky** and his foundational book [*The Society of Mind*](https://amzn.to/3PuFQ3K), this project transforms the concept of ephemeral LLM context windows into a durable, topologically connected knowledge base. It is a practical implementation of the cognitive architectures detailed in the research paper [*K-Lines: A Cognitively-Grounded Four-Memory Architecture for Persistent Conversational AI*](https://ssrn.com/abstract=6234218) (and its [companion repository](https://github.com/r0k3/k-lines)).

---

## Why Marvin?

Most agentic "memory skills" today simply dump chat logs into a hidden SQLite database or a black-box vector store. This is sufficient for casual chatbots, but it breaks down rapidly in professional software engineering workflows. When you take your agent's memories seriously, you need **ergonomics, interpretability, and safety**.

We built Marvin V2 around three core philosophies:

### 1. The Ergonomics of Obsidian (Markdown)
Developers already maintain knowledge bases. Instead of locking agent memories inside a proprietary database, Marvin writes everything as clean, human-readable **Markdown files with YAML frontmatter**. 

By pointing your [Obsidian](https://obsidian.md/) vault to Marvin's storage directory, you instantly get a beautiful, visual graph of everything your agent has learned. You can manually edit the files, add your own notes, and seamlessly co-author the knowledge base alongside your AI.

### 2. Safety through Git (Agentic Worktrees)
Agents hallucinate and explore dead ends. If an agent tries a risky 4-hour refactoring task and completely fails, its memories of that failure shouldn't pollute your ground-truth knowledge base.

Because Marvin's vault is natively backed by **Git**, agents can check out isolated "Worktrees" (branches). If a task succeeds, the memory branch is merged. If it fails, the branch is discarded. You get full `git blame` for your agent's thoughts.

### 3. Asynchronous Consolidation (Computational Sleep)
Biological memory isn't just stored; it is *consolidated* while we sleep. 

Marvin utilizes a Dockerized architecture with a **NATS** message broker and a background **Brain Worker**. We chose NATS because it is exceptionally lightweight, simple to deploy, and highly extensible. While your agent rapidly logs noisy, raw "Episodic" events, the Brain Worker asynchronously uses local NLP (`langextract`) to map entities, and a local LLM (default `qwen3.6:35b-a3b-q4_K_M` via Ollama) to consolidate memory in two phases: entity-scoped **Semantic** facts extracted from episodes, then cross-fact **Reflective** insights synthesized per aspect — overnight, without blocking your workflow.

---

## Features

- **Obsidian-Native Vault:** Memories are categorized into `Semantic`, `Procedural`, `Episodic`, and `Reflective` markdown folders.
- **Structured Semantic Facts:** Facts carry a predicate, value, aspect, and confidence; updating a fact soft-deprecates the old value (auditable, never silently overwritten, excluded from retrieval).
- **K-Line Procedural Templates:** Response strategies with trigger conditions (intents, styles, entity types, keywords), selected by weighted partial-match scoring and ranked by an adaptive effectiveness score.
- **Deep Semantic Graphing:** Zero-shot entity extraction automatically injects `[[Wikilinks]]` into text, connecting concepts without the agent having to do it manually.
- **Computational Sleep:** Asynchronous two-phase consolidation (episodic → semantic facts, semantic → reflective insights) using local open-weight models.
- **Hybrid Retrieval:** Embedded vector search (`sqlite-vec`) + full-text keyword search (FTS5) + an entity-graph stream, fused with Reciprocal Rank Fusion; optional cross-encoder reranking. 99.6% `recall_any@5` on LongMemEval-S.
- **MCP Native:** 20 tools — the full service surface — over the standard Model Context Protocol (SSE or stdio).
- **AXI CLI:** the same functionality as an [axi.md](https://axi.md/)-style command line with token-efficient TOON output, a live dashboard, and `help[]` next-step hints — built for agents driving a shell.

---

## Who is this for?

Marvin is not a simple "plug and play" toy script. It is an orchestrated cluster designed for **power users, AI researchers, and professional developers** who want to build a deeply integrated, highly maintainable, and completely local "second brain" for their autonomous coding agents.

If you are ready to give your agent a serious memory upgrade, head over to the [Getting Started](guide/getting-started.md) guide.

---

**Quick links:** [GitHub Repository](https://github.com/r0k3/marvin) | [Quick Start & Agent Config](https://github.com/r0k3/marvin#quick-start) | [MCP Tools (20)](reference/mcp-tools.md) | [CLI Reference (AXI)](reference/cli.md) | [Agent Skill](guide/skills.md) | [Research Paper (SSRN)](https://ssrn.com/abstract=6234218)
