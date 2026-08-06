# Changelog

All notable changes to **marvin-memory** are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/), generated from
[Conventional Commits](https://www.conventionalcommits.org/) via
[git-cliff](https://git-cliff.org/).

## [0.4.0] - 2026-08-06


### Added

- lean base install — marvin-memory on PyPI, consolidate/cluster extras
- single-process default — in-process event bus, vault-as-queue sleep pass
- marvin doctor — read-only install-state checkup with exact fixes
- auto-recall hooks, correction detection, project auto-tagging
- codex + grok installers, plugin templates for opencode/amp
- invisible-Unicode stripping on every write, injection scan on sleep output

### Fixed

- 0.4s hooks — lazy langextract and mcp-server imports

### Documentation

- single-process default story — profiles, extras, doctor, cluster as opt-in
- auto-recall guide, correction flow in the skill, reference rows

## [0.3.0] - 2026-07-04


### Added

- complete the tool surface (20 tools)
- AXI command-line interface (v0.3.0)
- ship the marvin-memory agent skill, test-first

### Documentation

- CLI reference (AXI), MCP additions, refreshed configs
- surface the v0.3 MCP + CLI story everywhere
- add SSRN paper citation to README, canonicalize paper links

## [0.2.0] - 2026-07-01


### Added

- add LongMemEval-S retrieval benchmark
- add optional cross-encoder reranker (bge-reranker-v2-m3)
- add --rerank option with per-chunk scoring and max-pool
- warn loudly on hash fallback; add MarvinService.health()
- expose first-stage over-fetch as a tunable setting
- persist a wikilink-driven entity graph alongside chunks
- graph stream as third RRF tier in hybrid_search
- expose kg_enabled and kg_rrf_k as MarvinSettings fields
- IDF-weighted graph ranker, fusion weight, opt-in at-ingest extraction
- MARVIN_RERANK_MODEL_FILE env var for ONNX file override
- marvin[gpu] extra + ctypes-CDLL bootstrap of CUDA libs
- --results-dir for SHA-versioned, auto-named benchmark JSONs
- time-aware freshness boost on the final note ranking
- kind-aware boost (episodic by default)
- structured semantic facts as canonical frontmatter
- entity-scoped extraction and reflective synthesis
- procedural templates, two-phase consolidation, fact API
- device-aware reranker weights; optional CPU-pinned embedder
- end-to-end QA arm for LongMemEval-S (reader + judge)
- four-memory retrieval demo over the Midsummer vault

### Fixed

- resolve server tool bugs, add linting, and harden repo
- switch FTS5 to bag-of-words OR semantics

### Changed

- replace experiments/ with examples/demo_vault
- unify prepare_session into a single retrieval pass

### Documentation

- align README with GitHub Pages intro, add cross-links
- increase Mermaid graph height in case study
- cite LongMemEval-S reranker lift (n=100)
- refresh benchmark with real fastembed + GPU rerank, full 500q
- refresh README, site, and skill for the v0.2 feature set

### Other

- Initial commit: Marvin — Obsidian-native long-term memory for AI agents
- Update index.md
