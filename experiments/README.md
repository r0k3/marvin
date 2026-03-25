# Experiments

This directory contains experiment scripts and generated artifacts from Marvin development and research.

## Scripts

| Script | Description |
|--------|-------------|
| `simulate_midsummer.py` | Demo: multi-phase workflow (conversation → extraction → consolidation → merging) using Shakespeare text |
| `full_play_experiment.py` | Large-scale experiment processing Act I of A Midsummer Night's Dream |
| `run_full_experiment.py` | Full simulation: reading, logging episodes, storing procedures, reflecting, sleep consolidation |
| `call_prepare.py` | Demonstrates `MarvinService.prepare_session()` API |
| `save_memory.py` | Demonstrates semantic memory saving via `MarvinService.remember_semantic()` |
| `generate_obsidian_graph.py` | Generates Mermaid diagrams from vault markdown links |
| `trigger_sleep.py` | Triggers computational sleep/consolidation via the broker |
| `patch_worker.py` | Maintenance script for patching worker.py |
| `update_case_study.py` | Updates docs/guide/case-study.md with experiment descriptions |
| `hello_marvin.py` | Minimal test script |
| `test_entities.py` | Tests entity extraction using langextract |
| `test_goose_sleep.py` | Integration test for Goose CLI consolidation |
| `test_lx.py` | Tests langextract extraction capabilities |
| `test_marvin.py` | Lists MCP tools in MarvinServer |
| `test_prompt.py` | Tests consolidation prompt JSON output parsing |

## Vaults

The `vaults/` directory contains generated Obsidian vault outputs from experiment runs. These are gitignored as they are generated artifacts.
