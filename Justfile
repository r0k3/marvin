# Marvin — reproducible tasks. `just` to list, `just <recipe>` to run.

# Point this at your LongMemEval-S dataset to run the benchmark
# (download with: uv run python scripts/download_longmemeval.py).
longmemeval_dataset := "data/longmemeval_s_cleaned.json"

# List available recipes.
default:
    @just --list

# ---- quality gates ----

# Run the full test suite.
test:
    uv run pytest -q

# Lint + format check (no changes written).
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-format the codebase.
fmt:
    uv run ruff format .

# ---- demonstration ----

# Midsummer Night's Dream demo: four-memory retrieval over a real vault.
demo:
    uv run python -m marvin.eval.demo

# ---- benchmark ----

# LongMemEval-S retrieval benchmark (hybrid + rerank). Set `longmemeval_dataset` first.
eval:
    uv run python -m marvin.eval --dataset {{longmemeval_dataset}} --rerank

# LongMemEval-S end-to-end QA: retrieval + local reader + LLM judge.
# MARVIN_EMBED_CPU=1 keeps the embedder off the GPU so it never contends
# with a co-resident local reader.
eval-qa:
    MARVIN_EMBED_CPU=1 uv run python -m marvin.eval --dataset {{longmemeval_dataset}} --rerank --qa

# ---- docs ----

# Serve the documentation site locally.
docs:
    uv run mkdocs serve

# ---- server & CLI ----

# Run the Marvin MCP server.
serve:
    uv run marvin serve

# Live vault dashboard (AXI CLI).
dashboard:
    uv run marvin
