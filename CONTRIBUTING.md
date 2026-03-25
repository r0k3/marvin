# Contributing to Marvin

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/r0k3/marvin.git
   cd marvin
   ```

2. **Install Python 3.12+**

3. **Install [uv](https://astral.sh/uv)** — a fast Python package manager

4. **Install dependencies**
   ```bash
   uv sync
   ```

5. **Full stack (optional)** — starts NATS, Ollama, MCP Gateway, and Brain Worker:
   ```bash
   docker compose up -d
   ```

## Running Tests

```bash
uv run pytest
```

Tests live in the `tests/` directory.

## Code Style

Marvin uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
src/marvin/         # Core library
├── server.py       # MCP Gateway (FastMCP)
├── service.py      # Business logic layer
├── vault.py        # Obsidian vault management
├── git.py          # Git integration
├── index.py        # Hybrid search index (sqlite-vec + FTS5)
├── embeddings.py   # Local ONNX embeddings (fastembed)
├── broker.py       # NATS message broker
├── worker.py       # Brain Worker daemon
├── consolidation.py # Computational sleep engine
├── extraction.py   # Entity extraction (langextract)
├── models.py       # Pydantic data models
├── config.py       # Configuration (pydantic-settings)
└── skill.md        # Agent skill instructions
tests/              # Pytest test suite
docs/               # MkDocs documentation
experiments/        # Research scripts and experiment outputs
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Ensure tests pass (`uv run pytest`)
3. Follow existing code style (`uv run ruff check .`)
4. Write clear commit messages
5. Open a pull request with a description of your changes
