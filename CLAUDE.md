# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ymago is an async CLI toolkit and Python library for generative AI media (images/video) using Google's Gemini models. Built with Typer (CLI), Rich (UI), aiohttp (HTTP), Pydantic v2 (models), and tenacity (retries). Requires Python 3.10+, managed with uv.

## Common Commands

```bash
# Setup
uv sync --extra dev --extra test

# Run CLI
uv run ymago image "prompt" --output-path ./out.png
uv run ymago video "prompt" --output-path ./out.mp4
uv run ymago batch --from-file prompts.csv --output-dir ./out/

# Tests
uv run pytest tests/ -v                    # all tests
uv run pytest tests/test_cli.py -v         # single file
uv run pytest tests/test_cli.py::TestClass::test_name -xvs  # single test
uv run pytest tests/ -n auto               # parallel

# Lint & Type Check
uv run ruff check .                        # lint
uv run ruff format .                       # format
uv run mypy src                            # type check (strict)
uv run basedpyright                        # stricter type check
uv run bandit -r src                       # security scan

# Coverage
uv run coverage run -m pytest tests/ && uv run coverage report
```

## Architecture

**CLI Layer** (`src/ymago/cli.py`) — Typer commands (`image`, `video`, `batch`) with Rich output. Uses `Annotated` types for parameters (required by basedpyright).

**Models** (`src/ymago/models.py`) — Pydantic v2 models: `GenerationJob` (single job), `GenerationRequest` (batch item with UUID), `GenerationResult`, `BatchResult` (timestamp as ISO string, not datetime), `BatchSummary`.

**Generation** (`src/ymago/core/generation.py`) — Async generation via google-genai with tenacity retries. Returns `GenerationResult` with metadata and file size.

**Batch Processing** (`src/ymago/core/backends.py`) — `LocalExecutionBackend` implements `ExecutionBackend` protocol. Features `TokenBucketRateLimiter` (burst capacity = rate/10), semaphore-based concurrency, atomic checkpoint writing (`_batch_state.jsonl`) with `asyncio.Lock`, and resume support.

**Batch Parsing** (`src/ymago/core/batch_parser.py`) — Parses CSV/JSONL input into `GenerationRequest` objects.

**Storage** (`src/ymago/core/storage.py`) — `StorageUploader` protocol with `LocalStorageUploader` and cloud backends registered via `StorageBackendRegistry`.

**Cloud Storage** (`src/ymago/core/cloud_storage.py`) — S3, GCS, and R2 uploaders. Optional deps: `ymago[aws]`, `ymago[gcp]`, `ymago[r2]`, `ymago[cloud]`.

**Notifications** (`src/ymago/core/notifications.py`) — Async webhook delivery with retry logic.

**Config** (`src/ymago/config.py`) — Loads from `ymago.toml`, env vars, or CLI args (in that precedence order: CLI > env > config file). `GOOGLE_API_KEY` is the primary required env var.

## Key Conventions

- **Typer params**: Always use `Annotated[Optional[T], typer.Option(...)] = None` pattern, never `typer.Option()` as default value (causes basedpyright `reportCallInDefaultInitializer` errors)
- **Timestamps**: Use ISO format strings in models, not `datetime` objects
- **Mocking**: Mock at the import/usage location, not the definition location
- **Async**: All I/O operations use async/await; use `async with` for async context managers
- **Paths**: Always use `pathlib.Path`
- **Ruff**: Line length 88, selects E/F/B/I (flake8, pyflakes, bugbear, isort)
- **mypy**: Strict mode with pydantic plugin
- **Tests**: pytest-asyncio for async tests, aioresponses for HTTP mocking, hypothesis for property-based testing
- **Rate limiter gotcha**: `TokenBucketRateLimiter` has burst capacity (bucket_size = rate/10); consume burst tokens first when testing rate limiting
- **Checkpoint race conditions**: Use `asyncio.Lock` when writing to shared checkpoint files
- **Pydantic v2 required**: Runtime guard in `__init__.py` enforces this
