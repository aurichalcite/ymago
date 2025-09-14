## Project Overview

Ymago is an asynchronous CLI and Python library for generative AI media creation using Google's Gemini models. It provides both image and video generation capabilities with cloud storage integration, batch processing, and a robust plugin architecture.

### CLI Commands
```bash
# Generate an image
ymago image generate "A sunset over mountains" --output-filename sunset --seed 42

# Generate a video
ymago video generate "A bird flying" --aspect-ratio 16:9

# Batch processing from CSV/JSONL
ymago batch run input.csv --output-dir ./results --concurrency 10 --rate-limit 60

# Show configuration
ymago config --show-path

# Show version
ymago version
```

## Development Commands

### Environment Setup
```bash
# Install with development dependencies
uv sync --extra dev --extra test

# Activate virtual environment
source .venv/bin/activate
```

### Code Quality
```bash
# Run linting
uv run ruff check .

# Format code
uv run ruff format .

# Type checking with mypy
uv run mypy src

# Type checking with basedpyright (more strict)
uv run basedpyright

# Security scan
uv run bandit -r src
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_specific.py -v

# Run specific test
uv run pytest tests/test_cli.py::TestCLIRunner::test_cli_help_command -xvs

# Run with coverage
uv run coverage run -m pytest tests/
uv run coverage report

# Run tests in parallel
uv run pytest tests/ -n auto

# Run only integration tests
uv run pytest tests/integration/ -v

# Run only unit tests
uv run pytest tests/unit/ -v
```

### Building & Distribution
```bash
# Build package
uv build

# Install locally in editable mode
uv pip install -e ".[dev]"
```

## Architecture

### Core Components

1. **CLI Layer** (`src/ymago/cli.py`)
   - Typer-based command interface with rich formatting
   - Commands: `image`, `video`, `batch`
   - Handles user interaction and progress display

2. **API Layer** (`src/ymago/api.py`)
   - Async interface to Google's Gemini API
   - Implements retry logic with tenacity
   - Manages API authentication and request building

3. **Core Processing** (`src/ymago/core/`)
   - `generation.py`: Orchestrates the generation pipeline
   - `backends.py`: Execution backend abstraction (LocalExecutionBackend, future: CloudTasksBackend)
   - `batch_parser.py`: Handles CSV/JSON batch input parsing
   - `storage.py`: Storage abstraction for local/cloud uploads
   - `io_utils.py`: File I/O and metadata management

4. **Models** (`src/ymago/models.py`)
   - Pydantic v2 models for type safety
   - `GenerationRequest`, `GenerationResult`, `BatchResult`, etc.
   - Validation and serialization logic

5. **Configuration** (`src/ymago/config.py`)
   - Settings management via environment variables and `ymago.toml`
   - API key management
   - Default model and parameter configuration

### Execution Flow

1. CLI command → Parse arguments → Create GenerationRequest
2. Request → ExecutionBackend → API call with retries
3. Response → Storage upload (if configured) → Save metadata
4. Result → Display to user with rich formatting

### Backend Architecture

The system uses an ExecutionBackend abstraction to support different execution strategies:
- **LocalExecutionBackend**: Direct async execution (current)
- **CloudTasksBackend**: Planned for distributed execution via Google Cloud Tasks
- Future backends can be added by implementing the `ExecutionBackend` protocol

### Batch Processing Architecture

The batch processing system (`LocalExecutionBackend.process_batch`) provides:
- **Checkpointing**: Atomic writes to `_batch_state.jsonl` for resume capability
- **Rate Limiting**: `TokenBucketRateLimiter` with configurable requests/minute and burst capacity
- **Concurrency Control**: Semaphore-based limiting of parallel requests
- **Resume Support**: Skips successfully completed requests on resume
- **Error Handling**: Failed requests are logged but don't stop the batch

Key files:
- `backends.py`: Contains `LocalExecutionBackend`, `TokenBucketRateLimiter`, and batch processing logic
- `batch_parser.py`: Handles CSV/JSONL parsing with field mapping and validation
- `models.py`: Defines `GenerationRequest`, `BatchResult`, and `BatchSummary`

### Key Design Patterns

- **Async/Await**: All I/O operations use asyncio for non-blocking execution
- **Dependency Injection**: Settings and backends are injected, not hardcoded
- **Protocol-based Abstractions**: Storage and execution use Python protocols for flexibility
- **Structured Concurrency**: Batch operations use asyncio.gather with proper error handling
- **Metadata Preservation**: Every generation saves a JSON sidecar with full parameters for reproducibility

## Testing Strategy

Tests are organized by module under `tests/` with fixtures in `conftest.py`. The test suite uses:
- `pytest-asyncio` for async test support
- `aioresponses` for mocking HTTP requests
- `pytest-mock` for general mocking
- `hypothesis` for property-based testing

## Configuration Files

- `ymago.toml`: User configuration for API keys, defaults, and output paths
- `pyproject.toml`: Package configuration, dependencies, and tool settings
- `.github/workflows/ci.yml`: CI/CD pipeline configuration

## API Key Management

The system checks for API keys in this order:
1. Command-line argument (`--api-key`)
2. Environment variable (`GOOGLE_API_KEY`)
3. Config file (`ymago.toml`)

## Type System and Testing Notes

### Type Checking
The project uses both mypy and basedpyright for type checking:
- **mypy**: Configured in strict mode in `pyproject.toml`
- **basedpyright**: More strict type checking, configured in `pyproject.toml` under `[tool.basedpyright]`

When using Typer for CLI commands, use `Annotated` types to avoid `reportCallInDefaultInitializer` errors:
```python
# Good
param: Annotated[str, typer.Option("--flag")] = "default"

# Bad (will fail basedpyright)
param: str = typer.Option("default", "--flag")
```

### Common Test Patterns
- **Mocking imports**: When testing methods that import internally (like `_process_request_with_retry`), mock at the module level:
  - `ymago.config.load_config` not `ymago.core.backends.load_config`
  - `ymago.core.generation.process_generation_job` not `ymago.core.backends.process_generation_job`

- **Rate limiter testing**: The `TokenBucketRateLimiter` has burst capacity (bucket_size = rate/10), so consume burst tokens before testing rate limiting

- **Batch resilience**: The checkpoint system only skips requests with `status="success"`. Failed or invalid entries are not automatically retried.

## File Conventions

- All Python files use type hints and are checked with mypy in strict mode and basedpyright
- Ruff handles both linting and formatting (line length: 88)
- Docstrings follow Google style
- Async functions are prefixed with descriptive verbs (e.g., `process_`, `generate_`)
- Test classes should declare instance variables (like `runner: CliRunner`) at class level to satisfy basedpyright