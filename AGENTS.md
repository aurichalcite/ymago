## Project Overview

ymago is an advanced generative AI media toolkit that provides a CLI and Python library for image and video generation using Google's Gemini models. It features:
- Asynchronous batch processing with resilience mechanisms
- Cloud storage integration (AWS S3, Google Cloud Storage, Cloudflare R2)
- Webhook notifications for job completion
- Token bucket rate limiting with burst capacity
- Atomic checkpoint writing for resumable batch operations

## Quick Start

### Running the CLI
```bash
# Generate an image
uv run ymago image "A sunset over mountains" --output-path ./sunset.png

# Generate with cloud storage
uv run ymago image "A forest scene" --destination s3://my-bucket/images/

# Generate with webhook notification
uv run ymago image "A city skyline" \
  --destination gs://my-bucket/images/ \
  --webhook-url https://api.example.com/webhook

# Generate a video
uv run ymago video "A bird flying through clouds" --output-path ./bird.mp4

# Batch processing with resume capability
uv run ymago batch --from-file prompts.csv --output-dir ./generated/ --resume

# Batch with concurrency and rate limiting
uv run ymago batch --from-file prompts.json \
  --output-dir ./batch_output/ \
  --concurrency 5 \
  --rate-limit 120
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

## Architecture Overview

### Core Components

1. **CLI Layer** (`src/ymago/cli.py`)
   - Typer-based command interface with Annotated type hints
   - Rich formatting for beautiful terminal output
   - Commands: `image`, `video`, `batch`
   - Cloud destination support via `--destination` flag
   - Webhook notification support via `--webhook-url` flag

2. **Generation Core** (`src/ymago/core/generation.py`)
   - Async generation functions using aiohttp
   - Integration with Google's generative AI models
   - Automatic retries with exponential backoff via tenacity
   - Returns GenerationResult with metadata and file size

3. **Batch Processing** (`src/ymago/core/backends.py`)
   - LocalExecutionBackend for concurrent processing
   - TokenBucketRateLimiter with burst capacity (bucket_size = rate/10)
   - Atomic checkpoint writing with asyncio.Lock for race condition prevention
   - Resume functionality from partial completion
   - BatchResult model with timestamp as string (ISO format)

4. **Models** (`src/ymago/models.py`)
   - Pydantic v2 models for validation
   - GenerationJob: Core job representation
   - GenerationRequest: Batch request with optional output_filename
   - GenerationResult: Output with metadata and file size
   - BatchResult: Processing result with timestamp as string
   - WebhookPayload: Notification payload structure
   - Type-safe configuration with proper defaults

5. **Configuration** (`src/ymago/config.py`)
   - TOML-based configuration (`ymago.toml`)
   - Environment variable support with precedence
   - Default model and output path settings
   - Cloud storage credentials configuration
   - Webhook settings (timeout, retry attempts)

6. **Cloud Storage** (`src/ymago/cloud/`)
   - S3Storage: AWS S3 integration with boto3
   - GCSStorage: Google Cloud Storage with google-cloud-storage
   - R2Storage: Cloudflare R2 with boto3 S3 compatibility
   - Automatic public URL generation

7. **Webhook Notifications** (`src/ymago/webhooks.py`)
   - Async webhook delivery with aiohttp
   - Configurable retry logic with exponential backoff
   - WebhookPayload model for structured notifications
   - Support for success and failure notifications

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

## Type Checking

The project uses both mypy and basedpyright for type checking:
- **mypy**: Configured in strict mode in `pyproject.toml`
- **basedpyright**: More strict type checking, configured in `pyproject.toml`

When using Typer for CLI commands, use `Annotated` types to avoid `reportCallInDefaultInitializer` errors:
```python
# Correct pattern
from typing import Annotated, Optional
import typer

def command(
    param: Annotated[
        Optional[str],
        typer.Option("--param", help="Parameter help")
    ] = None,  # Default value goes after the Annotated type
):
    pass

# Incorrect pattern (will cause basedpyright errors)
def command(
    param: str = typer.Option(default="value", help="Parameter help")
):
    pass
```

## Testing Strategy

Tests are organized by module under `tests/` with fixtures in `conftest.py`. The test suite uses:
- `pytest-asyncio` for async test support
- `aioresponses` for mocking HTTP requests
- `pytest-mock` for general mocking
- `hypothesis` for property-based testing

### Common Test Patterns

1. **Mocking API Calls**
```python
from unittest.mock import patch, MagicMock

# Mock at the module level, not the import source
@patch('ymago.core.generation.process_generation_job')
def test_generation(mock_process):
    mock_process.return_value = GenerationResult(...)
    # Test code
```

2. **Async Test Fixtures**
```python
import pytest
import asyncio

@pytest.fixture
async def async_client():
    # Setup
    client = await create_client()
    yield client
    # Teardown
    await client.close()

# For testing concurrent operations
@pytest.mark.asyncio
async def test_concurrent_processing():
    tasks = [process_item(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
```

3. **Testing with Temporary Files**
```python
import tempfile
from pathlib import Path

def test_with_temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.png"
        # Test code using output_path

# For checkpoint files
def test_checkpoint_writing():
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = Path(temp_dir) / "_batch_state.jsonl"
        # Write and verify checkpoint data
```

4. **Testing Rate Limiters**
```python
# Account for burst capacity when testing rate limits
limiter = TokenBucketRateLimiter(60)  # 60 requests/minute

# Consume burst tokens first (bucket_size = 60/10 = 6)
for _ in range(int(limiter.bucket_size)):
    await limiter.acquire()

# Now test actual rate limiting
start_time = time.time()
await limiter.acquire()
elapsed = time.time() - start_time
assert elapsed >= 0.9  # Should wait ~1 second
```

5. **Mocking Config Loading**
```python
# Always mock at the import location, not the definition
with patch('ymago.config.load_config') as mock_config:
    mock_config.return_value = MagicMock(
        google_api_key="test-key",
        default_model="test-model"
    )
    # Test code
```

## Common Gotchas

1. **Async Context Managers**: Always use `async with` for async resources
2. **Type Annotations**:
   - Use `Optional[T]` for nullable types
   - Use `Annotated[T, typer.Option()]` for Typer CLI parameters
   - Avoid `typer.Option()` as default value directly
3. **Path Handling**: Always use `pathlib.Path` for cross-platform compatibility
4. **Error Messages**: Include context in error messages for debugging
5. **Checkpoint Race Conditions**: Use asyncio.Lock when writing to shared checkpoint files
6. **Rate Limiter Burst**: TokenBucketRateLimiter has burst capacity = rate/10
7. **Timestamp Types**: Use string (ISO format) for timestamps in models, not datetime
8. **Mock Imports**: Mock at the usage location, not the definition location

## Development Workflow

1. **Before Committing**:
   ```bash
   # Format code
   uv run ruff format

   # Check linting
   uv run ruff check --fix

   # Type checking
   uv run mypy src/
   uv run basedpyright  # More strict checking

   # Run tests
   uv run pytest
   ```

2. **Adding New Features**:
   - Create feature branch from `main`
   - Add tests for new functionality
   - Update type hints and docstrings
   - Run full test suite and type checking
   - Update this CLAUDE.md if adding new patterns

## Key Files to Know

- `src/ymago/cli.py`: Main CLI entry point with Typer commands
- `src/ymago/core/generation.py`: Core async generation logic
- `src/ymago/core/backends.py`: Batch processing with LocalExecutionBackend
- `src/ymago/core/batch_parser.py`: CSV/JSON batch input parsing
- `src/ymago/models.py`: Pydantic v2 data models
- `src/ymago/config.py`: Configuration management with TOML support
- `src/ymago/cloud/`: Cloud storage implementations (S3, GCS, R2)
- `src/ymago/webhooks.py`: Webhook notification system
- `pyproject.toml`: Project configuration and dependencies
- `ymago.toml`: User configuration file (optional)
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for end-to-end flows

## Project Philosophy

- **Type Safety**: Everything should be properly typed with mypy and basedpyright
- **Async First**: Use async/await for all I/O operations
- **User Experience**: Rich CLI output with progress indicators and clear errors
- **Resilience**: Automatic retries, checkpointing, and graceful error handling
- **Testability**: Comprehensive test coverage with proper mocking patterns
- **Scalability**: Designed for batch processing with concurrent execution
- **Cloud Native**: Built-in support for cloud storage and webhooks
- **Atomic Operations**: Ensure file operations are atomic to prevent corruption

## Getting Help

When working on this project:
1. Read the existing code for patterns and conventions
2. Check test files for usage examples and mocking patterns
3. Use type hints and docstrings to understand function signatures
4. Run tests frequently to catch regressions
5. Use `uv run basedpyright` for strict type checking
6. Check `ymago.toml.example` for configuration options

## Environment Variables

```bash
# Required
export GOOGLE_API_KEY="your-api-key"

# Optional - Cloud Storage
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export R2_ACCOUNT_ID="your-account-id"
export R2_ACCESS_KEY_ID="your-key"
export R2_SECRET_ACCESS_KEY="your-secret"
```

## Batch File Formats

### CSV Format
```csv
prompt,output_filename,media_type,seed,negative_prompt
"A sunset",sunset1,image,42,"darkness"
"A forest",forest1,image,,
```

### JSON Format
```json
[
  {
    "prompt": "A sunset over mountains",
    "output_filename": "sunset1",
    "media_type": "image",
    "seed": 42,
    "negative_prompt": "darkness, night"
  },
  {
    "prompt": "A forest scene",
    "output_filename": "forest1"
  }
]
```

## Configuration Files

- `ymago.toml`: User configuration for API keys, defaults, and output paths
- `pyproject.toml`: Package configuration, dependencies, and tool settings
- `.github/workflows/ci.yml`: CI/CD pipeline configuration

## API Key Management

The system checks for API keys in this order:
1. Command-line argument (`--api-key`)
2. Environment variable (`GOOGLE_API_KEY`)
3. Config file (`ymago.toml`)

## File Conventions

- All Python files use type hints and are checked with mypy in strict mode and basedpyright
- Ruff handles both linting and formatting (line length: 88)
- Docstrings follow Google style
- Async functions are prefixed with descriptive verbs (e.g., `process_`, `generate_`)
- Test classes should declare instance variables (like `runner: CliRunner`) at class level to satisfy basedpyright