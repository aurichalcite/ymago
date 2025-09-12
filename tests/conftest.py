"""
Shared pytest fixtures for ymago test suite.

This module provides reusable fixtures for common test data and mock objects
used across multiple test modules.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from ymago.config import Auth, Defaults, Settings
from ymago.models import GenerationJob, GenerationResult


@pytest.fixture
def sample_auth():
    """Provide a sample Auth configuration for testing."""
    return Auth(google_api_key="test_api_key_12345")


@pytest.fixture
def sample_defaults():
    """Provide sample default configuration for testing."""
    return Defaults(
        image_model="gemini-2.5-flash-image-preview",
        output_path=Path("/tmp/test_output"),
    )


@pytest.fixture
def sample_config(sample_auth, sample_defaults):
    """Provide a complete Settings configuration for testing."""
    return Settings(auth=sample_auth, defaults=sample_defaults)


@pytest.fixture
def sample_generation_job():
    """Provide a sample GenerationJob for testing."""
    return GenerationJob(
        prompt="A beautiful sunset over mountains",
        seed=42,
        quality="standard",
        aspect_ratio="1:1",
        image_model="gemini-2.5-flash-image-preview",
        output_filename="test_image",
    )


@pytest.fixture
def sample_generation_result(sample_generation_job):
    """Provide a sample GenerationResult for testing."""
    return GenerationResult(
        local_path=Path("/tmp/test_output/test_image.png"),
        job=sample_generation_job,
        file_size_bytes=1024000,
        generation_time_seconds=2.5,
        metadata={
            "api_model": "gemini-2.5-flash-image-preview",
            "prompt_length": 32,
            "image_size_bytes": 1024000,
            "final_filename": "test_image.png",
            "storage_backend": "local",
            "generation_timestamp": 1234567890.0,
        },
    )


@pytest.fixture
def sample_image_bytes():
    """Provide sample image data for testing."""
    # Create a small fake PNG header for testing
    return b"\x89PNG\r\n\x1a\n" + b"fake_image_data" * 100


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_genai_client():
    """Provide a mocked Google Generative AI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Set up the response structure
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"

    mock_content = MagicMock()
    mock_part = MagicMock()
    mock_inline_data = MagicMock()
    mock_inline_data.data = b"fake_image_data"

    mock_part.inline_data = mock_inline_data
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content

    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_toml_config():
    """Provide sample TOML configuration data."""
    return {
        "auth": {"google_api_key": "toml_api_key_67890"},
        "defaults": {
            "image_model": "gemini-2.5-flash-image-preview",
            "output_path": "/home/user/images",
        },
    }


@pytest.fixture
def mock_env_vars():
    """Provide sample environment variables for testing."""
    return {
        "GOOGLE_API_KEY": "env_api_key_54321",
        "YMAGO_OUTPUT_PATH": "/env/output/path",
        "YMAGO_IMAGE_MODEL": "env-model",
    }


class _NoOpLiveComponent:
    """A no-op class to replace Rich's live-rendering components during tests."""

    def __init__(self, *args, **kwargs):
        pass  # Absorb all arguments without action.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Do not suppress exceptions.

    def update(self, *args, **kwargs):
        pass  # Absorb all update calls.

    def add_task(self, *args, **kwargs):
        return 0  # Return a dummy task ID for Progress compatibility

    def start(self, *args, **kwargs):
        pass  # For Status compatibility

    def stop(self, *args, **kwargs):
        pass  # For Status compatibility

    def refresh(self, *args, **kwargs):
        pass  # For Live compatibility


@pytest.fixture(autouse=True)
def mock_rich_live_display(monkeypatch):
    """
    Automatically mocks Rich live-rendering components and the console
    for all tests to ensure speed and deterministic output.
    """
    # 1. Patch the live-rendering classes to be complete no-ops
    import rich.live
    import rich.progress
    import rich.status

    monkeypatch.setattr(rich.status, "Status", _NoOpLiveComponent)
    monkeypatch.setattr(rich.progress, "Progress", _NoOpLiveComponent)
    monkeypatch.setattr(rich.live, "Live", _NoOpLiveComponent)

    # 2. Create a non-interactive console that still outputs to stdout for CliRunner
    test_console = Console(
        force_terminal=False,
        force_interactive=False,
        no_color=True,
        emoji=False,
        highlight=False,
        width=120,  # Use a fixed width for consistent output wrapping
    )

    # 3. Patch the console instance in the CLI module
    import ymago.cli

    monkeypatch.setattr(ymago.cli, "console", test_console)
