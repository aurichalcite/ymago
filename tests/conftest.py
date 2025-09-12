"""
Shared pytest fixtures for ymago test suite.

This module provides reusable fixtures for common test data and mock objects
used across multiple test modules.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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
