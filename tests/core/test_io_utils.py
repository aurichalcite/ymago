"""
Tests for ymago.core.io_utils module.

This module tests the I/O utilities including image downloading and metadata writing.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from ymago.core.io_utils import (
    DownloadError,
    FileReadError,
    MetadataError,
    MetadataModel,
    download_image,
    read_image_from_path,
    write_metadata,
)


class TestMetadataModel:
    """Test the MetadataModel Pydantic class."""

    def test_metadata_model_creation(self):
        """Test creating a metadata model with valid data."""
        metadata = MetadataModel(
            prompt="Test prompt",
            model_name="test-model",
            seed=42,
            aspect_ratio="16:9",
        )

        assert metadata.prompt == "Test prompt"
        assert metadata.model_name == "test-model"
        assert metadata.seed == 42
        assert metadata.aspect_ratio == "16:9"
        assert metadata.negative_prompt is None
        assert metadata.source_image_url is None

    def test_metadata_model_with_optional_fields(self):
        """Test creating a metadata model with optional fields."""
        metadata = MetadataModel(
            prompt="Test prompt",
            model_name="test-model",
            seed=42,
            negative_prompt="avoid this",
            source_image_url="https://example.com/image.jpg",
        )

        assert metadata.negative_prompt == "avoid this"
        assert metadata.source_image_url == "https://example.com/image.jpg"

    def test_metadata_model_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated."""
        metadata = MetadataModel(
            prompt="Test prompt",
            model_name="test-model",
            seed=42,
        )

        assert metadata.timestamp_utc is not None
        from datetime import datetime

        assert isinstance(metadata.timestamp_utc, datetime)


class TestDownloadImage:
    """Test the download_image function."""

    @pytest.mark.asyncio
    async def test_download_image_success(self):
        """Test successful image download."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.read = AsyncMock(return_value=b"fake_image_data")

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await download_image("https://example.com/image.jpg")

            assert result == b"fake_image_data"
            mock_get.assert_called_once_with("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_download_image_invalid_url(self):
        """Test download with invalid URL."""
        with pytest.raises(DownloadError, match="Invalid URL format"):
            await download_image("not-a-url")

    @pytest.mark.asyncio
    async def test_download_image_http_error(self):
        """Test download with HTTP error."""
        mock_response = Mock()
        mock_response.status = 404

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(DownloadError, match="HTTP 404"):
                await download_image("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_download_image_invalid_content_type(self):
        """Test download with invalid content type."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.read = AsyncMock(return_value=b"fake_html_data")

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response

            # The function should still work but log a warning
            result = await download_image("https://example.com/image.jpg")
            assert result == b"fake_html_data"

    @pytest.mark.asyncio
    async def test_download_image_network_error(self):
        """Test download with network error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Network error")

            with pytest.raises(DownloadError, match="Network error"):
                await download_image("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_download_image_timeout(self):
        """Test download with timeout."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ServerTimeoutError("Timeout")

            with pytest.raises(DownloadError, match="Network error"):
                await download_image("https://example.com/image.jpg")


class TestWriteMetadata:
    """Test the write_metadata function."""

    @pytest.mark.asyncio
    async def test_write_metadata_success(self):
        """Test successful metadata writing."""
        metadata = MetadataModel(
            prompt="Test prompt",
            model_name="test-model",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "test.json"

            await write_metadata(metadata, metadata_path)

            assert metadata_path.exists()

            with open(metadata_path) as f:
                saved_metadata = json.load(f)

            assert saved_metadata["prompt"] == "Test prompt"
            assert saved_metadata["model_name"] == "test-model"
            assert saved_metadata["seed"] == 42

    @pytest.mark.asyncio
    async def test_write_metadata_with_validation(self):
        """Test metadata writing with Pydantic validation."""
        metadata = MetadataModel(
            prompt="Test prompt",
            model_name="test-model",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "test.json"

            await write_metadata(metadata, metadata_path)

            with open(metadata_path) as f:
                saved_metadata = json.load(f)

            # Check that timestamp was auto-generated
            assert "timestamp_utc" in saved_metadata
            assert saved_metadata["timestamp_utc"] is not None

    @pytest.mark.asyncio
    async def test_write_metadata_permission_error(self):
        """Test metadata writing with permission error."""
        metadata = MetadataModel(
            prompt="Test prompt",
            model_name="test-model",
            seed=42,
        )

        # Try to write to a non-existent directory without creating it
        output_path = Path("/nonexistent/directory/test.json")

        with pytest.raises(MetadataError, match="Failed to write metadata"):
            await write_metadata(metadata, output_path)

    def test_write_metadata_invalid_data(self):
        """Test metadata writing with invalid data."""
        # Test that Pydantic validation works at model creation time
        with pytest.raises(ValueError, match="Field required"):
            MetadataModel(
                prompt="Test prompt",
                # Missing required model_name and seed
            )


class TestReadImageFromPath:
    """Test the read_image_from_path function."""

    @pytest.mark.asyncio
    async def test_read_image_from_path_success(self):
        """Test successful image reading from a path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(b"fake_image_data")
            temp_path = Path(temp_file.name)

        try:
            result = await read_image_from_path(temp_path)
            assert result == b"fake_image_data"
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_read_image_from_path_not_found(self):
        """Test reading a non-existent image file."""
        non_existent_path = Path("non_existent_file.png")
        with pytest.raises(FileReadError):
            await read_image_from_path(non_existent_path)
