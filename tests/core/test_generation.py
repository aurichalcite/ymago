"""
Tests for ymago core generation orchestration.

This module tests the process_generation_job function and the complete
workflow from job input to result output with mocked dependencies.
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ymago.core.generation import (
    GenerationError,
    StorageError,
    _create_temp_file,
    _generate_filename,
    process_generation_job,
)
from ymago.models import GenerationJob


class TestGenerateFilename:
    """Test the _generate_filename function."""

    def test_generate_filename_with_custom_filename(self, sample_generation_job):
        """Test filename generation with custom filename."""
        job = sample_generation_job
        job.output_filename = "custom_name"

        filename = _generate_filename(job)

        assert filename == "custom_name.png"

    def test_generate_filename_from_prompt(self):
        """Test filename generation from prompt."""
        job = GenerationJob(
            prompt="A beautiful sunset over mountains",
            output_filename=None,
        )

        filename = _generate_filename(job)

        # Should be sanitized prompt + UUID (note: first letter is capitalized)
        assert filename.startswith("A_beautiful_sunset_over_mountains_")
        assert filename.endswith(".png")
        assert len(filename.split("_")[-1].replace(".png", "")) == 8  # UUID length

    def test_generate_filename_sanitizes_special_characters(self):
        """Test filename generation sanitizes special characters."""
        job = GenerationJob(
            prompt='Test/with\\special:characters*and?quotes"',
            output_filename=None,
        )

        filename = _generate_filename(job)

        # Should not contain special characters
        assert "/" not in filename
        assert "\\" not in filename
        assert ":" not in filename
        assert "*" not in filename
        assert "?" not in filename
        assert '"' not in filename

    def test_generate_filename_truncates_long_prompts(self):
        """Test filename generation truncates very long prompts."""
        long_prompt = "word " * 100  # Very long prompt
        job = GenerationJob(prompt=long_prompt, output_filename=None)

        filename = _generate_filename(job)

        # Should be truncated to reasonable length
        assert len(filename) < 100
        assert filename.endswith(".png")


class TestCreateTempFile:
    """Test the _create_temp_file function."""

    @pytest.mark.asyncio
    async def test_create_temp_file_success(self, sample_image_bytes):
        """Test successful temporary file creation."""
        with (
            patch("ymago.core.generation.tempfile.mkstemp") as mock_mkstemp,
            patch("ymago.core.generation.aiofiles.open", create=True) as mock_open,
            patch("os.close") as mock_close,
        ):
            # Mock tempfile creation
            mock_fd = 123
            mock_path = "/tmp/temp_image_abc123"
            mock_mkstemp.return_value = (mock_fd, mock_path)

            # Mock file writing
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__.return_value = mock_file

            result = await _create_temp_file(sample_image_bytes)

            assert result == Path(mock_path)
            mock_mkstemp.assert_called_once_with(suffix=".png", prefix="ymago_")
            mock_close.assert_called_once_with(mock_fd)
            mock_file.write.assert_called_once_with(sample_image_bytes)

    @pytest.mark.asyncio
    async def test_create_temp_file_write_error(self, sample_image_bytes):
        """Test temporary file creation handles write errors."""
        with (
            patch("ymago.core.generation.tempfile.mkstemp") as mock_mkstemp,
            patch("ymago.core.generation.aiofiles.open", create=True) as mock_open,
            patch("os.close") as mock_close,
        ):
            mock_fd = 123
            mock_path = "/tmp/temp_image_abc123"
            mock_mkstemp.return_value = (mock_fd, mock_path)

            # Mock file write error
            mock_file = AsyncMock()
            mock_file.write.side_effect = OSError("Disk full")
            mock_open.return_value.__aenter__.return_value = mock_file

            with pytest.raises(
                GenerationError, match="Failed to create temporary file"
            ):
                await _create_temp_file(sample_image_bytes)

            # File descriptor should still be closed
            mock_close.assert_called_once_with(mock_fd)


class TestProcessGenerationJob:
    """Test the process_generation_job function."""

    @pytest.mark.asyncio
    async def test_process_generation_job_success(
        self, sample_generation_job, sample_config, sample_image_bytes
    ):
        """Test successful generation job processing."""
        with (
            patch(
                "ymago.core.generation.generate_image", new_callable=AsyncMock
            ) as mock_generate,
            patch("ymago.core.generation._create_temp_file") as mock_create_temp,
            patch("ymago.core.generation.LocalStorageUploader") as mock_uploader_class,
            patch("ymago.core.generation.aiofiles.os.remove") as mock_remove,
            patch("ymago.core.generation.aiofiles.os.path.getsize") as mock_getsize,
            patch("ymago.core.generation.aiofiles.os.path.exists") as mock_exists,
        ):
            # Mock API call
            mock_generate.return_value = sample_image_bytes

            # Mock temp file creation
            temp_path = Path("/tmp/temp_image_123.png")
            mock_create_temp.return_value = temp_path

            # Mock storage uploader
            mock_uploader = AsyncMock()
            final_path = "/output/test_image.png"
            mock_uploader.upload.return_value = final_path
            mock_uploader_class.return_value = mock_uploader

            # Mock file size and existence
            mock_getsize.return_value = len(sample_image_bytes)
            mock_exists.return_value = True

            # Record start time for timing validation
            start_time = time.time()

            result = await process_generation_job(sample_generation_job, sample_config)

            end_time = time.time()

            # Verify result structure
            assert result.local_path == Path(final_path)
            assert result.job == sample_generation_job
            assert result.file_size_bytes == len(sample_image_bytes)
            assert 0 <= result.generation_time_seconds <= (end_time - start_time)

            # Verify metadata
            assert result.metadata["api_model"] == sample_generation_job.image_model
            assert result.metadata["prompt_length"] == len(sample_generation_job.prompt)
            assert result.metadata["media_size_bytes"] == len(sample_image_bytes)
            assert "final_filename" in result.metadata
            assert result.metadata["storage_backend"] == "local"
            assert "generation_timestamp" in result.metadata

            # Verify API call
            mock_generate.assert_called_once_with(
                prompt=sample_generation_job.prompt,
                api_key=sample_config.auth.google_api_key,
                model=sample_generation_job.image_model,
                seed=sample_generation_job.seed,
                quality=sample_generation_job.quality,
                aspect_ratio=sample_generation_job.aspect_ratio,
                negative_prompt=sample_generation_job.negative_prompt,
                source_image=None,  # No source image in this test
            )

            # Verify storage operations
            mock_uploader_class.assert_called_once_with(
                base_directory=sample_config.defaults.output_path, create_dirs=True
            )
            mock_uploader.upload.assert_called_once()

            # Verify cleanup
            mock_remove.assert_called_once_with(temp_path)

    @pytest.mark.asyncio
    async def test_process_generation_job_api_error(
        self, sample_generation_job, sample_config
    ):
        """Test generation job processing with API error."""
        with patch(
            "ymago.core.generation.generate_image", new_callable=AsyncMock
        ) as mock_generate:
            # Mock API error
            from ymago.api import APIError

            mock_generate.side_effect = APIError("API quota exceeded")

            with pytest.raises(GenerationError, match="Generation job failed"):
                await process_generation_job(sample_generation_job, sample_config)

    @pytest.mark.asyncio
    async def test_process_generation_job_storage_error(
        self, sample_generation_job, sample_config, sample_image_bytes
    ):
        """Test generation job processing with storage error."""
        with (
            patch(
                "ymago.core.generation.generate_image", new_callable=AsyncMock
            ) as mock_generate,
            patch("ymago.core.generation._create_temp_file") as mock_create_temp,
            patch("ymago.core.generation.LocalStorageUploader") as mock_uploader_class,
            patch("ymago.core.generation.aiofiles.os.remove") as mock_remove,
            patch("ymago.core.generation.aiofiles.os.path.getsize") as mock_getsize,
            patch("ymago.core.generation.aiofiles.os.path.exists") as mock_exists,
        ):
            # Mock successful API call
            mock_generate.return_value = sample_image_bytes

            # Mock temp file creation
            temp_path = Path("/tmp/temp_image_123.png")
            mock_create_temp.return_value = temp_path

            # Mock storage error
            mock_uploader = AsyncMock()
            mock_uploader.upload.side_effect = OSError("Permission denied")
            mock_uploader_class.return_value = mock_uploader

            # Mock file size and existence (won't be reached due to error)
            mock_getsize.return_value = len(sample_image_bytes)
            mock_exists.return_value = True

            with pytest.raises(StorageError, match="Failed to save image to storage"):
                await process_generation_job(sample_generation_job, sample_config)

            # Verify cleanup still happens
            mock_remove.assert_called_once_with(temp_path)

    @pytest.mark.asyncio
    async def test_process_generation_job_temp_file_error(
        self, sample_generation_job, sample_config, sample_image_bytes
    ):
        """Test generation job processing with temp file creation error."""
        with (
            patch(
                "ymago.core.generation.generate_image", new_callable=AsyncMock
            ) as mock_generate,
            patch("ymago.core.generation._create_temp_file") as mock_create_temp,
        ):
            # Mock successful API call
            mock_generate.return_value = sample_image_bytes

            # Mock temp file creation error
            mock_create_temp.side_effect = OSError("No space left on device")

            with pytest.raises(GenerationError, match="Generation job failed"):
                await process_generation_job(sample_generation_job, sample_config)

    @pytest.mark.asyncio
    async def test_process_generation_job_cleanup_on_error(
        self, sample_generation_job, sample_config, sample_image_bytes
    ):
        """Test cleanup occurs even when storage fails."""
        with (
            patch(
                "ymago.core.generation.generate_image", new_callable=AsyncMock
            ) as mock_generate,
            patch("ymago.core.generation._create_temp_file") as mock_create_temp,
            patch("ymago.core.generation.LocalStorageUploader") as mock_uploader_class,
            patch("ymago.core.generation.aiofiles.os.remove") as mock_remove,
            patch("ymago.core.generation.aiofiles.os.path.getsize") as mock_getsize,
            patch("ymago.core.generation.aiofiles.os.path.exists") as mock_exists,
        ):
            # Mock successful API call and temp file creation
            mock_generate.return_value = sample_image_bytes
            temp_path = Path("/tmp/temp_image_123.png")
            mock_create_temp.return_value = temp_path

            # Mock storage failure
            mock_uploader = AsyncMock()
            mock_uploader.upload.side_effect = Exception("Storage failed")
            mock_uploader_class.return_value = mock_uploader

            # Mock file size and existence
            mock_getsize.return_value = len(sample_image_bytes)
            mock_exists.return_value = True

            with pytest.raises(StorageError):
                await process_generation_job(sample_generation_job, sample_config)

            # Verify cleanup was attempted
            mock_remove.assert_called_once_with(temp_path)

    @pytest.mark.asyncio
    async def test_process_generation_job_file_size_calculation(
        self, sample_generation_job, sample_config
    ):
        """Test file size calculation in generation result."""
        large_image_data = b"x" * 5000000  # 5MB of data

        with (
            patch(
                "ymago.core.generation.generate_image", new_callable=AsyncMock
            ) as mock_generate,
            patch("ymago.core.generation._create_temp_file") as mock_create_temp,
            patch("ymago.core.generation.LocalStorageUploader") as mock_uploader_class,
            patch("ymago.core.generation.aiofiles.os.remove"),
            patch("ymago.core.generation.aiofiles.os.path.getsize") as mock_getsize,
            patch("ymago.core.generation.aiofiles.os.path.exists") as mock_exists,
        ):
            # Mock API call with large data
            mock_generate.return_value = large_image_data

            # Mock temp file creation
            temp_path = Path("/tmp/temp_image_123.png")
            mock_create_temp.return_value = temp_path

            # Mock storage uploader
            mock_uploader = AsyncMock()
            mock_uploader.upload.return_value = "/output/large_image.png"
            mock_uploader_class.return_value = mock_uploader

            # Mock file size and existence
            mock_getsize.return_value = len(large_image_data)
            mock_exists.return_value = True

            result = await process_generation_job(sample_generation_job, sample_config)

            # Verify file size matches the generated data
            assert result.file_size_bytes == len(large_image_data)
            assert result.metadata["media_size_bytes"] == len(large_image_data)
