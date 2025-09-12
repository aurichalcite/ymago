"""
Tests for ymago storage layer.

This module tests the LocalStorageUploader class and storage operations
with mocked aiofiles operations.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ymago.core.storage import LocalStorageUploader


class TestLocalStorageUploader:
    """Test the LocalStorageUploader class."""

    def test_init_with_absolute_path(self):
        """Test initialization with absolute path."""
        base_dir = Path("/test/absolute/path")
        uploader = LocalStorageUploader(base_directory=base_dir, create_dirs=True)

        assert uploader.base_directory == base_dir.resolve()
        assert uploader.create_dirs is True

    def test_init_with_relative_path(self):
        """Test initialization with relative path resolves to absolute."""
        base_dir = Path("./relative/path")
        uploader = LocalStorageUploader(base_directory=base_dir, create_dirs=False)

        assert uploader.base_directory.is_absolute()
        assert uploader.create_dirs is False

    @pytest.mark.asyncio
    async def test_upload_success(self, temp_directory, sample_image_bytes):
        """Test successful file upload."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir, create_dirs=True)

        source_file = temp_directory / "source.png"
        destination_key = "images/test_image.png"

        with (
            patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists,
            patch("ymago.core.storage.aiofiles.os.makedirs") as mock_makedirs,
            patch("ymago.core.storage.aiofiles.open", create=True) as mock_open,
        ):
            # Mock source file exists
            mock_exists.return_value = True

            # Mock file operations
            mock_source_file = AsyncMock()
            mock_dest_file = AsyncMock()

            # IMPORTANT: First read returns data, second returns empty bytes (EOF)
            mock_source_file.read.side_effect = [sample_image_bytes, b""]
            mock_dest_file.write = AsyncMock()

            def open_side_effect(path, mode="r", **kwargs):
                mock_context = AsyncMock()
                if "source" in str(path):
                    mock_context.__aenter__.return_value = mock_source_file
                else:
                    mock_context.__aenter__.return_value = mock_dest_file
                return mock_context

            mock_open.side_effect = open_side_effect

            result = await uploader.upload(source_file, destination_key)

            expected_path = base_dir / destination_key
            assert result == str(expected_path.resolve())

            # Verify directory creation was called
            mock_makedirs.assert_called_once_with(expected_path.parent, exist_ok=True)

            # Verify file operations
            mock_source_file.read.assert_called()
            mock_dest_file.write.assert_called_with(sample_image_bytes)

    @pytest.mark.asyncio
    async def test_upload_source_file_not_found(self, temp_directory):
        """Test upload raises FileNotFoundError when source doesn't exist."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir)

        source_file = temp_directory / "nonexistent.png"
        destination_key = "images/test_image.png"

        with patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists:
            mock_exists.return_value = False

            with pytest.raises(FileNotFoundError, match="Source file not found"):
                await uploader.upload(source_file, destination_key)

    @pytest.mark.asyncio
    async def test_upload_without_create_dirs(self, temp_directory, sample_image_bytes):
        """Test upload without directory creation."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir, create_dirs=False)

        source_file = temp_directory / "source.png"
        destination_key = "test_image.png"

        with (
            patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists,
            patch("ymago.core.storage.aiofiles.os.makedirs") as mock_makedirs,
            patch("ymago.core.storage.aiofiles.open", create=True) as mock_open,
        ):
            mock_exists.return_value = True

            # Mock file operations
            mock_source_file = AsyncMock()
            mock_dest_file = AsyncMock()

            # IMPORTANT: First read returns data, second returns empty bytes (EOF)
            mock_source_file.read.side_effect = [sample_image_bytes, b""]

            def open_side_effect(path, mode="r", **kwargs):
                mock_context = AsyncMock()
                if "source" in str(path):
                    mock_context.__aenter__.return_value = mock_source_file
                else:
                    mock_context.__aenter__.return_value = mock_dest_file
                return mock_context

            mock_open.side_effect = open_side_effect

            await uploader.upload(source_file, destination_key)

            # Directory creation should not be called
            mock_makedirs.assert_not_called()

    @pytest.mark.asyncio
    async def test_exists_file_exists(self, temp_directory):
        """Test exists method returns True for existing file."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir)

        file_key = "images/test_image.png"

        with patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists:
            mock_exists.return_value = True

            result = await uploader.exists(file_key)

            assert result is True
            expected_path = base_dir / file_key
            mock_exists.assert_called_once_with(expected_path)

    @pytest.mark.asyncio
    async def test_exists_file_not_exists(self, temp_directory):
        """Test exists method returns False for non-existing file."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir)

        file_key = "images/nonexistent.png"

        with patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists:
            mock_exists.return_value = False

            result = await uploader.exists(file_key)

            assert result is False

    @pytest.mark.asyncio
    async def test_delete_existing_file(self, temp_directory):
        """Test delete method removes existing file."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir)

        file_key = "images/test_image.png"

        with (
            patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists,
            patch("ymago.core.storage.aiofiles.os.remove") as mock_remove,
        ):
            mock_exists.return_value = True

            result = await uploader.delete(file_key)

            assert result is True
            expected_path = base_dir / file_key
            mock_remove.assert_called_once_with(expected_path)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, temp_directory):
        """Test delete method returns False for non-existing file."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir)

        file_key = "images/nonexistent.png"

        with patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists:
            mock_exists.return_value = False

            result = await uploader.delete(file_key)

            assert result is False

    @pytest.mark.asyncio
    async def test_upload_permission_error(self, temp_directory):
        """Test upload handles permission errors gracefully."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir, create_dirs=True)

        source_file = temp_directory / "source.png"
        destination_key = "images/test_image.png"

        with (
            patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists,
            patch("ymago.core.storage.aiofiles.os.makedirs") as mock_makedirs,
        ):
            mock_exists.return_value = True
            mock_makedirs.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                await uploader.upload(source_file, destination_key)

    @pytest.mark.asyncio
    async def test_upload_with_chunked_reading(self, temp_directory):
        """Test upload handles large files with chunked reading."""
        base_dir = temp_directory / "output"
        uploader = LocalStorageUploader(base_directory=base_dir, create_dirs=True)

        source_file = temp_directory / "large_source.png"
        destination_key = "images/large_image.png"

        # Use small test chunks to avoid memory issues
        chunk1 = b"test_chunk_1"
        chunk2 = b"test_chunk_2"
        chunk3 = b""  # End of file - this is critical for stopping the while loop

        with (
            patch("ymago.core.storage.aiofiles.os.path.exists") as mock_exists,
            patch("ymago.core.storage.aiofiles.os.makedirs"),
            patch("ymago.core.storage.aiofiles.open", create=True) as mock_open,
        ):
            mock_exists.return_value = True

            # Mock file operations with chunked reading
            mock_source_file = AsyncMock()
            mock_dest_file = AsyncMock()

            # IMPORTANT: The empty bytes b"" signals EOF and stops the while loop
            mock_source_file.read.side_effect = [chunk1, chunk2, chunk3]
            mock_dest_file.write = AsyncMock()

            def open_side_effect(path, mode="r", **kwargs):
                mock_context = AsyncMock()
                if "large_source" in str(path):
                    mock_context.__aenter__.return_value = mock_source_file
                else:
                    mock_context.__aenter__.return_value = mock_dest_file
                return mock_context

            mock_open.side_effect = open_side_effect

            result = await uploader.upload(source_file, destination_key)

            expected_path = base_dir / destination_key
            assert result == str(expected_path.resolve())

            # Verify chunked writing (only non-empty chunks are written)
            assert mock_dest_file.write.call_count == 2  # chunk1 and chunk2
            mock_dest_file.write.assert_any_call(chunk1)
            mock_dest_file.write.assert_any_call(chunk2)
