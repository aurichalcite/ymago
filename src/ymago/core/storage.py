"""
Storage abstraction layer for ymago package.

This module provides abstract base classes for storage operations and concrete
implementations for local filesystem storage. Future implementations can extend
this to support cloud storage providers like AWS S3, Google Cloud Storage, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import aiofiles
import aiofiles.os


class StorageUploader(ABC):
    """
    Abstract base class for storage upload operations.

    This interface defines the contract for uploading files to various storage
    backends. Implementations should handle the specifics of each storage type
    while providing a consistent async interface.
    """

    @abstractmethod
    async def upload(self, file_path: Path, destination_key: str) -> str:
        """
        Upload a file to the storage backend.

        Args:
            file_path: Local path to the file to upload
            destination_key: Storage-specific key/path for the uploaded file

        Returns:
            str: Final storage location/URL of the uploaded file

        Raises:
            FileNotFoundError: If the source file doesn't exist
            PermissionError: If there are insufficient permissions
            StorageError: For storage-specific errors
        """
        pass

    @abstractmethod
    async def exists(self, destination_key: str) -> bool:
        """
        Check if a file exists at the given destination.

        Args:
            destination_key: Storage-specific key/path to check

        Returns:
            bool: True if the file exists, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self, destination_key: str) -> bool:
        """
        Delete a file from storage.

        Args:
            destination_key: Storage-specific key/path to delete

        Returns:
            bool: True if the file was deleted, False if it didn't exist

        Raises:
            PermissionError: If there are insufficient permissions
            StorageError: For storage-specific errors
        """
        pass


class LocalStorageUploader(StorageUploader):
    """
    Local filesystem storage implementation.

    This implementation handles file operations on the local filesystem using
    async I/O operations. It's designed for development, testing, and scenarios
    where local storage is preferred.

    Future cloud storage implementations (S3Uploader, GCSUploader, etc.) can
    follow this same pattern while implementing cloud-specific logic.
    """

    def __init__(self, base_directory: Path, create_dirs: bool = True):
        """
        Initialize the local storage uploader.

        Args:
            base_directory: Base directory for all file operations
            create_dirs: Whether to create directories if they don't exist
        """
        self.base_directory = Path(base_directory).resolve()
        self.create_dirs = create_dirs

    async def upload(self, file_path: Path, destination_key: str) -> str:
        """
        Copy a file to the local storage directory.

        Args:
            file_path: Source file path
            destination_key: Relative path within the base directory

        Returns:
            str: Absolute path to the copied file

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If insufficient permissions
            OSError: For other filesystem errors
        """
        source_path = Path(file_path).resolve()
        destination_path = self.base_directory / destination_key

        # Validate source file exists
        if not await aiofiles.os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Create destination directory if needed
        if self.create_dirs:
            destination_dir = destination_path.parent
            await aiofiles.os.makedirs(destination_dir, exist_ok=True)

        # Perform async file copy
        try:
            async with aiofiles.open(source_path, "rb") as src:
                async with aiofiles.open(destination_path, "wb") as dst:
                    # Copy in chunks to handle large files efficiently
                    chunk_size = 64 * 1024  # 64KB chunks
                    while True:
                        chunk = await src.read(chunk_size)
                        if not chunk:
                            break
                        await dst.write(chunk)

            return str(destination_path)

        except Exception as e:
            # Clean up partial file on error
            if await aiofiles.os.path.exists(destination_path):
                try:
                    await aiofiles.os.remove(destination_path)
                except (OSError, IOError) as cleanup_error:
                    # Best effort cleanup - log but don't fail
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to cleanup partial file {destination_path}: "
                        f"{cleanup_error}"
                    )
            raise e

    async def exists(self, destination_key: str) -> bool:
        """Check if a file exists in local storage."""
        file_path = self.base_directory / destination_key
        return await aiofiles.os.path.exists(file_path)

    async def delete(self, destination_key: str) -> bool:
        """Delete a file from local storage."""
        file_path = self.base_directory / destination_key

        if not await aiofiles.os.path.exists(file_path):
            return False

        try:
            await aiofiles.os.remove(file_path)
            return True
        except Exception:
            raise

    def get_full_path(self, destination_key: str) -> Path:
        """Get the full local path for a destination key."""
        return self.base_directory / destination_key
