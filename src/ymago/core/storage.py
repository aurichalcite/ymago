"""
Storage abstraction layer for ymago package.

This module provides abstract base classes for storage operations and concrete
implementations for local filesystem storage. Future implementations can extend
this to support cloud storage providers like AWS S3, Google Cloud Storage, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type
from urllib.parse import urlparse

import aiofiles
import aiofiles.os


class StorageError(Exception):
    """Base exception for storage-related errors."""

    pass


class StorageBackendRegistry:
    """
    Registry for storage backend implementations.

    This registry allows dynamic registration of storage backends based on URL schemes,
    enabling a pluggable architecture where new storage providers can be added
    without modifying core library code.
    """

    _backends: Dict[str, Type["StorageUploader"]] = {}

    @classmethod
    def register(cls, scheme: str, backend_class: Type["StorageUploader"]) -> None:
        """
        Register a storage backend for a URL scheme.

        Args:
            scheme: URL scheme (e.g., 's3', 'gs', 'file')
            backend_class: StorageUploader implementation class
        """
        cls._backends[scheme] = backend_class

    @classmethod
    def create_backend(cls, destination_url: str, **kwargs) -> "StorageUploader":
        """
        Create appropriate backend based on destination URL scheme.

        Args:
            destination_url: Destination URL (e.g., 's3://bucket/path', 'gs://bucket/path')
            **kwargs: Additional arguments passed to backend constructor

        Returns:
            StorageUploader: Configured storage backend instance

        Raises:
            ValueError: If URL scheme is not supported
            StorageError: If backend creation fails
        """
        parsed = urlparse(destination_url)
        scheme = parsed.scheme.lower()

        if scheme not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unsupported storage scheme '{scheme}'. Available schemes: {available}"
            )

        backend_class = cls._backends[scheme]
        try:
            return backend_class(destination_url=destination_url, **kwargs)
        except Exception as e:
            raise StorageError(f"Failed to create {scheme} storage backend: {e}") from e

    @classmethod
    def list_schemes(cls) -> list[str]:
        """List all registered storage schemes."""
        return list(cls._backends.keys())


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
    async def upload_bytes(
        self, data: bytes, destination_key: str, content_type: str
    ) -> str:
        """
        Upload raw bytes to the storage backend.

        Args:
            data: Raw bytes to upload
            destination_key: Storage-specific key/path for the uploaded file
            content_type: MIME type of the content

        Returns:
            str: Final storage location/URL of the uploaded file

        Raises:
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

    def __init__(
        self,
        base_directory: Optional[Path] = None,
        destination_url: Optional[str] = None,
        create_dirs: bool = True,
    ):
        """
        Initialize the local storage uploader.

        Args:
            base_directory: Base directory for all file operations (legacy parameter)
            destination_url: File URL for registry pattern (e.g., 'file:///path/to/dir')
            create_dirs: Whether to create directories if they don't exist
        """
        if destination_url:
            parsed = urlparse(destination_url)
            if parsed.scheme != "file":
                raise ValueError(
                    "LocalStorageUploader only supports 'file://' URLs, "
                    f"got: {parsed.scheme}"
                )
            self.base_directory = Path(parsed.path).resolve()
        elif base_directory:
            self.base_directory = Path(base_directory).resolve()
        else:
            raise ValueError(
                "Either base_directory or destination_url must be provided"
            )

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

    async def upload_bytes(
        self, data: bytes, destination_key: str, content_type: str
    ) -> str:
        """
        Write raw bytes to the local storage directory.

        Args:
            data: Raw bytes to write
            destination_key: Relative path within the base directory
            content_type: MIME type (ignored for local storage)

        Returns:
            str: Absolute path to the written file

        Raises:
            PermissionError: If insufficient permissions
            OSError: For other filesystem errors
        """
        destination_path = self.base_directory / destination_key

        # Create destination directory if needed
        if self.create_dirs:
            destination_dir = destination_path.parent
            await aiofiles.os.makedirs(destination_dir, exist_ok=True)

        # Write bytes to file
        try:
            async with aiofiles.open(destination_path, "wb") as dst:
                await dst.write(data)
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


# Register the local storage backend
StorageBackendRegistry.register("file", LocalStorageUploader)

# Import cloud storage backends to register them
try:
    from . import cloud_storage  # noqa: F401
except ImportError:
    # Cloud storage dependencies not available
    pass
