"""
I/O utilities for ymago package.

This module provides async utilities for downloading images from URLs,
writing metadata files, and other I/O operations with comprehensive
error handling and logging.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp
from pydantic import BaseModel, Field

# Configure logging for this module
logger = logging.getLogger(__name__)


class MetadataModel(BaseModel):
    """
    Pydantic model for generation metadata sidecars.

    This model defines the structure of metadata JSON files that are
    automatically generated alongside media files for reproducibility.
    """

    prompt: str = Field(..., description="Text prompt used for generation")
    negative_prompt: Optional[str] = Field(
        default=None, description="Negative prompt to avoid certain content"
    )
    model_name: str = Field(..., description="AI model used for generation")
    seed: int = Field(..., description="Random seed used for generation")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when generation was completed",
    )
    source_image_url: Optional[str] = Field(
        default=None, description="URL of source image if used"
    )
    aspect_ratio: Optional[str] = Field(
        default=None, description="Aspect ratio for images (e.g., '16:9')"
    )
    generation_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional generation parameters for extensibility",
    )


class DownloadError(Exception):
    """Exception raised when image download fails."""

    pass


class FileReadError(Exception):
    """Exception raised when reading a local file fails."""

    pass


class MetadataError(Exception):
    """Exception raised when metadata writing fails."""

    pass


def _validate_url(url: str) -> None:
    """
    Validate URL format before attempting download.

    Args:
        url: The URL to validate

    Raises:
        DownloadError: If URL format is invalid
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise DownloadError(f"Invalid URL format: {url}")

        if parsed.scheme not in ("http", "https"):
            raise DownloadError(f"Unsupported URL scheme: {parsed.scheme}")

    except Exception as e:
        if isinstance(e, DownloadError):
            raise
        raise DownloadError(f"URL validation failed: {e}") from e


async def download_image(url: str, timeout: int = 30) -> bytes:
    """
    Download image from URL using aiohttp.ClientSession.

    This function downloads an image from a URL with comprehensive error
    handling for network issues, HTTP errors, and invalid URLs.

    Args:
        url: The URL to download the image from
        timeout: Timeout in seconds for the download (default: 30)

    Returns:
        bytes: The downloaded image data

    Raises:
        DownloadError: For any download-related errors including:
            - Invalid URL format
            - Network connectivity issues
            - HTTP errors (404, 403, 500, etc.)
            - Timeout errors
            - Invalid response content
    """
    # Validate URL format first
    _validate_url(url)

    logger.info(f"Starting image download from URL: {url}")

    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)

        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(url) as response:
                # Check HTTP status
                if response.status != 200:
                    raise DownloadError(
                        f"HTTP {response.status}: {response.reason} for URL: {url}"
                    )

                # Check content type if available
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not content_type.startswith("image/"):
                    logger.warning(
                        f"Unexpected content type '{content_type}' for URL: {url}"
                    )

                # Read the response content
                image_data = await response.read()

                if not image_data:
                    raise DownloadError(f"Empty response from URL: {url}")

                logger.info(
                    f"Successfully downloaded image: {len(image_data)} bytes from {url}"
                )
                return image_data

    except aiohttp.ClientError as e:
        # Handle aiohttp-specific errors
        error_msg = f"Network error downloading from {url}: {e}"
        logger.error(error_msg)
        raise DownloadError(error_msg) from e

    except asyncio.TimeoutError as e:
        error_msg = f"Timeout downloading from {url} after {timeout} seconds"
        logger.error(error_msg)
        raise DownloadError(error_msg) from e

    except Exception as e:
        # Handle any other unexpected errors
        error_msg = f"Unexpected error downloading from {url}: {e}"
        logger.error(error_msg)
        raise DownloadError(error_msg) from e


async def write_metadata(metadata: MetadataModel, output_path: Path) -> None:
    """
    Write metadata JSON file using aiofiles.

    This function writes a metadata sidecar file alongside generated media
    for reproducibility and audit trails.

    Args:
        metadata: The metadata model to write
        output_path: Path where the metadata file should be written

    Raises:
        MetadataError: If writing the metadata file fails
    """
    try:
        logger.info(f"Writing metadata to: {output_path}")

        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert metadata to JSON with proper formatting
        metadata_json = metadata.model_dump_json(indent=2, exclude_none=True)

        # Write the file asynchronously
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(metadata_json)

        logger.info(f"Successfully wrote metadata file: {output_path}")

    except Exception as e:
        error_msg = f"Failed to write metadata to {output_path}: {e}"
        logger.error(error_msg)
        raise MetadataError(error_msg) from e


def get_metadata_path(media_path: Path) -> Path:
    """
    Generate the metadata file path for a given media file.

    Args:
        media_path: Path to the media file

    Returns:
        Path: Path where the metadata file should be stored
    """
    return media_path.with_suffix(media_path.suffix + ".json")


async def validate_image_data(image_data: bytes) -> None:
    """
    Validate that downloaded data appears to be a valid image.

    This performs basic validation by checking for common image file signatures.

    Args:
        image_data: The image data to validate

    Raises:
        DownloadError: If the data doesn't appear to be a valid image
    """
    if len(image_data) < 8:
        raise DownloadError("Downloaded data is too small to be a valid image")

    # Check for common image file signatures
    signatures = {
        b"\x89PNG\r\n\x1a\n": "PNG",
        b"\xff\xd8\xff": "JPEG",
        b"GIF87a": "GIF87a",
        b"GIF89a": "GIF89a",
        b"RIFF": "WebP (potential)",  # WebP starts with RIFF
    }

    for signature, format_name in signatures.items():
        if image_data.startswith(signature):
            logger.debug(f"Detected image format: {format_name}")
            return

    # If no signature matches, log a warning but don't fail
    # Some valid images might not have standard signatures
    logger.warning("Could not detect image format from file signature")


async def read_image_from_path(path: Path) -> bytes:
    """
    Read image data from a local file path.

    Args:
        path: The path to the image file.

    Returns:
        bytes: The image data.

    Raises:
        FileReadError: If the file cannot be read.
    """
    try:
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    except Exception as e:
        raise FileReadError(f"Failed to read image from path {path}: {e}") from e
