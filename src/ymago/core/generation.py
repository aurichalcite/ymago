"""
Core orchestration layer for ymago package.

This module coordinates the entire image generation process, from API calls
to file storage, with comprehensive error handling and cleanup.
"""

import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os

from ..api import generate_image
from ..config import Settings
from ..core.storage import LocalStorageUploader
from ..models import GenerationJob, GenerationResult


class GenerationError(Exception):
    """Base exception for generation-related errors."""

    pass


class StorageError(Exception):
    """Exception for storage-related errors."""

    pass


async def process_generation_job(
    job: GenerationJob, config: Settings
) -> GenerationResult:
    """
    Process a single generation job from start to finish.

    This function orchestrates the entire generation process:
    1. Generate image using AI API
    2. Save to temporary file
    3. Upload to final storage location
    4. Clean up temporary files
    5. Return result with metadata

    Args:
        job: The generation job to process
        config: Application configuration

    Returns:
        GenerationResult: Complete result with file path and metadata

    Raises:
        GenerationError: For generation-related failures
        StorageError: For storage-related failures
        ValueError: For invalid job parameters
    """
    start_time = time.time()
    temp_file_path: Optional[Path] = None

    try:
        # Step 1: Generate image using AI API
        image_bytes = await generate_image(
            prompt=job.prompt,
            api_key=config.auth.google_api_key,
            model=job.image_model,
            seed=job.seed,
            quality=job.quality,
            aspect_ratio=job.aspect_ratio,
        )

        # Step 2: Create temporary file for the image
        temp_file_path = await _create_temp_file(image_bytes)

        # Step 3: Determine final filename and storage location
        final_filename = _generate_filename(job)

        # Step 4: Set up storage uploader
        storage_uploader = LocalStorageUploader(
            base_directory=config.defaults.output_path, create_dirs=True
        )

        # Step 5: Upload to final storage location
        try:
            final_path = await storage_uploader.upload(
                file_path=temp_file_path, destination_key=final_filename
            )
        except Exception as e:
            raise StorageError(f"Failed to save image to storage: {e}") from e

        # Step 6: Get file size for metadata
        file_size = await aiofiles.os.path.getsize(final_path)

        # Step 7: Create and populate result
        result = GenerationResult(
            local_path=Path(final_path),
            job=job,
            file_size_bytes=file_size,
            generation_time_seconds=time.time() - start_time,
            metadata={
                "api_model": job.image_model,
                "prompt_length": len(job.prompt),
                "image_size_bytes": len(image_bytes),
                "final_filename": final_filename,
                "storage_backend": "local",
                "generation_timestamp": time.time(),
            },
        )

        # Add job-specific metadata
        if job.seed is not None:
            result.add_metadata("seed", job.seed)
        if job.quality:
            result.add_metadata("quality", job.quality)
        if job.aspect_ratio:
            result.add_metadata("aspect_ratio", job.aspect_ratio)

        return result

    except Exception as e:
        # Wrap non-generation errors appropriately
        if isinstance(e, (GenerationError, StorageError)):
            raise
        else:
            raise GenerationError(f"Generation job failed: {e}") from e

    finally:
        # Step 8: Clean up temporary file
        if temp_file_path and await aiofiles.os.path.exists(temp_file_path):
            try:
                await aiofiles.os.remove(temp_file_path)
            except Exception:
                # Log warning but don't fail the operation
                pass


async def _create_temp_file(image_bytes: bytes) -> Path:
    """
    Create a temporary file with the image data.

    Args:
        image_bytes: Raw image data

    Returns:
        Path: Path to the temporary file

    Raises:
        GenerationError: If temporary file creation fails
    """
    try:
        # Create temporary file with .png extension
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="ymago_")
        temp_file_path = Path(temp_path)

        # Close the file descriptor since we'll use aiofiles
        import os

        os.close(temp_fd)

        # Write image data asynchronously
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(image_bytes)

        return temp_file_path

    except Exception as e:
        raise GenerationError(f"Failed to create temporary file: {e}") from e


def _generate_filename(job: GenerationJob) -> str:
    """
    Generate a filename for the output image.

    Args:
        job: The generation job

    Returns:
        str: Generated filename with extension
    """
    if job.output_filename:
        # Use custom filename if provided
        base_name = job.output_filename
    else:
        # Generate filename from prompt and timestamp
        # Clean the prompt for use in filename
        allowed_chars = (" ", "-", "_")
        prompt_clean = "".join(
            c for c in job.prompt[:50] if c.isalnum() or c in allowed_chars
        ).strip()
        prompt_clean = prompt_clean.replace(" ", "_")

        # Add unique identifier
        unique_id = str(uuid.uuid4())[:8]
        base_name = f"{prompt_clean}_{unique_id}"

    # Ensure we have a .png extension
    if not base_name.lower().endswith(".png"):
        base_name += ".png"

    return base_name


async def validate_generation_job(job: GenerationJob) -> None:
    """
    Validate a generation job before processing.

    Args:
        job: The job to validate

    Raises:
        ValueError: If the job is invalid
    """
    # Basic validation is handled by Pydantic, but we can add
    # additional business logic validation here

    if len(job.prompt.strip()) < 3:
        raise ValueError("Prompt must be at least 3 characters long")

    # Add more validation rules as needed
    # For example, check for inappropriate content, validate model availability, etc.
