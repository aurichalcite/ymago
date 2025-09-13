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

from ..api import generate_image, generate_video
from ..config import Settings
from ..core.io_utils import (
    MetadataModel,
    download_image,
    get_metadata_path,
    read_image_from_path,
    write_metadata,
)
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
    1. Download source image if provided
    2. Generate media using AI API (image or video)
    3. Save to temporary file
    4. Upload to final storage location
    5. Generate metadata sidecar if enabled
    6. Clean up temporary files
    7. Return result with metadata

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
    source_image_bytes: Optional[bytes] = None

    try:
        # Step 1: Handle source image if provided
        if job.from_image:
            if job.from_image.lower().startswith(("http://", "https://")):
                source_image_bytes = await download_image(job.from_image)
            else:
                image_path = Path(job.from_image).expanduser()
                source_image_bytes = await read_image_from_path(image_path)

        # Step 2: Generate media using AI API
        if job.media_type == "video":
            media_bytes = await generate_video(
                prompt=job.prompt,
                api_key=config.auth.google_api_key,
                model=job.video_model,
                negative_prompt=job.negative_prompt,
                source_image=source_image_bytes,
            )
        else:
            # Image generation
            media_bytes = await generate_image(
                prompt=job.prompt,
                api_key=config.auth.google_api_key,
                model=job.image_model,
                seed=job.seed,
                quality=job.quality,
                aspect_ratio=job.aspect_ratio,
                negative_prompt=job.negative_prompt,
                source_image=source_image_bytes,
            )

        # Step 3: Create temporary file for the media
        temp_file_path = await _create_temp_file(media_bytes, job.file_extension)

        # Step 4: Determine final filename and storage location
        final_filename = _generate_filename(job)

        # Step 5: Set up storage uploader
        storage_uploader = LocalStorageUploader(
            base_directory=config.defaults.output_path, create_dirs=True
        )

        # Step 6: Upload to final storage location
        try:
            final_path = await storage_uploader.upload(
                file_path=temp_file_path, destination_key=final_filename
            )
        except Exception as e:
            raise StorageError(
                f"Failed to save {job.media_type} to storage: {e}"
            ) from e

        # Step 7: Get file size for metadata
        file_size = await aiofiles.os.path.getsize(final_path)

        # Step 8: Generate metadata sidecar if enabled
        if config.defaults.enable_metadata:
            try:
                metadata = MetadataModel(
                    prompt=job.prompt,
                    negative_prompt=job.negative_prompt,
                    model_name=job.model_name,
                    seed=job.seed or -1,  # Use -1 for random seeds
                    source_image_url=job.from_image,
                    aspect_ratio=job.aspect_ratio,
                    generation_parameters={
                        "media_type": job.media_type,
                        "quality": job.quality,
                        "generation_time_seconds": time.time() - start_time,
                        "file_size_bytes": file_size,
                    },
                )
                metadata_path = get_metadata_path(Path(final_path))
                await write_metadata(metadata, metadata_path)
            except Exception as e:
                # Log warning but don't fail the generation
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to write metadata: {e}")

        # Step 9: Create and populate result
        result = GenerationResult(
            local_path=Path(final_path),
            job=job,
            file_size_bytes=file_size,
            generation_time_seconds=time.time() - start_time,
            metadata={
                "api_model": job.model_name,
                "prompt_length": len(job.prompt),
                "media_size_bytes": len(media_bytes),
                "final_filename": final_filename,
                "storage_backend": "local",
                "generation_timestamp": time.time(),
                "media_type": job.media_type,
            },
        )

        # Add job-specific metadata
        if job.seed is not None:
            result.add_metadata("seed", job.seed)
        if job.quality:
            result.add_metadata("quality", job.quality)
        if job.aspect_ratio:
            result.add_metadata("aspect_ratio", job.aspect_ratio)
        if job.negative_prompt:
            result.add_metadata("negative_prompt", job.negative_prompt)
        if job.from_image:
            result.add_metadata("source_image_url", job.from_image)

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
            except (OSError, IOError) as e:
                # Log warning but don't fail the operation
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {e}")


async def _create_temp_file(media_bytes: bytes, extension: str = ".png") -> Path:
    """
    Create a temporary file with the media data.

    Args:
        media_bytes: Raw media data (image or video)
        extension: File extension to use (e.g., ".png", ".mp4")

    Returns:
        Path: Path to the temporary file

    Raises:
        GenerationError: If temporary file creation fails
    """
    try:
        # Create temporary file with appropriate extension
        temp_fd, temp_path = tempfile.mkstemp(suffix=extension, prefix="ymago_")
        temp_file_path = Path(temp_path)

        # Close the file descriptor since we'll use aiofiles
        import os

        os.close(temp_fd)

        # Write media data asynchronously
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(media_bytes)

        return temp_file_path

    except Exception as e:
        raise GenerationError(f"Failed to create temporary file: {e}") from e


def _generate_filename(job: GenerationJob) -> str:
    """
    Generate a filename for the output media.

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

    # Ensure we have the correct extension
    expected_extension = job.file_extension
    if not base_name.lower().endswith(expected_extension.lower()):
        base_name += expected_extension

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
