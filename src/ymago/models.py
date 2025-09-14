"""
Core data models for ymago package.

This module defines Pydantic models for generation jobs, results, and other
data structures used throughout the application.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL


class GenerationJob(BaseModel):  # type: ignore
    """
    Represents a single media generation job with all required parameters.

    This model encapsulates all the information needed to generate images or videos,
    including the prompt, model configuration, and generation parameters.
    """

    prompt: str = Field(
        ...,
        description="Text prompt for media generation",
        min_length=1,
        max_length=2000,
    )

    media_type: Literal["image", "video"] = Field(
        default="image",
        description="Type of media to generate (image or video)",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation (-1 for random)",
        ge=-1,
        le=2**32 - 1,
    )

    negative_prompt: Optional[str] = Field(
        default=None,
        description="Text describing what not to include in the generated media",
        max_length=1000,
    )

    from_image: Optional[str] = Field(
        default=None,
        description="URL of source image for image-to-image/image-to-video generation",
    )

    image_model: str = Field(
        default=DEFAULT_IMAGE_MODEL,
        description="AI model to use for image generation",
    )

    video_model: str = Field(
        default=DEFAULT_VIDEO_MODEL,
        description="AI model to use for video generation",
    )

    output_filename: Optional[str] = Field(
        default=None,
        description="Custom filename for the generated media (without extension)",
    )

    quality: Optional[str] = Field(
        default="standard",
        description="Media quality setting",
        pattern="^(draft|standard|high)$",
    )

    aspect_ratio: Optional[str] = Field(
        default="1:1",
        description="Aspect ratio for the generated media",
        pattern=r"^\d+:\d+$",
    )

    @field_validator("prompt")  # type: ignore
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and clean the prompt text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or only whitespace")
        return cleaned

    @field_validator("negative_prompt")  # type: ignore
    @classmethod
    def validate_negative_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean the negative prompt text."""
        if v is None:
            return v
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("from_image")  # type: ignore
    @classmethod
    def validate_from_image(cls, v: Optional[str]) -> Optional[str]:
        """Validate source image if provided (URL or path)."""
        if v is None:
            return v

        from urllib.parse import urlparse

        cleaned = v.strip()
        if not cleaned:
            return None

        def is_valid_url(value: str) -> bool:
            try:
                parsed = urlparse(value)
                return parsed.scheme in ("http", "https") and bool(parsed.netloc)
            except Exception:
                return False

        if not is_valid_url(cleaned):
            # Not a valid URL, so we assume it's a path.
            # No further validation here, as path existence is checked in the CLI.
            return cleaned

        # It's a valid URL, so we return it cleaned.
        return cleaned

    @field_validator("seed")  # type: ignore
    @classmethod
    def validate_seed(cls, v: Optional[int]) -> Optional[int]:
        """Validate seed value, converting -1 to None for random generation."""
        if v == -1:
            return None
        return v

    @field_validator("output_filename")  # type: ignore
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        """Validate custom filename if provided."""
        if v is None:
            return v

        # Remove any path separators and clean the filename
        cleaned = Path(v).name.strip()
        if not cleaned:
            raise ValueError("Output filename cannot be empty")

        # Check for invalid characters (basic validation)
        invalid_chars = '<>:"/\\|?*'
        if any(char in cleaned for char in invalid_chars):
            raise ValueError(
                f"Output filename contains invalid characters: {invalid_chars}"
            )

        return cleaned

    @property
    def model_name(self) -> str:
        """Get the appropriate model name based on media type."""
        return self.video_model if self.media_type == "video" else self.image_model

    @property
    def file_extension(self) -> str:
        """Get the appropriate file extension based on media type."""
        return ".mp4" if self.media_type == "video" else ".png"

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class GenerationResult(BaseModel):  # type: ignore
    """
    Represents the result of a completed media generation job.

    This model contains the output information from a successful generation,
    including file paths and metadata about the generation process.
    """

    local_path: Path = Field(
        ..., description="Local filesystem path where the generated media is stored"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the generation process",
    )

    job: GenerationJob = Field(
        ..., description="The original job that produced this result"
    )

    file_size_bytes: Optional[int] = Field(
        default=None, description="Size of the generated media file in bytes", ge=0
    )

    generation_time_seconds: Optional[float] = Field(
        default=None, description="Time taken to generate the media in seconds", ge=0.0
    )

    @field_validator("local_path")  # type: ignore
    @classmethod
    def validate_local_path(cls, v: Path) -> Path:
        """Validate that the local path is absolute."""
        path = Path(v).resolve()
        return path

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata entry to the result."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value with optional default."""
        return self.metadata.get(key, default)

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


# Batch Processing Models


class GenerationRequest(BaseModel):  # type: ignore
    """
    Represents a single generation request within a batch operation.

    This model extends the concept of GenerationJob with batch-specific
    identifiers and tracking information for resilient processing.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request within the batch",
    )

    prompt: str = Field(
        ...,
        description="Text prompt for media generation",
        min_length=1,
        max_length=2000,
    )

    media_type: Literal["image", "video"] = Field(
        default="image",
        description="Type of media to generate (image or video)",
    )

    output_filename: Optional[str] = Field(
        default=None,
        description="Custom filename for the generated media (without extension)",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation (-1 for random)",
        ge=-1,
        le=2**32 - 1,
    )

    negative_prompt: Optional[str] = Field(
        default=None,
        description="Text describing what not to include in the generated media",
        max_length=1000,
    )

    from_image: Optional[str] = Field(
        default=None,
        description=(
            "URL or path of source image for image-to-image/image-to-video generation"
        ),
    )

    quality: Optional[str] = Field(
        default="standard",
        description="Media quality setting",
        pattern="^(draft|standard|high)$",
    )

    aspect_ratio: Optional[str] = Field(
        default="1:1",
        description="Aspect ratio for the media",
        pattern="^\\d+:\\d+$",
    )

    image_model: str = Field(
        default=DEFAULT_IMAGE_MODEL,
        description="AI model to use for image generation",
    )

    video_model: str = Field(
        default=DEFAULT_VIDEO_MODEL,
        description="AI model to use for video generation",
    )

    # Batch-specific metadata
    row_number: Optional[int] = Field(
        default=None,
        description="Original row number from input file (for error reporting)",
        ge=1,
    )

    def to_generation_job(self) -> GenerationJob:
        """Convert this request to a GenerationJob for processing."""
        return GenerationJob(
            prompt=self.prompt,
            media_type=self.media_type,
            output_filename=self.output_filename,
            seed=self.seed,
            negative_prompt=self.negative_prompt,
            from_image=self.from_image,
            quality=self.quality,
            aspect_ratio=self.aspect_ratio,
            image_model=self.image_model,
            video_model=self.video_model,
        )

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class BatchResult(BaseModel):  # type: ignore
    """
    Represents the result of processing a single request within a batch.

    This model captures both successful and failed processing outcomes
    with detailed metadata for tracking and debugging.
    """

    request_id: str = Field(..., description="ID of the original GenerationRequest")

    status: Literal["success", "failure", "skipped"] = Field(
        ..., description="Processing status of this request"
    )

    output_path: Optional[str] = Field(
        default=None,
        description="Path to generated media file (for successful requests)",
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error description (for failed requests)",
    )

    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Time taken to process this request",
        ge=0.0,
    )

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when processing completed",
    )

    file_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of generated file in bytes (for successful requests)",
        ge=0,
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the processing",
    )

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


class BatchSummary(BaseModel):  # type: ignore
    """
    Represents the final summary of a completed batch processing operation.

    This model provides comprehensive statistics and file references
    for the entire batch operation.
    """

    total_requests: int = Field(
        ..., description="Total number of requests processed", ge=0
    )

    successful: int = Field(
        ..., description="Number of successfully processed requests", ge=0
    )

    failed: int = Field(..., description="Number of failed requests", ge=0)

    skipped: int = Field(..., description="Number of skipped requests", ge=0)

    processing_time_seconds: float = Field(
        ..., description="Total time for batch processing", ge=0.0
    )

    results_log_path: str = Field(..., description="Path to detailed results log file")

    rejected_rows_path: Optional[str] = Field(
        default=None,
        description="Path to file containing rejected input rows",
    )

    throughput_requests_per_minute: float = Field(
        ..., description="Average processing throughput", ge=0.0
    )

    start_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when batch processing started",
    )

    end_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when batch processing completed",
    )

    @field_validator("successful", "failed", "skipped")  # type: ignore
    @classmethod
    def validate_counts(cls, v: int, info: Any) -> int:
        """Validate that counts are non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} count cannot be negative")
        return v

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful / self.total_requests) * 100.0

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
