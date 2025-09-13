"""
Core data models for ymago package.

This module defines Pydantic models for generation jobs, results, and other
data structures used throughout the application.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL


class GenerationJob(BaseModel):
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

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and clean the prompt text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or only whitespace")
        return cleaned

    @field_validator("negative_prompt")
    @classmethod
    def validate_negative_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean the negative prompt text."""
        if v is None:
            return v
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("from_image")
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

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: Optional[int]) -> Optional[int]:
        """Validate seed value, converting -1 to None for random generation."""
        if v == -1:
            return None
        return v

    @field_validator("output_filename")
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


class GenerationResult(BaseModel):
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

    @field_validator("local_path")
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
