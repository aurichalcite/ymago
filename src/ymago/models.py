"""
Core data models for ymago package.

This module defines Pydantic models for generation jobs, results, and other
data structures used throughout the application.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class GenerationJob(BaseModel):
    """
    Represents a single image generation job with all required parameters.

    This model encapsulates all the information needed to generate an image,
    including the prompt, model configuration, and generation parameters.
    """

    prompt: str = Field(
        ...,
        description="Text prompt for image generation",
        min_length=1,
        max_length=2000,
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation",
        ge=0,
        le=2**32 - 1,
    )

    image_model: str = Field(
        default="gemini-2.5-flash-image-preview",
        description="AI model to use for image generation",
    )

    output_filename: Optional[str] = Field(
        default=None,
        description="Custom filename for the generated image (without extension)",
    )

    quality: Optional[str] = Field(
        default="standard",
        description="Image quality setting",
        pattern="^(draft|standard|high)$",
    )

    aspect_ratio: Optional[str] = Field(
        default="1:1",
        description="Aspect ratio for the generated image",
        pattern="^(1:1|16:9|9:16|4:3|3:4)$",
    )

    @validator("prompt")
    def validate_prompt(cls, v: str) -> str:
        """Validate and clean the prompt text."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or only whitespace")
        return cleaned

    @validator("output_filename")
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

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"


class GenerationResult(BaseModel):
    """
    Represents the result of a completed image generation job.

    This model contains the output information from a successful generation,
    including file paths and metadata about the generation process.
    """

    local_path: Path = Field(
        ..., description="Local filesystem path where the generated image is stored"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the generation process",
    )

    job: GenerationJob = Field(
        ..., description="The original job that produced this result"
    )

    file_size_bytes: Optional[int] = Field(
        default=None, description="Size of the generated image file in bytes", ge=0
    )

    generation_time_seconds: Optional[float] = Field(
        default=None, description="Time taken to generate the image in seconds", ge=0.0
    )

    @validator("local_path")
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

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"
