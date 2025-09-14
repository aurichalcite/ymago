"""
Configuration management for ymago package.

This module provides Pydantic schemas for configuration validation and loading,
with support for TOML files and environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any

import tomli
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Auth(BaseModel):
    """Authentication configuration for AI services."""

    google_api_key: str = Field(
        ..., description="Google Generative AI API key for image generation"
    )

    @field_validator("google_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty."""
        if not v.strip():
            raise ValueError("Google API key cannot be empty")
        return v.strip()


class Defaults(BaseModel):
    """Default configuration values for media generation."""

    image_model: str = Field(
        default="gemini-2.5-flash-image-preview",
        description="Default AI model for image generation",
    )

    video_model: str = Field(
        default="veo-3.0-generate-001",
        description="Default AI model for video generation",
    )

    output_path: Path = Field(
        default=Path.cwd() / "generated_media",
        description="Default directory for saving generated media",
    )

    enable_metadata: bool = Field(
        default=True,
        description="Whether to generate metadata sidecar files by default",
    )

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, v: Path) -> Path:
        """Ensure output path is absolute and create if needed."""
        path = Path(v).expanduser().resolve()
        return path


class Settings(BaseModel):
    """Top-level configuration combining authentication and defaults."""

    auth: Auth
    defaults: Defaults = Field(default_factory=Defaults)
    model_config = ConfigDict(validate_assignment=True, extra="forbid")


async def load_config() -> Settings:
    """
    Load configuration from TOML files and environment variables.

    Search order:
    1. ./ymago.toml (current directory)
    2. ~/.ymago.toml (user home directory)
    3. Environment variables (GOOGLE_API_KEY, etc.)

    Returns:
        Settings: Validated configuration object

    Raises:
        FileNotFoundError: If no configuration file is found and required env
            vars are missing
        ValueError: If configuration validation fails
        tomli.TOMLDecodeError: If TOML file is malformed
    """
    config_data = {}
    config_file_found = False

    # Search for configuration files
    config_paths = [Path.cwd() / "ymago.toml", Path.home() / ".ymago.toml"]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    file_config = tomli.load(f)
                    config_data.update(file_config)
                    config_file_found = True
                    break
            except tomli.TOMLDecodeError as e:
                raise ValueError(f"Invalid TOML syntax in {config_path}: {e}") from e
            except Exception as e:
                raise ValueError(f"Error reading config file {config_path}: {e}") from e

    # Apply environment variable overrides
    env_overrides: dict[str, dict[str, Any]] = {}

    # Google API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        env_overrides.setdefault("auth", {})["google_api_key"] = google_api_key

    # Output path from environment
    output_path = os.getenv("YMAGO_OUTPUT_PATH")
    if output_path:
        env_overrides.setdefault("defaults", {})["output_path"] = output_path

    # Image model from environment
    image_model = os.getenv("YMAGO_IMAGE_MODEL")
    if image_model:
        env_overrides.setdefault("defaults", {})["image_model"] = image_model

    # Video model from environment
    video_model = os.getenv("YMAGO_VIDEO_MODEL")
    if video_model:
        env_overrides.setdefault("defaults", {})["video_model"] = video_model

    # Metadata setting from environment
    enable_metadata = os.getenv("YMAGO_ENABLE_METADATA")
    if enable_metadata:
        env_overrides.setdefault("defaults", {})["enable_metadata"] = (
            enable_metadata.lower() in ("true", "1", "yes")
        )

    # Merge environment overrides
    if env_overrides:
        # Deep merge environment overrides into config_data
        for section, values in env_overrides.items():
            if section not in config_data:
                config_data[section] = {}
            config_data[section].update(values)

    # Validate that we have required configuration
    if not config_file_found and not google_api_key:
        raise FileNotFoundError(
            "No configuration file found (ymago.toml or ~/.ymago.toml) and "
            "GOOGLE_API_KEY environment variable is not set. "
            "Please create a configuration file or set the required "
            "environment variables."
        )

    # Ensure auth section exists
    if "auth" not in config_data:
        config_data["auth"] = {}

    # Validate and return settings
    try:
        return Settings(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
