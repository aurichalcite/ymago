"""
Global constants for the ymago package.

This module centralizes all constant values used throughout the application
to ensure consistency and ease of maintenance.
"""

# Default AI models for generation
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image-preview"
DEFAULT_VIDEO_MODEL = "veo-3.0-generate-001"

# API retry configuration
MAX_RETRY_ATTEMPTS = 5
RETRY_MAX_WAIT = 60  # seconds
RETRY_MULTIPLIER = 1

# Video generation timeouts
VIDEO_MAX_WAIT_TIME = 600  # 10 minutes
VIDEO_POLL_INTERVAL = 10  # seconds

# File size limits
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PROMPT_LENGTH = 2000
MAX_NEGATIVE_PROMPT_LENGTH = 1000

# Supported formats
SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_VIDEO_FORMATS = (".mp4", ".webm")

# Rate limiting information (for documentation)
API_RATE_LIMITS = {
    "requests_per_minute": 60,
    "requests_per_day": 1500,
    "retry_after_429": "Automatic exponential backoff (1-60 seconds)",
}
