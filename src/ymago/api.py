"""
AI service integration for ymago package.

This module provides integration with Google's Generative AI service for image
generation, including resilient API calls with retry logic and comprehensive
error handling.
"""

import asyncio
import logging
from typing import Any

import google.genai as genai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Configure logging for this module
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""

    pass


class QuotaExceededError(APIError):
    """Raised when API quota is exceeded."""

    pass


class InvalidResponseError(APIError):
    """Raised when API returns an invalid response."""

    pass


class NetworkError(APIError):
    """Raised for network-related errors."""

    pass


def _classify_exception(exc: Exception) -> Exception:
    """
    Classify exceptions into appropriate error types for better handling.

    Args:
        exc: The original exception

    Returns:
        Exception: Classified exception
    """
    exc_str = str(exc).lower()

    # Check for quota/rate limit errors
    quota_keywords = ["quota", "rate limit", "too many requests"]
    if any(keyword in exc_str for keyword in quota_keywords):
        return QuotaExceededError(f"API quota exceeded: {exc}")

    # Check for network errors
    if any(keyword in exc_str for keyword in ["network", "connection", "timeout"]):
        return NetworkError(f"Network error: {exc}")

    # Check for invalid response errors
    if any(keyword in exc_str for keyword in ["invalid", "malformed", "parse"]):
        return InvalidResponseError(f"Invalid API response: {exc}")

    # Default to generic API error
    return APIError(f"API error: {exc}")


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((NetworkError, QuotaExceededError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def generate_image(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash-image-preview",
    **params: Any,
) -> bytes:
    """
    Generate an image from a text prompt using Google's Generative AI.

    This function includes comprehensive retry logic with exponential backoff
    for handling rate limits and transient network issues.

    Args:
        prompt: Text prompt for image generation
        api_key: Google Generative AI API key
        model: AI model to use for generation
        **params: Additional parameters for image generation

    Returns:
        bytes: Generated image data

    Raises:
        QuotaExceededError: When API quota is exceeded
        InvalidResponseError: When API returns invalid response
        NetworkError: For network-related issues
        APIError: For other API errors
        ValueError: For invalid parameters
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if not api_key.strip():
        raise ValueError("API key cannot be empty")

    try:
        # Configure the client with the API key
        client = genai.Client(api_key=api_key)

        # Log the generation attempt (without sensitive data)
        logger.info(
            f"Generating image with model {model}, prompt length: {len(prompt)}"
        )

        # Make the API call with additional parameters
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=[prompt.strip()],
            **params,
        )

        # Validate response structure
        if not response or not hasattr(response, "candidates"):
            raise InvalidResponseError("API response missing candidates")

        if not response.candidates:
            raise InvalidResponseError("API response contains no candidates")

        candidate = response.candidates[0]

        # Check for content policy violations
        if hasattr(candidate, "finish_reason") and candidate.finish_reason != "STOP":
            if candidate.finish_reason == "SAFETY":
                raise APIError("Content was blocked due to safety policies")
            else:
                raise APIError(
                    f"Generation stopped with reason: {candidate.finish_reason}"
                )

        # Extract image data
        if not hasattr(candidate, "content") or not candidate.content:
            raise InvalidResponseError("API response missing content")

        content = candidate.content

        # Look for image parts in the content
        image_data = None
        if hasattr(content, "parts") and content.parts is not None:
            for part in content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    if hasattr(part.inline_data, "data"):
                        image_data = part.inline_data.data
                        break

        if image_data is None:
            raise InvalidResponseError("No image data found in API response")

        # Convert to bytes if needed
        if isinstance(image_data, str):
            import base64

            try:
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                raise InvalidResponseError(
                    f"Failed to decode base64 image data: {e}"
                ) from e
        else:
            image_bytes = bytes(image_data)

        if not image_bytes:
            raise InvalidResponseError("Image data is empty")

        logger.info(f"Successfully generated image, size: {len(image_bytes)} bytes")
        return image_bytes

    except Exception as exc:
        # Classify and re-raise the exception
        classified_exc = _classify_exception(exc)
        logger.error(f"Image generation failed: {classified_exc}")
        raise classified_exc from exc


async def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key by making a simple test request.

    Args:
        api_key: The API key to validate

    Returns:
        bool: True if the API key is valid, False otherwise
    """
    try:
        # Try a simple generation with a minimal prompt
        await generate_image(
            prompt="test", api_key=api_key, model="gemini-2.5-flash-image-preview"
        )
        return True
    except Exception:
        return False
