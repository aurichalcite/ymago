"""
AI service integration for ymago package.

This module provides integration with Google's Generative AI service for image
generation, including resilient API calls with retry logic and comprehensive
error handling.
"""

import asyncio
import logging
import time
from typing import Any, Optional

import google.genai as genai
from google.genai import types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .constants import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
    MAX_RETRY_ATTEMPTS,
    RETRY_MAX_WAIT,
    RETRY_MULTIPLIER,
    VIDEO_MAX_WAIT_TIME,
    VIDEO_POLL_INTERVAL,
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
    wait=wait_random_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT),
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    retry=retry_if_exception_type((NetworkError, QuotaExceededError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def generate_image(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_IMAGE_MODEL,
    negative_prompt: Optional[str] = None,
    source_image: Optional[bytes] = None,
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
        negative_prompt: Text describing what not to include
        source_image: Optional source image bytes for image-to-image
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

        # Prepare the prompt, potentially including negative prompt
        full_prompt = prompt.strip()
        if negative_prompt:
            # For image generation, we can append negative prompt guidance
            full_prompt += f" (avoid: {negative_prompt.strip()})"

        # Prepare contents for the API call
        contents: list[Any] = [full_prompt]

        # Add source image if provided (for image-to-image generation)
        if source_image:
            image_obj = types.Image(image_bytes=source_image, mime_type="image/png")
            contents.append(image_obj)

        # Prepare generation config
        config = types.GenerateContentConfig(seed=params.get("seed"))

        # Make the API call with additional parameters
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
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


@retry(
    wait=wait_random_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT),
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    retry=retry_if_exception_type((NetworkError, QuotaExceededError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def generate_video(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_VIDEO_MODEL,
    negative_prompt: Optional[str] = None,
    source_image: Optional[bytes] = None,
) -> bytes:
    """
    Generate a video from a text prompt using Google's Veo model.

    This function includes comprehensive retry logic with exponential backoff
    for handling rate limits and transient network issues. Video generation
    is asynchronous and requires polling for completion.

    Args:
        prompt: Text prompt for video generation
        api_key: Google Generative AI API key
        model: AI model to use for video generation
        negative_prompt: Text describing what not to include
        source_image: Optional source image bytes for image-to-video
        **params: Additional parameters for video generation

    Returns:
        bytes: Generated video data

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
            f"Generating video with model {model}, prompt length: {len(prompt)}"
        )

        # Start video generation operation
        if source_image:
            # Image-to-video generation
            from google.genai import types

            image_obj = types.Image(image_bytes=source_image, mime_type="image/png")
            operation = await asyncio.to_thread(
                client.models.generate_videos,
                model=model,
                prompt=prompt.strip(),
                image=image_obj,
            )
        else:
            # Text-to-video generation
            operation = await asyncio.to_thread(
                client.models.generate_videos,
                model=model,
                prompt=prompt.strip(),
            )

        # Poll for completion
        logger.info(f"Video generation started, operation: {operation.name}")

        max_wait_time = VIDEO_MAX_WAIT_TIME
        poll_interval = VIDEO_POLL_INTERVAL
        start_time = time.time()

        while not operation.done:
            if time.time() - start_time > max_wait_time:
                raise APIError(
                    f"Video generation timed out after {max_wait_time} seconds"
                )

            logger.debug("Waiting for video generation to complete...")
            await asyncio.sleep(poll_interval)

            # Refresh operation status
            operation = await asyncio.to_thread(client.operations.get, operation)

        # Check if operation completed successfully
        if not hasattr(operation, "response") or not operation.response:
            raise InvalidResponseError(
                "Video generation operation completed without response"
            )

        response = operation.response

        # Extract video data
        if not hasattr(response, "generated_videos") or not response.generated_videos:
            raise InvalidResponseError("No videos found in generation response")

        generated_video = response.generated_videos[0]

        if not hasattr(generated_video, "video") or not generated_video.video:
            raise InvalidResponseError("No video data found in response")

        # Download the video file
        video_file = generated_video.video
        video_bytes = await asyncio.to_thread(client.files.download, file=video_file)

        if not video_bytes:
            raise InvalidResponseError("Video data is empty")

        logger.info(f"Successfully generated video, size: {len(video_bytes)} bytes")
        return video_bytes

    except Exception as exc:
        # Classify and re-raise the exception
        classified_exc = _classify_exception(exc)
        logger.error(f"Video generation failed: {classified_exc}")
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
        await generate_image(prompt="test", api_key=api_key, model=DEFAULT_IMAGE_MODEL)
        return True
    except (QuotaExceededError, APIError, ValueError):
        return False
    except Exception as e:
        logger.warning(f"Unexpected error validating API key: {e}")
        return False
