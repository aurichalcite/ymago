"""
Tests for ymago AI service integration.

This module tests the Google Generative AI integration, retry logic,
error handling, and response processing.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ymago.api import (
    APIError,
    InvalidResponseError,
    NetworkError,
    QuotaExceededError,
    _classify_exception,
    generate_image,
    generate_video,
    validate_api_key,
)


class TestExceptionClassification:
    """Test the _classify_exception function."""

    def test_classify_quota_exceeded_error(self):
        """Test classification of quota exceeded errors."""
        exc = Exception("API quota exceeded for this request")
        classified = _classify_exception(exc)
        assert isinstance(classified, QuotaExceededError)
        assert "API quota exceeded" in str(classified)

    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        exc = Exception("Rate limit exceeded, please try again later")
        classified = _classify_exception(exc)
        assert isinstance(classified, QuotaExceededError)

    def test_classify_network_error(self):
        """Test classification of network errors."""
        exc = Exception("Network connection failed")
        classified = _classify_exception(exc)
        assert isinstance(classified, NetworkError)
        assert "Network error" in str(classified)

    def test_classify_invalid_response_error(self):
        """Test classification of invalid response errors."""
        exc = Exception("Invalid JSON response from server")
        classified = _classify_exception(exc)
        assert isinstance(classified, InvalidResponseError)
        assert "Invalid API response" in str(classified)

    def test_classify_generic_api_error(self):
        """Test classification of generic API errors."""
        exc = Exception("Unknown API error occurred")
        classified = _classify_exception(exc)
        assert isinstance(classified, APIError)
        assert "API error" in str(classified)


class TestGenerateImage:
    """Test the generate_image async function."""


    @pytest.mark.asyncio
    async def test_generate_image_with_base64_data(self):
        """Test image generation with base64 encoded response."""
        image_data = b"test_image_data"
        base64_data = base64.b64encode(image_data).decode("utf-8")

        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Create mock response with base64 data
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "STOP"

            mock_content = MagicMock()
            mock_part = MagicMock()
            mock_inline_data = MagicMock()
            mock_inline_data.data = base64_data  # String instead of bytes

            mock_part.inline_data = mock_inline_data
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]

            mock_to_thread.return_value = mock_response

            result = await generate_image(prompt="Test prompt", api_key="test_api_key")

            assert result == image_data

    @pytest.mark.asyncio
    async def test_generate_image_empty_prompt_raises_error(self):
        """Test ValueError for empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await generate_image(prompt="", api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_generate_image_empty_api_key_raises_error(self):
        """Test ValueError for empty API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            await generate_image(prompt="Test prompt", api_key="")

    @pytest.mark.asyncio
    async def test_generate_image_missing_candidates_raises_error(self):
        """Test InvalidResponseError when response has no candidates."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Response with no candidates
            mock_response = MagicMock()
            mock_response.candidates = []

            mock_to_thread.return_value = mock_response

            with pytest.raises(APIError, match="API response contains no candidates"):
                await generate_image(prompt="Test prompt", api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_generate_image_safety_violation_raises_error(self):
        """Test APIError when content is blocked for safety."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Response with safety violation
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "SAFETY"

            mock_response.candidates = [mock_candidate]
            mock_to_thread.return_value = mock_response

            with pytest.raises(
                APIError, match="Content was blocked due to safety policies"
            ):
                await generate_image(prompt="Test prompt", api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_generate_image_missing_content_raises_error(self):
        """Test InvalidResponseError when candidate has no content."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Response with candidate but no content
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "STOP"
            mock_candidate.content = None

            mock_response.candidates = [mock_candidate]
            mock_to_thread.return_value = mock_response

            with pytest.raises(APIError, match="API response missing content"):
                await generate_image(prompt="Test prompt", api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_generate_image_no_image_data_raises_error(self):
        """Test InvalidResponseError when no image data is found."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Response with content but no image data
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "STOP"

            mock_content = MagicMock()
            mock_content.parts = []  # No parts with image data

            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]
            mock_to_thread.return_value = mock_response

            with pytest.raises(APIError, match="No image data found in API response"):
                await generate_image(prompt="Test prompt", api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_generate_image_retry_logic_success(self, sample_image_bytes):
        """Test retry logic succeeds after initial failures."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First 2 calls fail with network error, 3rd succeeds
            network_error = NetworkError("Connection failed")

            # Create successful response
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "STOP"

            mock_content = MagicMock()
            mock_part = MagicMock()
            mock_inline_data = MagicMock()
            mock_inline_data.data = sample_image_bytes

            mock_part.inline_data = mock_inline_data
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]

            mock_to_thread.side_effect = [network_error, network_error, mock_response]

            result = await generate_image(prompt="Test prompt", api_key="test_api_key")

            assert result == sample_image_bytes
            assert mock_to_thread.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_image_invalid_base64_raises_error(self):
        """Test InvalidResponseError for invalid base64 data."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Response with invalid base64 data
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "STOP"

            mock_content = MagicMock()
            mock_part = MagicMock()
            mock_inline_data = MagicMock()
            mock_inline_data.data = "invalid_base64_data!"  # Invalid base64

            mock_part.inline_data = mock_inline_data
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]

            mock_to_thread.return_value = mock_response

            with pytest.raises(
                InvalidResponseError, match="Failed to decode base64 image data"
            ):
                await generate_image(prompt="Test prompt", api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_generate_image_parameters_passed_through(self, sample_image_bytes):
        """Test that generation parameters are passed correctly via GenerationConfig."""
        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
            patch("ymago.api.types.GenerationConfig") as mock_gen_config_class,
        ):
            # Set up mock client and response
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Create mock response structure
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.finish_reason = "STOP"

            mock_content = MagicMock()
            mock_part = MagicMock()
            mock_inline_data = MagicMock()
            mock_inline_data.data = sample_image_bytes

            mock_part.inline_data = mock_inline_data
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_response.candidates = [mock_candidate]

            mock_to_thread.return_value = mock_response
            mock_gen_config_instance = MagicMock()
            mock_gen_config_class.return_value = mock_gen_config_instance

            # Test the function with multiple parameters
            await generate_image(
                prompt="Test prompt",
                api_key="test_api_key",
                model="test-model",
                seed=42,
                quality="high",
                aspect_ratio="16:9",
            )

            # Verify that GenerationConfig is created with the seed
            mock_gen_config_class.assert_called_once_with(seed=42)

            # Verify that generate_content is called with the GenerationConfig instance
            mock_to_thread.assert_called_once()
            call_args, call_kwargs = mock_to_thread.call_args
            assert "config" in call_kwargs
            assert call_kwargs["config"] == mock_gen_config_instance


class TestValidateApiKey:
    """Test the validate_api_key function."""

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self):
        """Test API key validation with successful generation."""
        with patch("ymago.api.generate_image", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = b"fake_image_data"

            result = await validate_api_key("valid_api_key")

            assert result is True
            mock_generate.assert_called_once_with(
                prompt="test",
                api_key="valid_api_key",
                model="gemini-2.5-flash-image-preview",
            )

    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self):
        """Test API key validation with failed generation."""
        with patch("ymago.api.generate_image", new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = APIError("Invalid API key")

            result = await validate_api_key("invalid_api_key")

            assert result is False


class TestGenerateVideo:
    """Test the generate_video function."""

    @pytest.mark.asyncio
    async def test_generate_video_success(self):
        """Test successful video generation."""
        from unittest.mock import Mock

        # Mock the entire video generation flow
        video_data = b"fake_video_data"

        with (
            patch("ymago.api.genai.Client") as mock_client_class,
            patch("ymago.api.asyncio.to_thread") as mock_to_thread,
        ):
            # Mock the operation that's returned from generate_videos
            mock_operation = Mock()
            mock_operation.name = "operations/test-operation-123"
            mock_operation.done = True

            # Mock the response structure
            mock_response = Mock()
            mock_generated_video = Mock()
            mock_video_file = Mock()
            mock_generated_video.video = mock_video_file
            mock_response.generated_videos = [mock_generated_video]
            mock_operation.response = mock_response

            # Mock client
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock asyncio.to_thread calls in order:
            # 1. client.models.generate_videos -> returns operation
            # 2. client.files.download -> returns video bytes
            mock_to_thread.side_effect = [mock_operation, video_data]

            result = await generate_video(
                prompt="A cat playing", api_key="test_key", model="veo-3.0-generate-001"
            )

            assert result == video_data
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_video_empty_prompt_raises_error(self):
        """Test that empty prompt raises an error."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await generate_video(
                prompt="", api_key="test_key", model="veo-3.0-generate-001"
            )

    @pytest.mark.asyncio
    async def test_generate_video_empty_api_key_raises_error(self):
        """Test that empty API key raises an error."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            await generate_video(
                prompt="A cat playing", api_key="", model="veo-3.0-generate-001"
            )
