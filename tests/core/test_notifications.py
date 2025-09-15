"""
Tests for webhook notification service.

This module tests the NotificationService and webhook payload models
with mocked HTTP requests to ensure they work correctly without making
actual network calls.
"""

import asyncio
from datetime import datetime

import aiohttp
import pytest
from aioresponses import aioresponses

from ymago.core.notifications import (
    NotificationService,
    WebhookPayload,
    create_failure_payload,
    create_success_payload,
)


class TestWebhookPayload:
    """Test webhook payload model."""

    def test_success_payload_creation(self):
        """Test creating a success webhook payload."""
        payload = WebhookPayload(
            job_id="test-job-123",
            job_status="success",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.2,
            file_size_bytes=1024,
            metadata={"model": "test-model"},
        )

        assert payload.job_id == "test-job-123"
        assert payload.job_status == "success"
        assert payload.output_url == "s3://bucket/file.jpg"
        assert payload.processing_time_seconds == 5.2
        assert payload.file_size_bytes == 1024
        assert payload.metadata["model"] == "test-model"
        assert isinstance(payload.timestamp, datetime)

    def test_failure_payload_creation(self):
        """Test creating a failure webhook payload."""
        payload = WebhookPayload(
            job_id="test-job-456",
            job_status="failure",
            error_message="Generation failed",
            processing_time_seconds=2.1,
            metadata={"error_code": "API_ERROR"},
        )

        assert payload.job_id == "test-job-456"
        assert payload.job_status == "failure"
        assert payload.output_url is None
        assert payload.error_message == "Generation failed"
        assert payload.processing_time_seconds == 2.1
        assert payload.metadata["error_code"] == "API_ERROR"

    def test_payload_json_serialization(self):
        """Test that payload can be serialized to JSON."""
        payload = WebhookPayload(
            job_id="test-job-789",
            job_status="success",
            output_url="gs://bucket/file.mp4",
            processing_time_seconds=10.5,
            file_size_bytes=2048,
        )

        json_str = payload.model_dump_json()
        assert "test-job-789" in json_str
        assert "success" in json_str
        assert "gs://bucket/file.mp4" in json_str

    def test_create_success_payload_helper(self):
        """Test create_success_payload helper function."""
        payload = create_success_payload(
            job_id="helper-test",
            output_url="s3://bucket/result.jpg",
            processing_time_seconds=3.7,
            file_size_bytes=512,
            metadata={"prompt": "test prompt"},
        )

        assert payload.job_id == "helper-test"
        assert payload.job_status == "success"
        assert payload.output_url == "s3://bucket/result.jpg"
        assert payload.processing_time_seconds == 3.7
        assert payload.file_size_bytes == 512
        assert payload.metadata["prompt"] == "test prompt"

    def test_create_failure_payload_helper(self):
        """Test create_failure_payload helper function."""
        payload = create_failure_payload(
            job_id="failure-test",
            error_message="API quota exceeded",
            processing_time_seconds=1.2,
            metadata={"retry_count": 3},
        )

        assert payload.job_id == "failure-test"
        assert payload.job_status == "failure"
        assert payload.error_message == "API quota exceeded"
        assert payload.processing_time_seconds == 1.2
        assert payload.metadata["retry_count"] == 3


class TestNotificationService:
    """Test notification service functionality."""

    def test_init_default_config(self):
        """Test notification service initialization with default config."""
        service = NotificationService()

        assert service.timeout_seconds == 30
        assert service.retry_attempts == 3
        assert service.retry_backoff_factor == 2.0

    def test_init_custom_config(self):
        """Test notification service initialization with custom config."""
        service = NotificationService(
            timeout_seconds=60, retry_attempts=5, retry_backoff_factor=1.5
        )

        assert service.timeout_seconds == 60
        assert service.retry_attempts == 5
        assert service.retry_backoff_factor == 1.5

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Test successful webhook notification delivery."""
        service = NotificationService()
        payload = create_success_payload(
            job_id="test-success",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
        )

        with aioresponses() as mock_responses:
            # Mock successful webhook response
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=200,
                payload={"status": "received"},
            )

            async with aiohttp.ClientSession() as session:
                # Should not raise any exceptions
                await service.send_notification(
                    session, "https://webhook.example.com/notify", payload
                )

    @pytest.mark.asyncio
    async def test_send_notification_http_error(self):
        """Test webhook notification with HTTP error (should not raise)."""
        service = NotificationService(
            retry_attempts=1
        )  # Reduce retries for faster test
        payload = create_success_payload(
            job_id="test-error",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
        )

        with aioresponses() as mock_responses:
            # Mock HTTP error response
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=500,
                payload={"error": "Internal server error"},
            )

            async with aiohttp.ClientSession() as session:
                # Should not raise any exceptions (fire-and-forget)
                await service.send_notification(
                    session, "https://webhook.example.com/notify", payload
                )

    @pytest.mark.asyncio
    async def test_send_notification_timeout(self):
        """Test webhook notification with timeout (should not raise)."""
        service = NotificationService(timeout_seconds=1, retry_attempts=1)
        payload = create_success_payload(
            job_id="test-timeout",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
        )

        with aioresponses():
            # Mock timeout by not adding any response
            pass

        async with aiohttp.ClientSession() as session:
            # Should not raise any exceptions (fire-and-forget)
            await service.send_notification(
                session, "https://webhook.example.com/notify", payload
            )

    @pytest.mark.asyncio
    async def test_send_notification_retry_logic(self):
        """Test webhook notification retry logic."""
        service = NotificationService(retry_attempts=2)
        payload = create_success_payload(
            job_id="test-retry",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
        )

        with aioresponses() as mock_responses:
            # First attempt fails
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=503,
                payload={"error": "Service unavailable"},
            )
            # Second attempt succeeds
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=200,
                payload={"status": "received"},
            )

            async with aiohttp.ClientSession() as session:
                # Should not raise any exceptions
                await service.send_notification(
                    session, "https://webhook.example.com/notify", payload
                )

    @pytest.mark.asyncio
    async def test_send_notification_async_task(self):
        """Test creating async task for webhook notification."""
        service = NotificationService()
        payload = create_success_payload(
            job_id="test-async",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
        )

        with aioresponses() as mock_responses:
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=200,
                payload={"status": "received"},
            )

            async with aiohttp.ClientSession() as session:
                # Create async task
                task = asyncio.create_task(
                    service.send_notification(
                        session, "https://webhook.example.com/notify", payload
                    )
                )

                assert isinstance(task, asyncio.Task)

                # Wait for task to complete
                await task

    @pytest.mark.asyncio
    async def test_webhook_request_headers(self):
        """Test that webhook requests include correct headers."""
        service = NotificationService()
        payload = create_success_payload(
            job_id="test-headers",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
        )

        with aioresponses() as mock_responses:
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=200,
                payload={"status": "received"},
            )

            async with aiohttp.ClientSession() as session:
                await service.send_notification(
                    session, "https://webhook.example.com/notify", payload
                )

                # Check that the request was made with correct headers
                requests = mock_responses.requests
                assert len(requests) == 1

                # Get the first request made to any URL
                first_request_list = list(requests.values())[0]
                assert len(first_request_list) == 1
                request_kwargs = first_request_list[0].kwargs

                assert request_kwargs["headers"]["Content-Type"] == "application/json"
                assert request_kwargs["headers"]["User-Agent"] == "ymago-webhook/1.0"

    @pytest.mark.asyncio
    async def test_webhook_payload_content(self):
        """Test that webhook request contains correct payload."""
        service = NotificationService()
        payload = create_success_payload(
            job_id="test-content",
            output_url="s3://bucket/file.jpg",
            processing_time_seconds=5.0,
            file_size_bytes=1024,
            metadata={"model": "test-model"},
        )

        with aioresponses() as mock_responses:
            mock_responses.post(
                "https://webhook.example.com/notify",
                status=200,
                payload={"status": "received"},
            )

            async with aiohttp.ClientSession() as session:
                await service.send_notification(
                    session, "https://webhook.example.com/notify", payload
                )

                # Check that the request was made with correct payload
                requests = mock_responses.requests
                assert len(requests) == 1

                # Get the first request made to any URL
                first_request_list = list(requests.values())[0]
                assert len(first_request_list) == 1
                request_data = first_request_list[0].kwargs.get("data")

                assert "test-content" in request_data
                assert "success" in request_data
                assert "s3://bucket/file.jpg" in request_data
                assert "test-model" in request_data
