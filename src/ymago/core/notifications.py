"""
Webhook notification service for ymago package.

This module provides a notification service for sending webhook notifications
upon job completion. The service is designed to be fire-and-forget, ensuring
that slow or failing webhook endpoints do not block the main generation pipeline.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import aiohttp
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


class WebhookPayload(BaseModel):
    """
    Standardized webhook payload for job completion notifications.

    This model defines the structure of webhook notifications sent when
    generation jobs complete, providing consistent information to external
    systems about job status and results.
    """

    job_id: str = Field(..., description="Unique identifier for the generation job")

    job_status: Literal["success", "failure"] = Field(
        ..., description="Final status of the generation job"
    )

    output_url: Optional[str] = Field(
        default=None, description="URL where the generated content can be accessed"
    )

    error_message: Optional[str] = Field(
        default=None, description="Error details if the job failed"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional job-specific information"
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the job completed (UTC)",
    )

    processing_time_seconds: Optional[float] = Field(
        default=None, description="Total time taken to process the job", ge=0.0
    )

    file_size_bytes: Optional[int] = Field(
        default=None, description="Size of the generated file in bytes", ge=0
    )


class NotificationService:
    """
    Webhook notification service for sending job completion notifications.

    This service provides fire-and-forget webhook delivery with built-in
    retry logic and proper error handling. It's designed to not block the
    main generation pipeline even if webhook endpoints are slow or failing.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        retry_attempts: int = 3,
        retry_backoff_factor: float = 2.0,
    ):
        """
        Initialize the notification service.

        Args:
            timeout_seconds: HTTP request timeout
            retry_attempts: Number of retry attempts for failed requests
            retry_backoff_factor: Exponential backoff factor for retries
        """
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_factor = retry_backoff_factor

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(3),  # Will be overridden by instance config
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _send_webhook_request(
        self, session: aiohttp.ClientSession, webhook_url: str, payload: WebhookPayload
    ) -> None:
        """
        Send a single webhook request with retry logic.

        Args:
            session: aiohttp client session
            webhook_url: Target webhook URL
            payload: Webhook payload to send

        Raises:
            aiohttp.ClientError: For HTTP-related errors
            asyncio.TimeoutError: For request timeouts
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ymago-webhook/1.0",
        }

        # Convert payload to JSON
        payload_json = payload.model_dump_json()

        logger.debug(f"Sending webhook to {webhook_url}")

        async with session.post(
            webhook_url,
            data=payload_json,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
        ) as response:
            # Log response status
            if response.status >= 400:
                response_text = await response.text()
                logger.warning(
                    f"Webhook returned {response.status}: {response_text[:200]}"
                )
                response.raise_for_status()
            else:
                logger.debug(f"Webhook delivered successfully: {response.status}")

    async def send_notification(
        self, session: aiohttp.ClientSession, webhook_url: str, payload: WebhookPayload
    ) -> None:
        """
        Send a webhook notification (fire-and-forget).

        This method is designed to be called with asyncio.create_task() to
        ensure it doesn't block the main generation pipeline.

        Args:
            session: aiohttp client session
            webhook_url: Target webhook URL
            payload: Webhook payload to send
        """
        try:
            # Update retry configuration based on instance settings
            self._send_webhook_request.retry.stop = stop_after_attempt(  # type: ignore[attr-defined]
                self.retry_attempts
            )
            self._send_webhook_request.retry.wait = wait_random_exponential(  # type: ignore[attr-defined]
                multiplier=self.retry_backoff_factor, max=30
            )

            await self._send_webhook_request(session, webhook_url, payload)
            logger.info(f"Webhook notification sent successfully to {webhook_url}")

        except Exception as e:
            # Log error but don't raise - this is fire-and-forget
            logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")


def create_success_payload(
    job_id: str,
    output_url: str,
    processing_time_seconds: float,
    file_size_bytes: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> WebhookPayload:
    """
    Create a webhook payload for successful job completion.

    Args:
        job_id: Unique job identifier
        output_url: URL where the generated content can be accessed
        processing_time_seconds: Time taken to process the job
        file_size_bytes: Size of the generated file
        metadata: Additional job metadata

    Returns:
        WebhookPayload: Configured payload for success notification
    """
    return WebhookPayload(
        job_id=job_id,
        job_status="success",
        output_url=output_url,
        processing_time_seconds=processing_time_seconds,
        file_size_bytes=file_size_bytes,
        metadata=metadata or {},
    )


def create_failure_payload(
    job_id: str,
    error_message: str,
    processing_time_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WebhookPayload:
    """
    Create a webhook payload for failed job completion.

    Args:
        job_id: Unique job identifier
        error_message: Description of the error that occurred
        processing_time_seconds: Time taken before failure (optional)
        metadata: Additional job metadata

    Returns:
        WebhookPayload: Configured payload for failure notification
    """
    return WebhookPayload(
        job_id=job_id,
        job_status="failure",
        error_message=error_message,
        processing_time_seconds=processing_time_seconds,
        metadata=metadata or {},
    )
