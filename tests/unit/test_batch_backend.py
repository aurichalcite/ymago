"""
Unit tests for batch processing backend functionality.

This module tests the LocalExecutionBackend batch processing capabilities
including concurrency control, rate limiting, and checkpointing.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ymago.core.backends import LocalExecutionBackend, TokenBucketRateLimiter
from ymago.models import BatchResult, GenerationRequest, GenerationResult


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter implementation."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiting functionality."""
        # 60 requests per minute = 1 per second
        limiter = TokenBucketRateLimiter(60)

        # Consume burst tokens first (bucket_size is 60/10 = 6)
        for _ in range(int(limiter.bucket_size)):
            await limiter.acquire()

        start_time = time.time()

        # Next request should be rate limited
        await limiter.acquire()
        first_time = time.time() - start_time
        assert first_time >= 0.9  # Should wait ~1 second for new token

    @pytest.mark.asyncio
    async def test_rate_limiter_burst(self):
        """Test burst capability of rate limiter."""
        # 600 requests per minute with burst capability
        limiter = TokenBucketRateLimiter(600)

        start_time = time.time()

        # Should allow several requests immediately (burst)
        for _ in range(5):
            await limiter.acquire()

        burst_time = time.time() - start_time
        assert burst_time < 0.5  # Burst should be fast

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = TokenBucketRateLimiter(120)  # 120 requests per minute

        assert limiter.requests_per_minute == 120
        assert limiter.tokens_per_second == 2.0  # 120/60
        assert limiter.bucket_size >= 1
        assert limiter.tokens <= limiter.bucket_size


class TestLocalExecutionBackendBatch:
    """Test LocalExecutionBackend batch processing methods."""

    @pytest.fixture
    def backend(self):
        """Create a LocalExecutionBackend for testing."""
        return LocalExecutionBackend(max_concurrent_jobs=2)

    @pytest.fixture
    def sample_requests(self):
        """Create sample GenerationRequest objects."""
        return [
            GenerationRequest(
                id="req1", prompt="A beautiful sunset", output_filename="sunset1"
            ),
            GenerationRequest(
                id="req2", prompt="A mountain landscape", output_filename="mountain1"
            ),
            GenerationRequest(
                id="req3", prompt="A forest scene", output_filename="forest1"
            ),
        ]

    @pytest.mark.asyncio
    async def test_load_checkpoint_empty(self, backend):
        """Test loading checkpoint from non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "nonexistent.jsonl"

            completed = await backend._load_checkpoint(state_file)
            assert completed == set()

    @pytest.mark.asyncio
    async def test_load_checkpoint_with_data(self, backend):
        """Test loading checkpoint with existing data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "state.jsonl"

            # Create checkpoint file with some completed requests
            checkpoint_data = [
                {"request_id": "req1", "status": "success", "output_path": "/path1"},
                {"request_id": "req2", "status": "failure", "error_message": "Error"},
                {"request_id": "req3", "status": "success", "output_path": "/path3"},
            ]

            with open(state_file, "w") as f:
                for item in checkpoint_data:
                    f.write(json.dumps(item) + "\n")

            completed = await backend._load_checkpoint(state_file)
            assert completed == {"req1", "req3"}  # Only successful requests

    @pytest.mark.asyncio
    async def test_load_checkpoint_invalid_json(self, backend):
        """Test loading checkpoint with invalid JSON lines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "state.jsonl"

            # Create checkpoint file with invalid JSON
            with open(state_file, "w") as f:
                f.write('{"request_id": "req1", "status": "success"}\n')
                f.write("invalid json line\n")
                f.write('{"request_id": "req2", "status": "success"}\n')

            completed = await backend._load_checkpoint(state_file)
            assert completed == {"req1", "req2"}  # Should skip invalid line

    @pytest.mark.asyncio
    async def test_write_checkpoint(self, backend):
        """Test writing checkpoint data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "state.jsonl"

            result = BatchResult(
                request_id="test_req",
                status="success",
                output_path="/test/path",
                processing_time_seconds=1.5,
            )

            await backend._write_checkpoint(state_file, result)

            # Verify file was created and contains correct data
            assert state_file.exists()

            with open(state_file, "r") as f:
                line = f.readline().strip()
                data = json.loads(line)
                assert data["request_id"] == "test_req"
                assert data["status"] == "success"
                assert data["output_path"] == "/test/path"

    @pytest.mark.asyncio
    async def test_process_request_with_retry_success(self, backend):
        """Test successful request processing with retry logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "state.jsonl"

            request = GenerationRequest(id="test_req", prompt="Test prompt")

            # Mock the dependencies
            mock_result = GenerationResult(
                local_path=Path("/test/output.png"),
                job=request.to_generation_job(),
                file_size_bytes=1024,
                metadata={"test": "data"},
            )

            with patch("ymago.config.load_config") as mock_config:
                with patch(
                    "ymago.core.generation.process_generation_job"
                ) as mock_process:
                    mock_config.return_value = MagicMock()
                    mock_process.return_value = mock_result

                    result = await backend._process_request_with_retry(
                        request, output_dir, state_file
                    )

                    assert result.request_id == "test_req"
                    assert result.status == "success"
                    assert result.output_path == str(mock_result.local_path)
                    assert result.file_size_bytes == 1024

    @pytest.mark.asyncio
    async def test_process_request_with_retry_failure(self, backend):
        """Test request processing failure handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "state.jsonl"

            request = GenerationRequest(id="test_req", prompt="Test prompt")

            # Mock the dependencies to raise an exception
            with patch("ymago.config.load_config") as mock_config:
                with patch(
                    "ymago.core.generation.process_generation_job"
                ) as mock_process:
                    mock_config.return_value = MagicMock()
                    mock_process.side_effect = Exception("Test error")

                    result = await backend._process_request_with_retry(
                        request, output_dir, state_file
                    )

                    assert result.request_id == "test_req"
                    assert result.status == "failure"
                    assert "Test error" in result.error_message
                    assert result.processing_time_seconds > 0

    @pytest.mark.asyncio
    async def test_process_batch_basic(self, backend, sample_requests):
        """Test basic batch processing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Create async generator from sample requests
            async def request_generator():
                for req in sample_requests:
                    yield req

            # Mock the request processing to return success
            with patch.object(backend, "_process_request_with_retry") as mock_process:
                mock_process.side_effect = [
                    BatchResult(
                        request_id="req1", status="success", output_path="/path1"
                    ),
                    BatchResult(
                        request_id="req2", status="success", output_path="/path2"
                    ),
                    BatchResult(
                        request_id="req3", status="failure", error_message="Error"
                    ),
                ]

                summary = await backend.process_batch(
                    requests=request_generator(),
                    output_dir=output_dir,
                    concurrency=2,
                    rate_limit=60,
                    resume=False,
                )

                assert summary.total_requests == 3
                assert summary.successful == 2
                assert summary.failed == 1
                assert summary.skipped == 0
                assert summary.processing_time_seconds > 0
                assert summary.throughput_requests_per_minute > 0

    @pytest.mark.asyncio
    async def test_process_batch_with_resume(self, backend, sample_requests):
        """Test batch processing with resume functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "_batch_state.jsonl"

            # Create existing checkpoint with one completed request
            checkpoint_data = {
                "request_id": "req1",
                "status": "success",
                "output_path": "/existing/path1",
            }

            with open(state_file, "w") as f:
                f.write(json.dumps(checkpoint_data) + "\n")

            # Create async generator from sample requests
            async def request_generator():
                for req in sample_requests:
                    yield req

            # Mock the request processing
            with patch.object(backend, "_process_request_with_retry") as mock_process:
                mock_process.side_effect = [
                    BatchResult(
                        request_id="req2", status="success", output_path="/path2"
                    ),
                    BatchResult(
                        request_id="req3", status="success", output_path="/path3"
                    ),
                ]

                summary = await backend.process_batch(
                    requests=request_generator(),
                    output_dir=output_dir,
                    concurrency=2,
                    rate_limit=60,
                    resume=True,
                )

                assert summary.total_requests == 3
                assert summary.successful == 2  # req2 and req3 processed
                assert summary.failed == 0
                assert summary.skipped == 1  # req1 was skipped (already completed)

    @pytest.mark.asyncio
    async def test_process_batch_concurrency_control(self, backend):
        """Test that concurrency is properly controlled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Create requests that will help us test concurrency
            requests = [
                GenerationRequest(id=f"req{i}", prompt=f"Prompt {i}") for i in range(5)
            ]

            async def request_generator():
                for req in requests:
                    yield req

            # Track concurrent executions
            concurrent_count = 0
            max_concurrent = 0

            async def mock_process_with_delay(request, output_dir, state_file):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

                # Simulate processing time
                await asyncio.sleep(0.1)

                concurrent_count -= 1
                return BatchResult(
                    request_id=request.id,
                    status="success",
                    output_path=f"/path/{request.id}",
                )

            with patch.object(
                backend,
                "_process_request_with_retry",
                side_effect=mock_process_with_delay,
            ):
                await backend.process_batch(
                    requests=request_generator(),
                    output_dir=output_dir,
                    concurrency=2,  # Limit to 2 concurrent
                    rate_limit=300,  # High rate limit to not interfere
                    resume=False,
                )

                # Should never exceed the concurrency limit
                assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_process_batch_empty_requests(self, backend):
        """Test batch processing with no requests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            async def empty_generator():
                return
                yield  # This line is never reached

            summary = await backend.process_batch(
                requests=empty_generator(),
                output_dir=output_dir,
                concurrency=2,
                rate_limit=60,
                resume=False,
            )

            assert summary.total_requests == 0
            assert summary.successful == 0
            assert summary.failed == 0
            assert summary.skipped == 0
