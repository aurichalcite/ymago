"""
Resilience tests for batch processing functionality.

This module tests the batch processing system's ability to handle
interruptions, resume from checkpoints, and recover from various failure scenarios.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ymago.core.backends import LocalExecutionBackend
from ymago.models import BatchResult, GenerationRequest, GenerationResult


class TestBatchResilience:
    """Test batch processing resilience and recovery capabilities."""

    @pytest.fixture
    def backend(self):
        """Create a LocalExecutionBackend for testing."""
        return LocalExecutionBackend(max_concurrent_jobs=2)

    @pytest.fixture
    def sample_requests(self):
        """Create sample GenerationRequest objects."""
        return [
            GenerationRequest(id="req1", prompt="Prompt 1"),
            GenerationRequest(id="req2", prompt="Prompt 2"),
            GenerationRequest(id="req3", prompt="Prompt 3"),
            GenerationRequest(id="req4", prompt="Prompt 4"),
            GenerationRequest(id="req5", prompt="Prompt 5"),
        ]

    @pytest.mark.asyncio
    async def test_resume_from_partial_completion(self, backend, sample_requests):
        """Test resuming batch processing from partial completion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "_batch_state.jsonl"

            # Simulate partial completion by creating checkpoint file
            partial_results = [
                {"request_id": "req1", "status": "success", "output_path": "/path1"},
                {
                    "request_id": "req2",
                    "status": "failure",
                    "error_message": "Network error",
                },
                {"request_id": "req3", "status": "success", "output_path": "/path3"},
            ]

            with open(state_file, "w") as f:
                for result in partial_results:
                    f.write(json.dumps(result) + "\n")

            async def request_generator():
                for req in sample_requests:
                    yield req

            # Mock processing for remaining requests
            with patch.object(backend, "_process_request_with_retry") as mock_process:
                mock_process.side_effect = [
                    BatchResult(
                        request_id="req2", status="success", output_path="/path2_retry"
                    ),  # req2 is retried since it failed previously
                    BatchResult(
                        request_id="req4", status="success", output_path="/path4"
                    ),
                    BatchResult(
                        request_id="req5", status="success", output_path="/path5"
                    ),
                ]

                summary = await backend.process_batch(
                    requests=request_generator(),
                    output_dir=output_dir,
                    concurrency=2,
                    rate_limit=60,
                    resume=True,
                )

                # Should process req2 (retry), req4, and req5
                # (req1 and req3 were successful and skipped)
                assert summary.total_requests == 5
                assert (
                    summary.successful == 3
                )  # req2 (retry), req4, and req5 newly processed
                assert summary.failed == 0  # No new failures
                assert (
                    summary.skipped == 2
                )  # req1 and req3 were skipped (already successful)

                # Verify only remaining requests were processed
                assert mock_process.call_count == 3  # req2, req4, req5
                processed_ids = {
                    call.args[0].id for call in mock_process.call_args_list
                }
                assert processed_ids == {"req2", "req4", "req5"}

    @pytest.mark.asyncio
    async def test_corrupted_checkpoint_recovery(self, backend, sample_requests):
        """Test recovery from corrupted checkpoint file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "_batch_state.jsonl"

            # Create checkpoint file with mixed valid and invalid JSON
            with open(state_file, "w") as f:
                f.write(
                    '{"request_id": "req1", "status": "success", '
                    '"output_path": "/path1"}\n'
                )
                f.write("invalid json line that should be skipped\n")
                f.write(
                    '{"request_id": "req2", "status": "success"}\n'
                )  # Missing output_path
                f.write(
                    '{"request_id": "req3", "status": "success", '
                    '"output_path": "/path3"}\n'
                )

            async def request_generator():
                for req in sample_requests:
                    yield req

            with patch.object(backend, "_process_request_with_retry") as mock_process:
                mock_process.side_effect = [
                    BatchResult(
                        request_id="req2", status="success", output_path="/path2"
                    ),
                    BatchResult(
                        request_id="req4", status="success", output_path="/path4"
                    ),
                    BatchResult(
                        request_id="req5", status="success", output_path="/path5"
                    ),
                ]

                summary = await backend.process_batch(
                    requests=request_generator(),
                    output_dir=output_dir,
                    concurrency=2,
                    rate_limit=60,
                    resume=True,
                )

                # Should skip req1 and req3 (valid successful entries)
                # Should process req2 (invalid entry), req4, and req5
                assert summary.total_requests == 5
                assert summary.successful == 3  # req2, req4, req5 newly processed
                assert summary.skipped == 2  # req1, req3 skipped

    @pytest.mark.asyncio
    async def test_network_failure_retry(self, backend):
        """Test retry logic for network failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "state.jsonl"

            request = GenerationRequest(id="test_req", prompt="Test prompt")

            # Mock network failures followed by success
            call_count = 0

            async def mock_process_job(job, config):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # First two calls fail
                    raise ConnectionError("Network timeout")
                else:  # Third call succeeds
                    return GenerationResult(
                        local_path=Path("/test/output.png"),
                        job=job,
                        file_size_bytes=1024,
                    )

            with patch("ymago.core.backends.load_config") as mock_config:
                with patch(
                    "ymago.core.backends.process_generation_job",
                    side_effect=mock_process_job,
                ):
                    mock_config.return_value = MagicMock()

                    result = await backend._process_request_with_retry(
                        request, output_dir, state_file
                    )

                    assert result.status == "success"
                    assert call_count == 3  # Should have retried twice

    @pytest.mark.asyncio
    async def test_permanent_failure_handling(self, backend):
        """Test handling of permanent failures that exceed retry limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "state.jsonl"

            request = GenerationRequest(id="test_req", prompt="Test prompt")

            # Mock permanent failure
            with patch("ymago.core.backends.load_config") as mock_config:
                with patch(
                    "ymago.core.backends.process_generation_job"
                ) as mock_process:
                    mock_config.return_value = MagicMock()
                    mock_process.side_effect = ConnectionError(
                        "Permanent network failure"
                    )

                    result = await backend._process_request_with_retry(
                        request, output_dir, state_file
                    )

                    assert result.status == "failure"
                    assert "Permanent network failure" in result.error_message
                    assert result.processing_time_seconds > 0

    @pytest.mark.asyncio
    async def test_checkpoint_atomicity(self, backend):
        """Test that checkpoint writes are atomic and don't corrupt the file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            state_file = output_dir / "state.jsonl"

            # Create multiple results to write concurrently
            results = [
                BatchResult(
                    request_id=f"req{i}", status="success", output_path=f"/path{i}"
                )
                for i in range(10)
            ]

            # Write all results concurrently to test atomicity
            tasks = [
                backend._write_checkpoint(state_file, result) for result in results
            ]

            await asyncio.gather(*tasks)

            # Verify all results were written correctly
            written_results = []
            with open(state_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        written_results.append(json.loads(line))

            assert len(written_results) == 10
            written_ids = {result["request_id"] for result in written_results}
            expected_ids = {f"req{i}" for i in range(10)}
            assert written_ids == expected_ids

    @pytest.mark.asyncio
    async def test_rate_limiter_under_load(self, backend):
        """Test rate limiter behavior under high load."""
        from ymago.core.backends import TokenBucketRateLimiter

        # Create rate limiter for 60 requests per minute (1 per second)
        limiter = TokenBucketRateLimiter(60)

        # Record timing of multiple requests
        start_time = time.time()
        request_times = []

        # Make 5 requests rapidly
        for _ in range(5):
            await limiter.acquire()
            request_times.append(time.time() - start_time)

        # First few requests should be fast (burst), later ones should be rate limited
        assert request_times[0] < 0.1  # First request immediate
        assert request_times[-1] >= 3.0  # Last request should be delayed significantly

    @pytest.mark.asyncio
    async def test_concurrent_processing_isolation(self, backend, sample_requests):
        """Test that concurrent request processing doesn't interfere with each other."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Track which requests are processed concurrently
            active_requests = set()
            max_concurrent = 0
            processing_log = []

            async def mock_process_with_tracking(request, output_dir, state_file):
                nonlocal max_concurrent

                active_requests.add(request.id)
                max_concurrent = max(max_concurrent, len(active_requests))
                processing_log.append(f"Started {request.id}")

                # Simulate variable processing time
                await asyncio.sleep(0.1 + (hash(request.id) % 3) * 0.05)

                active_requests.remove(request.id)
                processing_log.append(f"Finished {request.id}")

                return BatchResult(
                    request_id=request.id,
                    status="success",
                    output_path=f"/path/{request.id}",
                )

            async def request_generator():
                for req in sample_requests:
                    yield req

            with patch.object(
                backend,
                "_process_request_with_retry",
                side_effect=mock_process_with_tracking,
            ):
                summary = await backend.process_batch(
                    requests=request_generator(),
                    output_dir=output_dir,
                    concurrency=3,  # Allow up to 3 concurrent
                    rate_limit=300,  # High rate limit to not interfere
                    resume=False,
                )

                # Verify concurrency was respected
                assert max_concurrent <= 3
                assert summary.total_requests == 5
                assert summary.successful == 5

                # Verify all requests were processed
                started_requests = {
                    log.split()[1] for log in processing_log if "Started" in log
                }
                finished_requests = {
                    log.split()[1] for log in processing_log if "Finished" in log
                }
                assert started_requests == finished_requests
                assert len(started_requests) == 5

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_batch(self, backend):
        """Test memory efficiency with a large number of requests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Create a large number of requests
            num_requests = 1000

            async def large_request_generator():
                for i in range(num_requests):
                    yield GenerationRequest(id=f"req{i:04d}", prompt=f"Prompt {i}")

            # Mock fast processing to test memory usage
            with patch.object(backend, "_process_request_with_retry") as mock_process:
                mock_process.side_effect = lambda req, *args: BatchResult(
                    request_id=req.id, status="success", output_path=f"/path/{req.id}"
                )

                summary = await backend.process_batch(
                    requests=large_request_generator(),
                    output_dir=output_dir,
                    concurrency=10,
                    rate_limit=6000,  # High rate limit for speed
                    resume=False,
                )

                assert summary.total_requests == num_requests
                assert summary.successful == num_requests

                # Verify checkpoint file contains all results
                state_file = output_dir / "_batch_state.jsonl"
                assert state_file.exists()

                line_count = 0
                with open(state_file, "r") as f:
                    for line in f:
                        if line.strip():
                            line_count += 1

                assert line_count == num_requests
