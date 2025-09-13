"""
Tests for execution backend implementations.

This module tests the execution backend abstraction and LocalExecutionBackend
implementation, including job submission, concurrency control, and error handling.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ymago.config import Auth, Defaults, Settings
from ymago.core.backends import ExecutionBackend, LocalExecutionBackend
from ymago.models import GenerationJob, GenerationResult


class TestExecutionBackendInterface:
    """Test the abstract ExecutionBackend interface."""

    def test_execution_backend_is_abstract(self):
        """Test that ExecutionBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ExecutionBackend()  # type: ignore

    def test_execution_backend_requires_submit_method(self):
        """Test that subclasses must implement submit method."""

        class IncompleteBackend(ExecutionBackend):
            async def get_status(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBackend()  # type: ignore

    def test_execution_backend_requires_get_status_method(self):
        """Test that subclasses must implement get_status method."""

        class IncompleteBackend(ExecutionBackend):
            async def submit(self, jobs: list[GenerationJob]) -> list[GenerationResult]:
                return []

        # This should fail since get_status is not implemented
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBackend()


class TestLocalExecutionBackend:
    """Test the LocalExecutionBackend implementation."""

    @pytest.fixture
    def backend(self):
        """Create a LocalExecutionBackend instance."""
        return LocalExecutionBackend(max_concurrent_jobs=2)

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return Settings(
            auth=Auth(google_api_key="test_key"),
            defaults=Defaults(
                output_path="/tmp/test",
                image_model="test-model",
                video_model="test-video-model",
            ),
        )

    @pytest.fixture
    def sample_jobs(self):
        """Create sample generation jobs."""
        return [
            GenerationJob(prompt="Test prompt 1", output_filename="test1"),
            GenerationJob(prompt="Test prompt 2", output_filename="test2"),
            GenerationJob(prompt="Test prompt 3", output_filename="test3"),
        ]

    def test_backend_initialization(self, backend):
        """Test LocalExecutionBackend initialization."""
        assert backend.max_concurrent_jobs == 2
        assert backend._active_jobs == 0
        assert backend._total_jobs_executed == 0

    @pytest.mark.asyncio
    async def test_submit_empty_jobs_list(self, backend):
        """Test submitting an empty jobs list raises ValueError."""
        with pytest.raises(ValueError, match="Jobs list cannot be empty"):
            await backend.submit([])

    @pytest.mark.asyncio
    async def test_submit_single_job_success(self, backend, mock_config, sample_jobs):
        """Test successful submission of a single job."""
        job = sample_jobs[0]
        mock_result = GenerationResult(
            local_path=Path("/tmp/test/test1.png"),
            job=job,
            metadata={"test": "metadata"},
        )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job"
            ) as mock_process_job:
                mock_load_config.return_value = mock_config
                mock_process_job.return_value = mock_result

                results = await backend.submit([job])

                assert len(results) == 1
                assert results[0] == mock_result
                assert results[0].generation_time_seconds is not None
                assert results[0].get_metadata("execution_backend") == "local"
                mock_process_job.assert_called_once_with(job, mock_config)

    @pytest.mark.asyncio
    async def test_submit_multiple_jobs_with_concurrency(
        self, backend, mock_config, sample_jobs
    ):
        """Test submission of multiple jobs respects concurrency limits."""
        mock_results = [
            GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )
            for job in sample_jobs
        ]

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_process_job(job, config):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate processing time
            concurrent_count -= 1
            index = sample_jobs.index(job)
            return mock_results[index]

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                results = await backend.submit(sample_jobs)

                assert len(results) == 3
                assert all(
                    r.get_metadata("execution_backend") == "local" for r in results
                )
                # Max concurrent should not exceed the limit
                assert max_concurrent <= backend.max_concurrent_jobs

    @pytest.mark.asyncio
    async def test_submit_with_job_failure(self, backend, mock_config, sample_jobs):
        """Test handling of job failures during submission."""

        async def mock_process_job(job, config):
            if job == sample_jobs[1]:  # Second job fails
                raise RuntimeError("Processing failed")
            return GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                with pytest.raises(RuntimeError, match="Processing failed"):
                    await backend.submit(sample_jobs)

    @pytest.mark.asyncio
    async def test_submit_with_unexpected_result_type(
        self, backend, mock_config, sample_jobs
    ):
        """Test handling of unexpected result types from job processing."""

        async def mock_process_job(job, config):
            if job == sample_jobs[0]:
                return "unexpected_string"  # Wrong type
            return GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                with pytest.raises(
                    RuntimeError,
                    match="Job execution failed.*Job 0 returned unexpected result type str",
                ):
                    await backend.submit(sample_jobs)

    @pytest.mark.asyncio
    async def test_get_status(self, backend):
        """Test getting backend status."""
        status = await backend.get_status()

        assert status["backend_type"] == "local"
        assert status["max_concurrent_jobs"] == 2
        assert status["active_jobs"] == 0
        assert status["total_jobs_executed"] == 0
        assert status["available_slots"] == 2

    @pytest.mark.asyncio
    async def test_get_status_during_execution(self, backend, mock_config, sample_jobs):
        """Test getting status while jobs are executing."""
        status_during_execution = None

        async def mock_process_job(job, config):
            nonlocal status_during_execution
            if status_during_execution is None:
                status_during_execution = await backend.get_status()
            await asyncio.sleep(0.05)
            return GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                await backend.submit(sample_jobs[:2])

                # Check status captured during execution
                assert status_during_execution is not None
                assert status_during_execution["active_jobs"] > 0
                assert status_during_execution["available_slots"] < 2

                # Check final status
                final_status = await backend.get_status()
                assert final_status["active_jobs"] == 0
                assert final_status["total_jobs_executed"] == 2

    @pytest.mark.asyncio
    async def test_execution_metadata_added(self, backend, mock_config):
        """Test that execution metadata is properly added to results."""
        job = GenerationJob(prompt="Test prompt", output_filename="test")
        mock_result = GenerationResult(
            local_path=Path("/tmp/test/test.png"),
            job=job,
            metadata={},
        )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job"
            ) as mock_process_job:
                mock_load_config.return_value = mock_config
                mock_process_job.return_value = mock_result

                results = await backend.submit([job])

                result = results[0]
                assert result.generation_time_seconds is not None
                assert result.generation_time_seconds >= 0
                assert result.get_metadata("execution_backend") == "local"
                assert result.get_metadata("execution_time") is not None

    @pytest.mark.asyncio
    async def test_concurrent_submission_tracking(self, backend, mock_config):
        """Test that job execution count is tracked correctly."""
        jobs = [
            GenerationJob(prompt=f"Test prompt {i}", output_filename=f"test{i}")
            for i in range(5)
        ]

        async def mock_process_job(job, config):
            return GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                # Submit first batch
                await backend.submit(jobs[:3])
                status1 = await backend.get_status()
                assert status1["total_jobs_executed"] == 3

                # Submit second batch
                await backend.submit(jobs[3:])
                status2 = await backend.get_status()
                assert status2["total_jobs_executed"] == 5

    @pytest.mark.asyncio
    async def test_backend_handles_asyncio_cancellation(
        self, backend, mock_config, sample_jobs
    ):
        """Test that backend handles asyncio cancellation gracefully."""
        cancel_event = asyncio.Event()

        async def mock_process_job(job, config):
            cancel_event.set()
            await asyncio.sleep(10)  # Long running task
            return GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                task = asyncio.create_task(backend.submit(sample_jobs))
                await cancel_event.wait()
                task.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await task

    @pytest.mark.asyncio
    async def test_backend_respects_semaphore(self, backend, mock_config):
        """Test that backend properly uses semaphore for concurrency control."""
        execution_order = []

        async def mock_process_job(job, config):
            execution_order.append(f"start_{job.output_filename}")
            await asyncio.sleep(0.1)
            execution_order.append(f"end_{job.output_filename}")
            return GenerationResult(
                local_path=Path(f"/tmp/test/{job.output_filename}.png"),
                job=job,
                metadata={},
            )

        jobs = [
            GenerationJob(prompt=f"Test {i}", output_filename=f"test{i}")
            for i in range(4)
        ]

        with patch("ymago.config.load_config") as mock_load_config:
            with patch(
                "ymago.core.generation.process_generation_job",
                side_effect=mock_process_job,
            ):
                mock_load_config.return_value = mock_config

                await backend.submit(jobs)

                # With max_concurrent_jobs=2, we should see at most 2 jobs
                # starting before any complete
                starts_before_first_end = []
                for event in execution_order:
                    if event.startswith("end_"):
                        break
                    if event.startswith("start_"):
                        starts_before_first_end.append(event)

                assert len(starts_before_first_end) <= backend.max_concurrent_jobs
