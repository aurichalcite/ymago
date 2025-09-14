"""
Execution backend abstraction for ymago package.

This module provides abstract base classes for job execution and concrete
implementations for local execution. Future implementations can extend this
to support distributed execution, cloud functions, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, List, Set

import aiofiles
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import (
    BatchResult,
    BatchSummary,
    GenerationJob,
    GenerationRequest,
    GenerationResult,
)

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


class ExecutionBackend(ABC):
    """
    Abstract base class for job execution backends.

    This interface defines the contract for executing generation jobs across
    different execution environments. Implementations should handle the specifics
    of each execution model while providing a consistent async interface.
    """

    @abstractmethod
    async def submit(self, jobs: List[GenerationJob]) -> List[GenerationResult]:
        """
        Submit generation jobs for execution.

        Args:
            jobs: List of generation jobs to execute

        Returns:
            List[GenerationResult]: Results for each job in the same order

        Raises:
            ExecutionError: For execution-specific errors
            ValueError: For invalid job parameters
        """
        pass

    @abstractmethod
    async def process_batch(
        self,
        requests: AsyncGenerator[GenerationRequest, None],
        output_dir: Path,
        concurrency: int,
        rate_limit: int,
        resume: bool = False,
    ) -> BatchSummary:
        """
        Process a batch of generation requests with resilient execution.

        Args:
            requests: Async generator of generation requests
            output_dir: Directory for output files and state management
            concurrency: Maximum number of concurrent requests
            rate_limit: Maximum requests per minute
            resume: Whether to resume from existing checkpoint

        Returns:
            BatchSummary: Complete summary of batch processing results

        Raises:
            ExecutionError: For execution-specific errors
            ValueError: For invalid parameters
        """
        pass

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the execution backend.

        Returns:
            dict: Status information including capacity, active jobs, etc.
        """
        pass


class LocalExecutionBackend(ExecutionBackend):
    """
    Local execution backend implementation.

    This implementation executes jobs locally using asyncio for concurrency.
    It's designed to handle multiple jobs concurrently while respecting system
    resources and API rate limits.

    Future distributed implementations (KubernetesBackend, AWSLambdaBackend, etc.)
    can follow this same pattern while implementing distributed execution logic.
    """

    def __init__(self, max_concurrent_jobs: int = 3):
        """
        Initialize the local execution backend.

        Args:
            max_concurrent_jobs: Maximum number of jobs to execute concurrently
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self._active_jobs = 0
        self._total_jobs_executed = 0
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)

    async def submit(self, jobs: List[GenerationJob]) -> List[GenerationResult]:
        """
        Execute generation jobs locally with controlled concurrency.

        Args:
            jobs: List of generation jobs to execute

        Returns:
            List[GenerationResult]: Results for each job in the same order

        Raises:
            ValueError: If jobs list is empty
            ExecutionError: For execution failures
        """
        if not jobs:
            raise ValueError("Jobs list cannot be empty")

        # Import here to avoid circular imports
        from ..config import load_config
        from ..core.generation import process_generation_job

        # Load configuration once for all jobs
        config: Settings = await load_config()

        # Create tasks for concurrent execution
        tasks = []
        for job in jobs:
            task = self._execute_single_job(job, config, process_generation_job)
            tasks.append(task)

        # Execute all jobs concurrently with controlled concurrency
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            final_results: list[GenerationResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # For now, re-raise the exception
                    # In a production system, you might want to return error results
                    raise result
                elif isinstance(result, GenerationResult):
                    final_results.append(result)
                else:
                    # Handle unexpected result types - this should not happen in normal
                    # operation but provides better error reporting
                    raise TypeError(
                        f"Job {i} returned unexpected result type "
                        f"{type(result).__name__}: {result}"
                    )

            return final_results

        except Exception as e:
            raise RuntimeError(f"Job execution failed: {e}") from e

    async def _execute_single_job(
        self,
        job: GenerationJob,
        config: Settings,
        process_func: Callable[[GenerationJob, Settings], Awaitable[GenerationResult]],
    ) -> GenerationResult:
        """
        Execute a single job with semaphore-controlled concurrency.

        Args:
            job: The generation job to execute
            config: Configuration settings
            process_func: The function to process the job

        Returns:
            GenerationResult: The result of the job execution
        """
        async with self._semaphore:
            self._active_jobs += 1
            start_time = time.time()

            try:
                result = await process_func(job, config)

                # Only add metadata if result is a GenerationResult
                if isinstance(result, GenerationResult):
                    # Add execution metadata
                    execution_time = time.time() - start_time
                    result.generation_time_seconds = execution_time
                    result.add_metadata("execution_backend", "local")
                    result.add_metadata("execution_time", execution_time)

                return result

            finally:
                self._active_jobs -= 1
                self._total_jobs_executed += 1

    async def get_status(self) -> dict[str, Any]:
        """Get the current status of the local execution backend."""
        return {
            "backend_type": "local",
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "active_jobs": self._active_jobs,
            "total_jobs_executed": self._total_jobs_executed,
            "available_slots": self.max_concurrent_jobs - self._active_jobs,
        }

    async def process_batch(
        self,
        requests: AsyncGenerator[GenerationRequest, None],
        output_dir: Path,
        concurrency: int,
        rate_limit: int,
        resume: bool = False,
    ) -> BatchSummary:
        """
        Process a batch of generation requests with resilient execution.

        This implementation provides:
        - Controlled concurrency using semaphores
        - Rate limiting with token bucket algorithm
        - Atomic checkpointing for resume capability
        - Comprehensive error handling and retry logic
        - Progress tracking and detailed logging

        Args:
            requests: Async generator of generation requests
            output_dir: Directory for output files and state management
            concurrency: Maximum number of concurrent requests
            rate_limit: Maximum requests per minute
            resume: Whether to resume from existing checkpoint

        Returns:
            BatchSummary: Complete summary of batch processing results
        """
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state management
        state_file = output_dir / "_batch_state.jsonl"
        completed_requests: Set[str] = set()

        # Load existing state if resuming
        if resume and state_file.exists():
            completed_requests = await self._load_checkpoint(state_file)
            logger.info(
                f"Resuming batch: {len(completed_requests)} requests already completed"
            )

        # Initialize counters and rate limiter
        total_requests = 0
        successful = 0
        failed = 0
        skipped = 0

        rate_limiter = TokenBucketRateLimiter(rate_limit)
        semaphore = asyncio.Semaphore(concurrency)

        # Process requests concurrently
        async def process_single_request(request: GenerationRequest) -> BatchResult:
            """Process a single request with rate limiting and error handling."""
            nonlocal successful, failed, skipped

            # Skip if already completed (resume scenario)
            if request.id in completed_requests:
                skipped += 1
                return BatchResult(
                    request_id=request.id,
                    status="skipped",
                    timestamp=time.time(),
                )

            # Apply rate limiting
            await rate_limiter.acquire()

            async with semaphore:
                return await self._process_request_with_retry(
                    request, output_dir, state_file
                )

        # Collect all requests and process them
        request_list = []
        async for request in requests:
            total_requests += 1
            request_list.append(request)

        # Process requests concurrently
        tasks = [process_single_request(req) for req in request_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results and handle exceptions
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Unexpected error in batch processing: {result}")
            elif isinstance(result, BatchResult):
                if result.status == "success":
                    successful += 1
                elif result.status == "failure":
                    failed += 1
                elif result.status == "skipped":
                    pass  # Already counted above

        # Calculate final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        throughput = (
            (total_requests / processing_time * 60) if processing_time > 0 else 0
        )

        # Create summary
        summary = BatchSummary(
            total_requests=total_requests,
            successful=successful,
            failed=failed,
            skipped=skipped,
            processing_time_seconds=processing_time,
            results_log_path=str(state_file),
            throughput_requests_per_minute=throughput,
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
        )

        logger.info(
            f"Batch processing completed: {successful}/{total_requests} successful"
        )
        return summary

    async def _load_checkpoint(self, state_file: Path) -> Set[str]:
        """Load completed request IDs from checkpoint file."""
        completed_requests: Set[str] = set()

        try:
            async with aiofiles.open(state_file, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            result = json.loads(line)
                            if result.get("status") == "success":
                                completed_requests.add(result["request_id"])
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in state file: {line}")
        except FileNotFoundError:
            pass  # No checkpoint file exists yet
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

        return completed_requests

    @retry(  # type: ignore
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _process_request_with_retry(
        self, request: GenerationRequest, output_dir: Path, state_file: Path
    ) -> BatchResult:
        """Process a single request with retry logic and atomic checkpointing."""
        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from ..config import load_config
            from ..core.generation import process_generation_job

            # Convert to GenerationJob and process
            job = request.to_generation_job()
            config = await load_config()

            result = await process_generation_job(job, config)

            # Create batch result
            processing_time = time.time() - start_time
            batch_result = BatchResult(
                request_id=request.id,
                status="success",
                output_path=str(result.local_path),
                processing_time_seconds=processing_time,
                file_size_bytes=result.file_size_bytes,
                metadata=result.metadata,
            )

            # Atomically write to checkpoint
            await self._write_checkpoint(state_file, batch_result)

            return batch_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Request {request.id} failed: {error_msg}")

            batch_result = BatchResult(
                request_id=request.id,
                status="failure",
                error_message=error_msg,
                processing_time_seconds=processing_time,
            )

            # Write failure to checkpoint
            await self._write_checkpoint(state_file, batch_result)

            return batch_result

    async def _write_checkpoint(self, state_file: Path, result: BatchResult) -> None:
        """Atomically write a batch result to the checkpoint file."""
        try:
            # Write to temporary file first for atomicity
            temp_file = state_file.with_suffix(".tmp")

            async with aiofiles.open(temp_file, "a") as f:
                result_json = result.model_dump_json()
                await f.write(f"{result_json}\n")
                await f.flush()

            # Atomic move (on most filesystems)
            if state_file.exists():
                # Append to existing file
                async with aiofiles.open(state_file, "a") as f:
                    result_json = result.model_dump_json()
                    await f.write(f"{result_json}\n")
                    await f.flush()
                # Remove temp file
                temp_file.unlink(missing_ok=True)
            else:
                # Move temp file to final location
                temp_file.rename(state_file)

        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rate."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_second = requests_per_minute / 60.0
        self.bucket_size = max(1, requests_per_minute // 10)  # Allow small bursts
        self.tokens = float(self.bucket_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()

            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(
                self.bucket_size, self.tokens + elapsed * self.tokens_per_second
            )
            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.tokens_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
