"""
Execution backend abstraction for ymago package.

This module provides abstract base classes for job execution and concrete
implementations for local execution. Future implementations can extend this
to support distributed execution, cloud functions, etc.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List

from ..models import GenerationJob, GenerationResult

if TYPE_CHECKING:
    from ..config import Settings


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
