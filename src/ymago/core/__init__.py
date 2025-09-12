"""
Core functionality for ymago package.

This package contains the core abstractions and implementations for
storage, execution backends, and generation orchestration.
"""

from .backends import ExecutionBackend, LocalExecutionBackend
from .generation import GenerationError, StorageError, process_generation_job
from .storage import LocalStorageUploader, StorageUploader

__all__ = [
    "ExecutionBackend",
    "LocalExecutionBackend",
    "StorageUploader",
    "LocalStorageUploader",
    "process_generation_job",
    "GenerationError",
    "StorageError",
]
