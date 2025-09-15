"""
Core functionality for ymago package.

This package contains the core abstractions and implementations for
storage, execution backends, and generation orchestration.
"""

from .backends import ExecutionBackend, LocalExecutionBackend
from .batch_parser import BatchParseError, parse_batch_input
from .generation import GenerationError, StorageError, process_generation_job
from .notifications import (
    NotificationService,
    WebhookPayload,
    create_failure_payload,
    create_success_payload,
)
from .storage import LocalStorageUploader, StorageBackendRegistry, StorageUploader

__all__ = [
    "ExecutionBackend",
    "LocalExecutionBackend",
    "StorageUploader",
    "LocalStorageUploader",
    "StorageBackendRegistry",
    "NotificationService",
    "WebhookPayload",
    "create_success_payload",
    "create_failure_payload",
    "process_generation_job",
    "GenerationError",
    "StorageError",
    "BatchParseError",
    "parse_batch_input",
]
