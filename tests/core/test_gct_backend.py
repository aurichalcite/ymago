"""
Tests for Google Cloud Tasks execution backend.
"""

import pytest
from ymago.core.backends import ExecutionBackend
from ymago.config import Settings, Auth, CloudTasksConfig

try:
    from ymago.core.backends import CloudTasksExecutionBackend
except ImportError:
    CloudTasksExecutionBackend = None


class TestCloudTasksExecutionBackendSkeleton:
    """Test the skeleton of CloudTasksExecutionBackend."""

    def test_backend_exists(self):
        """Test that CloudTasksExecutionBackend is defined."""
        assert CloudTasksExecutionBackend is not None

    def test_backend_inherits_from_execution_backend(self):
        """Test that CloudTasksExecutionBackend inherits from ExecutionBackend."""
        assert issubclass(CloudTasksExecutionBackend, ExecutionBackend)

    @pytest.mark.asyncio
    async def test_backend_initialization(self):
        """Test that CloudTasksExecutionBackend can be initialized."""
        config = Settings(
            auth=Auth(google_api_key="test-key"),
            cloud_tasks=CloudTasksConfig(
                gct_project_id="test-project",
                worker_url="https://worker.example.com"
            )
        )
        backend = CloudTasksExecutionBackend(config)
        assert backend.config == config
        assert backend.project_id == "test-project"

    @pytest.mark.asyncio
    async def test_stub_methods_exist(self):
        """Test that stub methods exist and are callable."""
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        
        # These should exist and not raise NotImplementedError if they are stubs
        # (though the protocol says implement stubs, usually meaning they return something or are pass)
        assert hasattr(backend, "submit")
        assert hasattr(backend, "process_batch")
        assert hasattr(backend, "get_status")

    @pytest.mark.asyncio
    async def test_get_status_returns_backend_type(self):
        """Test that get_status returns the correct backend type."""
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        status = await backend.get_status()
        assert status["backend_type"] == "cloud-tasks"

    @pytest.mark.asyncio
    async def test_submit_returns_empty_list_as_stub(self):
        """Test that submit stub returns an empty list."""
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        results = await backend.submit([])
        assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_returns_summary_as_stub(self, tmp_path):
        """Test that process_batch stub returns a BatchSummary."""
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        
        async def mock_requests():
            yield None # type: ignore
            
        summary = await backend.process_batch(
            mock_requests(),
            output_dir=tmp_path,
            concurrency=5,
            rate_limit=60
        )
        from ymago.models import BatchSummary
        assert isinstance(summary, BatchSummary)
        assert summary.total_requests == 0
