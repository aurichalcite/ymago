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
    async def test_submit_raises_error_if_empty(self):
        """Test that submit raises ValueError if jobs list is empty."""
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        with pytest.raises(ValueError, match="Jobs list cannot be empty"):
            await backend.submit([])

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

    def test_payload_serialization(self):
        """Test serialization of GenerationJob to JSON payload for GCT."""
        from ymago.models import GenerationJob
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        
        job = GenerationJob(
            prompt="A beautiful sunset",
            image_model="gemini-2.5-flash-image-preview",
            aspect_ratio="16:9",
            destination="gs://my-bucket/output.png",
            webhook_url="https://api.example.com/webhook"
        )
        
        payload = backend._serialize_job(job)
        import json
        data = json.loads(payload)
        
        assert data["prompt"] == "A beautiful sunset"
        assert data["image_model"] == "gemini-2.5-flash-image-preview"
        assert data["aspect_ratio"] == "16:9"
        assert data["destination"] == "gs://my-bucket/output.png"
        assert data["webhook_url"] == "https://api.example.com/webhook"
        assert "request_id" in data

    @pytest.mark.asyncio
    async def test_create_gct_task_calls_client(self):
        """Test that _create_gct_task calls the GCT client correctly."""
        from unittest.mock import MagicMock, patch
        from ymago.models import GenerationJob
        
        config = Settings(
            auth=Auth(google_api_key="test-key"),
            cloud_tasks=CloudTasksConfig(
                gct_project_id="test-project",
                gct_location="us-central1",
                gct_queue_name="test-queue",
                worker_url="https://worker.com",
                service_account_email="sa@example.com"
            )
        )
        backend = CloudTasksExecutionBackend(config)
        
        job = GenerationJob(prompt="Test")
        
        mock_client = MagicMock()
        mock_client.queue_path.return_value = "projects/test-project/locations/us-central1/queues/test-queue"
        backend._client = mock_client
        
        with patch.object(backend, "_serialize_job", return_value='{"test": "data"}'):
            await backend._create_gct_task(job)
            
            mock_client.create_task.assert_called_once()
            _, kwargs = mock_client.create_task.call_args
            request = kwargs.get("request")
            parent = request["parent"]
            task = request["task"]
            
            assert "projects/test-project/locations/us-central1/queues/test-queue" in str(parent)
            assert task["http_request"]["url"] == "https://worker.com"
            assert task["http_request"]["body"] == b'{"test": "data"}'
            assert task["http_request"]["oidc_token"]["service_account_email"] == "sa@example.com"

    @pytest.mark.asyncio
    async def test_submit_calls_create_task_per_job(self):
        """Test that submit dispatches a task for each job."""
        from unittest.mock import patch, AsyncMock
        from ymago.models import GenerationJob
        from pathlib import Path
        
        config = Settings(auth=Auth(google_api_key="test-key"))
        backend = CloudTasksExecutionBackend(config)
        
        jobs = [
            GenerationJob(prompt="Job 1"),
            GenerationJob(prompt="Job 2"),
        ]
        
        with patch.object(backend, "_create_gct_task", new_callable=AsyncMock) as mock_create:
            results = await backend.submit(jobs)
            
            assert mock_create.call_count == 2
            assert len(results) == 2
            assert all(r.get_metadata("execution_backend") == "cloud-tasks" for r in results)
            assert all(r.local_path == Path("cloud-task-dispatched").resolve() for r in results)
