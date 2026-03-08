"""
Integration tests for Google Cloud Tasks backend.
"""

import pytest

pytest.importorskip("google.cloud.tasks")

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ymago.cli import app


class TestGCTIntegration:
    """End-to-end integration tests for GCT backend."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_to_gct_task_creation(self):
        """Test the full flow from CLI to GCT client task creation."""
        # Mock GCT client
        mock_client = MagicMock()
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"

        with patch("google.cloud.tasks_v2.CloudTasksClient", return_value=mock_client):
            # Mock configuration loading to ensure GCT is configured
            from ymago.config import Auth, CloudTasksConfig, Defaults, Settings

            sample_config = Settings(
                auth=Auth(google_api_key="test-key"),
                defaults=Defaults(output_path=Path("/tmp")),
                cloud_tasks=CloudTasksConfig(
                    gct_project_id="test-project", worker_url="https://worker.com"
                ),
            )

            with patch("ymago.cli.load_config", return_value=sample_config):
                from ymago.core.backends import get_backend

                with patch(
                    "ymago.cli.get_backend", wraps=get_backend
                ) as mock_get_backend:
                    # Run CLI command
                    result = self.runner.invoke(
                        app,
                        [
                            "image",
                            "generate",
                            "Integration test",
                            "--backend",
                            "cloud-tasks",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "✓ Image generated successfully!" in result.stdout

                    # Verify get_backend was called
                    mock_get_backend.assert_called_once()

                    # Verify GCT client was called
                    mock_client.create_task.assert_called_once()
                _, kwargs = mock_client.create_task.call_args
                request = kwargs.get("request")
                assert request["parent"] == "projects/p/locations/l/queues/q"
                assert request["task"]["http_request"]["url"] == "https://worker.com"

    def test_batch_run_to_gct_task_creation(self, tmp_path):
        """Test batch run flow to GCT client task creation."""
        input_file = tmp_path / "batch.csv"
        input_file.write_text("prompt\nBatch Job 1\nBatch Job 2")

        mock_client = MagicMock()
        mock_client.queue_path.return_value = "projects/p/locations/l/queues/q"

        with patch("google.cloud.tasks_v2.CloudTasksClient", return_value=mock_client):
            from ymago.config import Auth, CloudTasksConfig, Defaults, Settings

            sample_config = Settings(
                auth=Auth(google_api_key="test-key"),
                defaults=Defaults(output_path=Path("/tmp")),
                cloud_tasks=CloudTasksConfig(
                    gct_project_id="test-project", worker_url="https://worker.com"
                ),
            )

            with patch("ymago.cli.load_config", return_value=sample_config):
                from ymago.core.backends import get_backend

                with patch(
                    "ymago.cli.get_backend", wraps=get_backend
                ) as mock_get_backend:
                    result = self.runner.invoke(
                        app,
                        [
                            "batch",
                            "run",
                            str(input_file),
                            "-o",
                            str(tmp_path),
                            "--backend",
                            "cloud-tasks",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Batch Processing Complete!" in result.stdout

                    # Verify get_backend was called
                    mock_get_backend.assert_called_once()

                    # Verify GCT client was called twice
                    assert mock_client.create_task.call_count == 2
