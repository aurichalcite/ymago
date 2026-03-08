"""
Tests for Google Cloud Tasks CLI integration.
"""

import pytest

pytest.importorskip("google.cloud.tasks")

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from ymago.cli import app
from ymago.core.backends import CloudTasksExecutionBackend, LocalExecutionBackend
from ymago.models import BatchSummary, GenerationJob, GenerationResult


class TestCLIGCTIntegration:
    """Test CLI integration with Google Cloud Tasks backend."""

    runner: CliRunner = None  # type: ignore

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    @pytest.fixture
    def sample_config(self):
        from ymago.config import Auth, CloudTasksConfig, Defaults, Settings

        return Settings(
            auth=Auth(google_api_key="test-key"),
            defaults=Defaults(output_path=Path("/tmp")),
            cloud_tasks=CloudTasksConfig(gct_project_id="test-project"),
        )

    def test_image_generate_with_backend_option(self, sample_config):
        """Test that image generate supports --backend option."""
        mock_result = GenerationResult(
            local_path=Path("cloud-task-dispatched"),
            job=GenerationJob(prompt="test"),
            metadata={"execution_backend": "cloud-tasks"},
        )

        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load:
            with patch("ymago.cli.get_backend") as mock_get_backend:
                mock_load.return_value = sample_config
                mock_backend = MagicMock(spec=CloudTasksExecutionBackend)
                mock_backend.submit = AsyncMock(return_value=[mock_result])
                mock_get_backend.return_value = mock_backend

                result = self.runner.invoke(
                    app,
                    ["image", "generate", "test prompt", "--backend", "cloud-tasks"],
                )

                assert result.exit_code == 0
                assert "✓ Image generated successfully!" in result.stdout
                mock_get_backend.assert_called_once()
                mock_backend.submit.assert_called_once()

    def test_image_generate_with_local_backend(self, sample_config):
        """Test that image generate defaults to local backend."""
        mock_result = GenerationResult(
            local_path=Path("/tmp/output.png"),
            job=GenerationJob(prompt="test"),
            metadata={"api_model": "test"},
        )

        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load:
            with patch("ymago.cli.get_backend") as mock_get_backend:
                with patch(
                    "ymago.cli.process_generation_job", new_callable=AsyncMock
                ) as mock_process:
                    mock_load.return_value = sample_config
                    mock_process.return_value = mock_result
                    mock_get_backend.return_value = MagicMock(
                        spec=LocalExecutionBackend
                    )

                    result = self.runner.invoke(
                        app, ["image", "generate", "test prompt"]
                    )

                    assert result.exit_code == 0
                    mock_get_backend.assert_called_once()
                    mock_process.assert_called_once()

    def test_video_generate_with_backend_option(self, sample_config):
        """Test that video generate supports --backend option."""
        mock_result = GenerationResult(
            local_path=Path("cloud-task-dispatched"),
            job=GenerationJob(prompt="test", media_type="video"),
            metadata={"execution_backend": "cloud-tasks"},
        )

        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load:
            with patch("ymago.cli.get_backend") as mock_get_backend:
                mock_load.return_value = sample_config
                mock_backend = MagicMock(spec=CloudTasksExecutionBackend)
                mock_backend.submit = AsyncMock(return_value=[mock_result])
                mock_get_backend.return_value = mock_backend

                result = self.runner.invoke(
                    app,
                    ["video", "generate", "test prompt", "--backend", "cloud-tasks"],
                )

                assert result.exit_code == 0
                mock_get_backend.assert_called_once()

    def test_batch_run_with_backend_option(self, sample_config, tmp_path):
        """Test that batch run supports --backend option."""
        input_file = tmp_path / "prompts.csv"
        input_file.write_text("prompt\ntest 1\ntest 2")

        mock_summary = BatchSummary(
            total_requests=2,
            successful=2,
            failed=0,
            skipped=0,
            processing_time_seconds=1.0,
            results_log_path=str(tmp_path / "log.jsonl"),
            throughput_requests_per_minute=120.0,
        )

        from ymago.models import GenerationRequest

        async def mock_requests(*args, **kwargs):
            yield GenerationRequest(prompt="test 1")
            yield GenerationRequest(prompt="test 2")

        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load:
            with patch("ymago.cli.get_backend") as mock_get_backend:
                with patch("ymago.cli.parse_batch_input", side_effect=mock_requests):
                    mock_load.return_value = sample_config
                    mock_backend = MagicMock(spec=CloudTasksExecutionBackend)
                    mock_backend.process_batch = AsyncMock(return_value=mock_summary)
                    mock_get_backend.return_value = mock_backend

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
                    mock_get_backend.assert_called_once()
                    mock_backend.process_batch.assert_called_once()

    def test_batch_run_with_local_backend(self, sample_config, tmp_path):
        """Test that batch run defaults to local backend."""
        input_file = tmp_path / "prompts.csv"
        input_file.write_text("prompt\ntest 1")

        from ymago.models import BatchSummary, GenerationRequest

        async def mock_requests(*args, **kwargs):
            yield GenerationRequest(prompt="test 1")

        mock_summary = BatchSummary(
            total_requests=1,
            successful=1,
            failed=0,
            skipped=0,
            processing_time_seconds=1.0,
            results_log_path="log.jsonl",
            throughput_requests_per_minute=60.0,
        )

        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load:
            with patch("ymago.cli.get_backend") as mock_get_backend:
                with patch("ymago.cli.parse_batch_input", side_effect=mock_requests):
                    mock_load.return_value = sample_config
                    mock_backend = MagicMock(spec=LocalExecutionBackend)
                    mock_backend.process_batch = AsyncMock(return_value=mock_summary)
                    mock_get_backend.return_value = mock_backend

                    result = self.runner.invoke(
                        app, ["batch", "run", str(input_file), "-o", str(tmp_path)]
                    )

                    assert result.exit_code == 0
                    mock_get_backend.assert_called_once()
                    mock_backend.process_batch.assert_called_once()
