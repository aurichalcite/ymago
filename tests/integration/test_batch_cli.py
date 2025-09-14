"""
Integration tests for batch CLI commands.

This module tests the complete batch processing workflow through the CLI
interface, including argument validation and end-to-end processing.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ymago.cli import app
from ymago.models import BatchSummary


class TestBatchCLI:
    """Test batch CLI command functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        csv_content = """prompt,output_name,seed
"A beautiful sunset","sunset",42
"A mountain landscape","mountain",123
"A forest scene","forest",456
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            return Path(f.name)

    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a sample JSONL file for testing."""
        jsonl_content = """{"prompt": "A beautiful sunset", "output_filename": "sunset", "seed": 42}
{"prompt": "A mountain landscape", "output_filename": "mountain", "seed": 123}
{"prompt": "A forest scene", "output_filename": "forest", "seed": 456}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            return Path(f.name)

    def test_batch_run_help(self, runner):
        """Test batch run command help."""
        result = runner.invoke(app, ["batch", "run", "--help"])
        assert result.exit_code == 0
        assert "Process a batch of generation requests" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--concurrency" in result.stdout
        assert "--rate-limit" in result.stdout
        assert "--resume" in result.stdout

    def test_batch_run_missing_arguments(self, runner):
        """Test batch run command with missing required arguments."""
        result = runner.invoke(app, ["batch", "run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Usage:" in result.stdout

    def test_batch_run_nonexistent_file(self, runner):
        """Test batch run command with non-existent input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                app, ["batch", "run", "/nonexistent/file.csv", "--output-dir", temp_dir]
            )
            assert result.exit_code == 1
            assert "Input file not found" in result.stdout

    def test_batch_run_invalid_output_dir(self, runner, sample_csv_file):
        """Test batch run command with invalid output directory."""
        try:
            result = runner.invoke(
                app,
                [
                    "batch",
                    "run",
                    str(sample_csv_file),
                    "--output-dir",
                    "/invalid/readonly/path",
                ],
            )
            assert result.exit_code == 1
            assert "Cannot write to output directory" in result.stdout
        finally:
            sample_csv_file.unlink()

    def test_batch_run_dry_run_csv(self, runner, sample_csv_file):
        """Test batch run command with dry run on CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch("ymago.cli.load_config") as mock_config:
                    mock_config.return_value = MagicMock()

                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "run",
                            str(sample_csv_file),
                            "--output-dir",
                            temp_dir,
                            "--dry-run",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Dry run completed successfully" in result.stdout
                    assert "Found 3 valid requests" in result.stdout
                    assert "Would process 3 requests" in result.stdout
            finally:
                sample_csv_file.unlink()

    def test_batch_run_dry_run_jsonl(self, runner, sample_jsonl_file):
        """Test batch run command with dry run on JSONL file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch("ymago.cli.load_config") as mock_config:
                    mock_config.return_value = MagicMock()

                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "run",
                            str(sample_jsonl_file),
                            "--output-dir",
                            temp_dir,
                            "--format",
                            "jsonl",
                            "--dry-run",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Dry run completed successfully" in result.stdout
                    assert "Found 3 valid requests" in result.stdout
            finally:
                sample_jsonl_file.unlink()

    def test_batch_run_parameter_validation(self, runner, sample_csv_file):
        """Test batch run command parameter validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test invalid concurrency (too high)
                result = runner.invoke(
                    app,
                    [
                        "batch",
                        "run",
                        str(sample_csv_file),
                        "--output-dir",
                        temp_dir,
                        "--concurrency",
                        "100",  # Max is 50
                        "--dry-run",
                    ],
                )
                assert result.exit_code != 0

                # Test invalid concurrency (too low)
                result = runner.invoke(
                    app,
                    [
                        "batch",
                        "run",
                        str(sample_csv_file),
                        "--output-dir",
                        temp_dir,
                        "--concurrency",
                        "0",  # Min is 1
                        "--dry-run",
                    ],
                )
                assert result.exit_code != 0

            finally:
                sample_csv_file.unlink()

    def test_batch_run_full_execution_mock(self, runner, sample_csv_file):
        """Test full batch execution with mocked backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Mock the configuration and backend
                mock_summary = BatchSummary(
                    total_requests=3,
                    successful=2,
                    failed=1,
                    skipped=0,
                    processing_time_seconds=10.5,
                    results_log_path=f"{temp_dir}/state.jsonl",
                    throughput_requests_per_minute=17.1,
                    start_time="2024-01-01T00:00:00Z",
                    end_time="2024-01-01T00:00:10Z",
                )

                with patch("ymago.cli.load_config") as mock_config:
                    with patch("ymago.cli.LocalExecutionBackend") as mock_backend_class:
                        mock_config.return_value = MagicMock()
                        mock_backend = MagicMock()
                        mock_backend.process_batch = AsyncMock(
                            return_value=mock_summary
                        )
                        mock_backend_class.return_value = mock_backend

                        result = runner.invoke(
                            app,
                            [
                                "batch",
                                "run",
                                str(sample_csv_file),
                                "--output-dir",
                                temp_dir,
                                "--concurrency",
                                "5",
                                "--rate-limit",
                                "120",
                            ],
                        )

                        assert result.exit_code == 0
                        assert "Batch Processing Complete" in result.stdout
                        assert "Total Requests" in result.stdout
                        assert "Successful" in result.stdout
                        assert "Failed" in result.stdout

                        # Verify backend was called with correct parameters
                        mock_backend.process_batch.assert_called_once()
                        call_args = mock_backend.process_batch.call_args
                        assert call_args.kwargs["concurrency"] == 5
                        assert call_args.kwargs["rate_limit"] == 120
                        assert call_args.kwargs["resume"] == False
            finally:
                sample_csv_file.unlink()

    def test_batch_run_with_resume(self, runner, sample_csv_file):
        """Test batch run command with resume option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                mock_summary = BatchSummary(
                    total_requests=3,
                    successful=1,
                    failed=0,
                    skipped=2,  # 2 were already completed
                    processing_time_seconds=5.0,
                    results_log_path=f"{temp_dir}/state.jsonl",
                    throughput_requests_per_minute=12.0,
                    start_time="2024-01-01T00:00:00Z",
                    end_time="2024-01-01T00:00:05Z",
                )

                with patch("ymago.cli.load_config") as mock_config:
                    with patch("ymago.cli.LocalExecutionBackend") as mock_backend_class:
                        mock_config.return_value = MagicMock()
                        mock_backend = MagicMock()
                        mock_backend.process_batch = AsyncMock(
                            return_value=mock_summary
                        )
                        mock_backend_class.return_value = mock_backend

                        result = runner.invoke(
                            app,
                            [
                                "batch",
                                "run",
                                str(sample_csv_file),
                                "--output-dir",
                                temp_dir,
                                "--resume",
                            ],
                        )

                        assert result.exit_code == 0
                        assert "Skipped" in result.stdout

                        # Verify resume was passed correctly
                        call_args = mock_backend.process_batch.call_args
                        assert call_args.kwargs["resume"] == True
            finally:
                sample_csv_file.unlink()

    def test_batch_run_verbose_output(self, runner, sample_csv_file):
        """Test batch run command with verbose output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch("ymago.cli.load_config") as mock_config:
                    mock_config.return_value = MagicMock()

                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "run",
                            str(sample_csv_file),
                            "--output-dir",
                            temp_dir,
                            "--verbose",
                            "--dry-run",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Configuration loaded" in result.stdout
                    assert "Input file:" in result.stdout
                    assert "Output directory:" in result.stdout
                    assert "Concurrency:" in result.stdout
                    assert "Rate limit:" in result.stdout
            finally:
                sample_csv_file.unlink()

    def test_batch_run_empty_file(self, runner):
        """Test batch run command with empty input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("prompt,output_name\n")  # Header only, no data
            empty_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch("ymago.cli.load_config") as mock_config:
                    mock_config.return_value = MagicMock()

                    result = runner.invoke(
                        app, ["batch", "run", str(empty_file), "--output-dir", temp_dir]
                    )

                    assert result.exit_code == 0
                    assert "No valid requests found" in result.stdout
            finally:
                empty_file.unlink()

    def test_batch_run_keyboard_interrupt(self, runner, sample_csv_file):
        """Test batch run command handling of keyboard interrupt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch("ymago.cli.load_config") as mock_config:
                    with patch("ymago.cli.LocalExecutionBackend") as mock_backend_class:
                        mock_config.return_value = MagicMock()
                        mock_backend = MagicMock()
                        mock_backend.process_batch = AsyncMock(
                            side_effect=KeyboardInterrupt()
                        )
                        mock_backend_class.return_value = mock_backend

                        result = runner.invoke(
                            app,
                            [
                                "batch",
                                "run",
                                str(sample_csv_file),
                                "--output-dir",
                                temp_dir,
                            ],
                        )

                        assert result.exit_code == 1
                        assert "cancelled by user" in result.stdout
                        assert "Use --resume to continue" in result.stdout
            finally:
                sample_csv_file.unlink()

    def test_batch_run_with_format_hint(self, runner, sample_csv_file):
        """Test batch run command with explicit format hint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch("ymago.cli.load_config") as mock_config:
                    mock_config.return_value = MagicMock()

                    result = runner.invoke(
                        app,
                        [
                            "batch",
                            "run",
                            str(sample_csv_file),
                            "--output-dir",
                            temp_dir,
                            "--format",
                            "csv",
                            "--dry-run",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Found 3 valid requests" in result.stdout
            finally:
                sample_csv_file.unlink()
