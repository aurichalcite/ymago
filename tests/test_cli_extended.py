"""
Tests for CLI integration with cloud storage and webhook options.

This module tests the new CLI options for cloud storage destinations
and webhook notifications added in Milestone 4.
"""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ymago.cli import app


class TestCLICloudStorageOptions:
    """Test CLI cloud storage destination options."""

    runner: Optional[CliRunner] = None

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("ymago.cli.process_generation_job")
    @patch("ymago.cli.load_config")
    def test_image_generate_with_s3_destination(
        self, mock_load_config, mock_process_job
    ):
        """Test image generation with S3 destination."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.auth.google_api_key = "test-key"
        mock_config.defaults.output_path = Path("/tmp")
        mock_config.defaults.image_model = "gemini-pro"
        mock_config.cloud_storage.aws_access_key_id = "test-key"
        mock_config.cloud_storage.aws_secret_access_key = "test-secret"
        mock_config.cloud_storage.aws_region = "us-east-1"
        mock_load_config.return_value = mock_config

        # Mock generation result
        mock_result = MagicMock()
        mock_result.local_path = Path("/tmp/test.jpg")
        mock_result.file_size_bytes = 1024
        mock_result.generation_time_seconds = 5.0
        mock_result.metadata = {"model": "test-model"}
        mock_process_job.return_value = mock_result

        # Run CLI command
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "image",
                "generate",
                "test prompt",
                "--destination",
                "s3://test-bucket/images/",
            ],
        )

        assert result.exit_code == 0

        # Verify process_generation_job was called with destination
        mock_process_job.assert_called_once()
        call_args = mock_process_job.call_args
        assert call_args[1]["destination_url"] == "s3://test-bucket/images/"

    @patch("ymago.cli.process_generation_job")
    @patch("ymago.cli.load_config")
    def test_image_generate_with_webhook_url(self, mock_load_config, mock_process_job):
        """Test image generation with webhook URL."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.auth.google_api_key = "test-key"
        mock_config.defaults.output_path = Path("/tmp")
        mock_config.defaults.image_model = "gemini-pro"
        mock_config.webhooks.timeout_seconds = 30
        mock_config.webhooks.retry_attempts = 3
        mock_config.webhooks.retry_backoff_factor = 2.0
        mock_load_config.return_value = mock_config

        # Mock generation result
        mock_result = MagicMock()
        mock_result.local_path = Path("/tmp/test.jpg")
        mock_result.file_size_bytes = 1024
        mock_result.generation_time_seconds = 5.0
        mock_result.metadata = {"model": "test-model"}
        mock_process_job.return_value = mock_result

        # Run CLI command
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "image",
                "generate",
                "test prompt",
                "--webhook-url",
                "https://webhook.example.com/notify",
            ],
        )

        assert result.exit_code == 0

        # Verify process_generation_job was called with webhook URL
        mock_process_job.assert_called_once()
        call_args = mock_process_job.call_args
        assert call_args[1]["webhook_url"] == "https://webhook.example.com/notify"

    @patch("ymago.cli.process_generation_job")
    @patch("ymago.cli.load_config")
    def test_video_generate_with_gcs_destination(
        self, mock_load_config, mock_process_job
    ):
        """Test video generation with GCS destination."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.auth.google_api_key = "test-key"
        mock_config.defaults.output_path = Path("/tmp")
        mock_config.defaults.video_model = "veo"
        mock_config.cloud_storage.gcp_service_account_path = None
        mock_load_config.return_value = mock_config

        # Mock generation result
        mock_result = MagicMock()
        mock_result.local_path = Path("/tmp/test.mp4")
        mock_result.file_size_bytes = 2048
        mock_result.generation_time_seconds = 15.0
        mock_result.metadata = {"model": "test-model"}
        mock_process_job.return_value = mock_result

        # Run CLI command
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "video",
                "generate",
                "test video prompt",
                "--destination",
                "gs://test-bucket/videos/",
            ],
        )

        assert result.exit_code == 0

        # Verify process_generation_job was called with destination
        mock_process_job.assert_called_once()
        call_args = mock_process_job.call_args
        assert call_args[1]["destination_url"] == "gs://test-bucket/videos/"

    def test_invalid_destination_url_scheme(self):
        """Test CLI validation of invalid destination URL scheme."""
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "image",
                "generate",
                "test prompt",
                "--destination",
                "ftp://invalid.com/path/",
            ],
        )

        assert result.exit_code == 1
        assert "Destination must be a valid cloud storage URL" in result.stdout

    def test_invalid_destination_url_format(self):
        """Test CLI validation of invalid destination URL format."""
        assert self.runner is not None
        result = self.runner.invoke(
            app, ["image", "generate", "test prompt", "--destination", "not-a-url"]
        )

        assert result.exit_code == 1
        assert "Destination must be a valid cloud storage URL" in result.stdout

    @patch("ymago.cli.process_generation_job")
    @patch("ymago.cli.load_config")
    def test_combined_destination_and_webhook(self, mock_load_config, mock_process_job):
        """Test using both destination and webhook options together."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.auth.google_api_key = "test-key"
        mock_config.defaults.output_path = Path("/tmp")
        mock_config.defaults.image_model = "gemini-pro"
        mock_config.cloud_storage.r2_account_id = "test-account"
        mock_config.cloud_storage.r2_access_key_id = "test-key"
        mock_config.cloud_storage.r2_secret_access_key = "test-secret"
        mock_config.webhooks.timeout_seconds = 30
        mock_config.webhooks.retry_attempts = 3
        mock_config.webhooks.retry_backoff_factor = 2.0
        mock_load_config.return_value = mock_config

        # Mock generation result
        mock_result = MagicMock()
        mock_result.local_path = Path("/tmp/test.jpg")
        mock_result.file_size_bytes = 1024
        mock_result.generation_time_seconds = 5.0
        mock_result.metadata = {"model": "test-model"}
        mock_process_job.return_value = mock_result

        # Run CLI command with both options
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "image",
                "generate",
                "test prompt",
                "--destination",
                "r2://test-bucket/images/",
                "--webhook-url",
                "https://webhook.example.com/notify",
            ],
        )

        assert result.exit_code == 0

        # Verify both options were passed
        mock_process_job.assert_called_once()
        call_args = mock_process_job.call_args
        assert call_args[1]["destination_url"] == "r2://test-bucket/images/"
        assert call_args[1]["webhook_url"] == "https://webhook.example.com/notify"


class TestCLIValidationFunctions:
    """Test CLI validation functions for new options."""

    def test_validate_destination_url_valid_schemes(self):
        """Test destination URL validation with valid schemes."""
        from ymago.cli import _validate_destination_url

        # Test valid schemes
        assert _validate_destination_url("s3://bucket/path/") is True
        assert _validate_destination_url("gs://bucket/path/") is True
        assert _validate_destination_url("r2://bucket/path/") is True
        assert _validate_destination_url("file:///local/path/") is True

        # Test case insensitivity
        assert _validate_destination_url("S3://bucket/path/") is True
        assert _validate_destination_url("GS://bucket/path/") is True

    def test_validate_destination_url_invalid_schemes(self):
        """Test destination URL validation with invalid schemes."""
        from ymago.cli import _validate_destination_url

        # Test invalid schemes
        assert _validate_destination_url("ftp://server/path/") is False
        assert _validate_destination_url("http://server/path/") is False
        assert _validate_destination_url("https://server/path/") is False
        assert _validate_destination_url("invalid://server/path/") is False
        assert _validate_destination_url("not-a-url") is False


class TestCLIErrorHandling:
    """Test CLI error handling for cloud storage and webhook failures."""

    runner: Optional[CliRunner] = None

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("ymago.cli.process_generation_job")
    @patch("ymago.cli.load_config")
    def test_storage_error_handling(self, mock_load_config, mock_process_job):
        """Test CLI handling of storage errors."""
        from ymago.core.storage import StorageError

        # Mock configuration
        mock_config = MagicMock()
        mock_config.auth.google_api_key = "test-key"
        mock_config.defaults.output_path = Path("/tmp")
        mock_config.defaults.image_model = "gemini-pro"
        mock_load_config.return_value = mock_config

        # Mock storage error
        mock_process_job.side_effect = StorageError("Failed to upload to S3")

        # Run CLI command
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "image",
                "generate",
                "test prompt",
                "--destination",
                "s3://test-bucket/images/",
            ],
        )

        assert result.exit_code == 1
        assert "Failed to upload to S3" in result.stdout

    @patch("ymago.cli.process_generation_job")
    @patch("ymago.cli.load_config")
    def test_generation_with_missing_cloud_dependencies(
        self, mock_load_config, mock_process_job
    ):
        """Test CLI handling when cloud storage dependencies are missing."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.auth.google_api_key = "test-key"
        mock_config.defaults.output_path = Path("/tmp")
        mock_config.defaults.image_model = "gemini-pro"
        mock_load_config.return_value = mock_config

        # Mock import error for missing dependencies
        mock_process_job.side_effect = ImportError(
            "AWS S3 support requires 'aioboto3'. Install with: pip install 'ymago[aws]'"
        )

        # Run CLI command
        assert self.runner is not None
        result = self.runner.invoke(
            app,
            [
                "image",
                "generate",
                "test prompt",
                "--destination",
                "s3://test-bucket/images/",
            ],
        )

        assert result.exit_code == 1
        assert "AWS S3 support requires 'aioboto3'" in result.stdout


class TestCLIHelpText:
    """Test CLI help text includes new options."""

    runner: Optional[CliRunner] = None

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_image_generate_help_includes_new_options(self):
        """Test that image generate help includes destination and webhook options."""
        assert self.runner is not None
        result = self.runner.invoke(app, ["image", "generate", "--help"])

        assert result.exit_code == 0
        assert "--destination" in result.stdout
        assert "--webhook-url" in result.stdout
        assert "Output & Delivery" in result.stdout
        assert "Cloud storage destination" in result.stdout
        assert "Webhook URL for job completion notifications" in result.stdout

    def test_video_generate_help_includes_new_options(self):
        """Test that video generate help includes destination and webhook options."""
        assert self.runner is not None
        result = self.runner.invoke(app, ["video", "generate", "--help"])

        assert result.exit_code == 0
        assert "--destination" in result.stdout
        assert "--webhook-url" in result.stdout
        assert "Output & Delivery" in result.stdout
        assert "Cloud storage destination" in result.stdout
        assert "Webhook URL for job completion notifications" in result.stdout

    def test_help_shows_example_usage(self):
        """Test that help text shows example usage of new options."""
        assert self.runner is not None
        result = self.runner.invoke(app, ["image", "generate", "--help"])

        assert result.exit_code == 0
        assert "s3://my-bucket/images/" in result.stdout
        assert "https://api.example.com/webhook" in result.stdout
