"""
Tests for ymago command-line interface.

This module tests the Typer CLI commands, parameter parsing, output validation,
and error handling using CliRunner.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from ymago import __version__
from ymago.cli import app
from ymago.core.generation import GenerationError, StorageError


class TestCLIRunner:
    """Test CLI commands using Typer's CliRunner."""

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_cli_help_command(self):
        """Test main CLI help output."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "ymago" in result.stdout
        assert "An advanced, asynchronous command-line toolkit" in result.stdout
        assert "image" in result.stdout
        assert "config" in result.stdout
        assert "version" in result.stdout

    def test_version_command(self):
        """Test version command output."""
        result = self.runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_image_help_command(self):
        """Test image subcommand help."""
        result = self.runner.invoke(app, ["image", "--help"])

        assert result.exit_code == 0
        assert "generate" in result.stdout

    def test_image_generate_help_command(self):
        """Test image generate command help."""
        result = self.runner.invoke(app, ["image", "generate", "--help"])

        assert result.exit_code == 0
        assert "prompt" in result.stdout
        assert "--filename" in result.stdout
        assert "--seed" in result.stdout
        assert "--quality" in result.stdout
        assert "--aspect-ratio" in result.stdout
        assert "--model" in result.stdout
        assert "--verbose" in result.stdout


class TestConfigCommand:
    """Test the config command."""

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_config_command_success(self, sample_config):
        """Test config command with successful configuration loading."""
        with patch("ymago.cli.load_config") as mock_load_config:
            mock_load_config.return_value = sample_config

            result = self.runner.invoke(app, ["config"])

            assert result.exit_code == 0
            assert "Current Configuration" in result.stdout
            assert "Image Model" in result.stdout
            assert "Output Path" in result.stdout
            assert "API Key" in result.stdout
            assert "***2345" in result.stdout  # Masked API key

    def test_config_command_with_show_path(self, sample_config):
        """Test config command with --show-path option."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.Path.exists"
        ) as mock_exists:

            mock_load_config.return_value = sample_config
            mock_exists.side_effect = lambda path: "ymago.toml" in str(path)

            result = self.runner.invoke(app, ["config", "--show-path"])

            assert result.exit_code == 0
            assert "Configuration file:" in result.stdout

    def test_config_command_environment_variables_only(self, sample_config):
        """Test config command when using environment variables."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.Path.exists"
        ) as mock_exists:

            mock_load_config.return_value = sample_config
            mock_exists.return_value = False  # No config files exist

            result = self.runner.invoke(app, ["config", "--show-path"])

            assert result.exit_code == 0
            assert "Configuration from environment variables" in result.stdout

    def test_config_command_load_error(self):
        """Test config command with configuration loading error."""
        with patch("ymago.cli.load_config") as mock_load_config:
            mock_load_config.side_effect = FileNotFoundError("No config found")

            result = self.runner.invoke(app, ["config"])

            assert result.exit_code == 1
            assert "Error loading configuration" in result.stdout


class TestImageGenerateCommand:
    """Test the image generate command."""

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_generate_command_success(self, sample_config, sample_generation_result):
        """Test successful image generation command."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_generation_result

            result = self.runner.invoke(
                app, ["image", "generate", "A beautiful sunset"]
            )

            assert result.exit_code == 0
            assert "✓ Image generated successfully!" in result.stdout
            assert str(sample_generation_result.local_path) in result.stdout

    def test_generate_command_with_all_options(
        self, sample_config, sample_generation_result
    ):
        """Test image generation with all command options."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_generation_result

            result = self.runner.invoke(
                app,
                [
                    "image",
                    "generate",
                    "A test prompt",
                    "--filename",
                    "custom_name",
                    "--seed",
                    "42",
                    "--quality",
                    "high",
                    "--aspect-ratio",
                    "16:9",
                    "--model",
                    "custom-model",
                    "--verbose",
                ],
            )

            assert result.exit_code == 0
            assert "✓ Image generated successfully!" in result.stdout

            # Verify the job was created with correct parameters
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.prompt == "A test prompt"
            assert job_arg.output_filename == "custom_name"
            assert job_arg.seed == 42
            assert job_arg.quality == "high"
            assert job_arg.aspect_ratio == "16:9"
            assert job_arg.image_model == "custom-model"

    def test_generate_command_verbose_mode(
        self, sample_config, sample_generation_result
    ):
        """Test image generation in verbose mode."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_generation_result

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt", "--verbose"]
            )

            assert result.exit_code == 0
            assert "Generation Job Details" in result.stdout
            assert "Prompt" in result.stdout
            assert "Model" in result.stdout
            assert "Generation Details" in result.stdout

    def test_generate_command_config_error(self):
        """Test image generation with configuration error."""
        with patch("ymago.cli.load_config") as mock_load_config:
            mock_load_config.side_effect = FileNotFoundError("No config found")

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt"]
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_generate_command_generation_error(self, sample_config):
        """Test image generation with generation error."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = GenerationError("API quota exceeded")

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt"]
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "API quota exceeded" in result.stdout

    def test_generate_command_storage_error(self, sample_config):
        """Test image generation with storage error."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = StorageError("Permission denied")

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt"]
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Permission denied" in result.stdout

    def test_generate_command_unexpected_error(self, sample_config):
        """Test image generation with unexpected error."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = Exception("Unexpected error")

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt"]
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_generate_command_unexpected_error_verbose(self, sample_config):
        """Test image generation with unexpected error in verbose mode."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = Exception("Unexpected error")

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt", "--verbose"]
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            # In verbose mode, should show more details
            assert "Exception" in result.stdout or "Traceback" in result.stdout


class TestParameterValidation:
    """Test CLI parameter validation and parsing."""

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_generate_command_missing_prompt(self):
        """Test error when prompt is missing."""
        result = self.runner.invoke(app, ["image", "generate"])

        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Error" in result.stdout

    def test_generate_command_invalid_seed(self, sample_config):
        """Test parameter validation with invalid seed."""
        with patch("ymago.cli.load_config") as mock_load_config:
            mock_load_config.return_value = sample_config

            result = self.runner.invoke(
                app, ["image", "generate", "test", "--seed", "not_a_number"]
            )

            assert result.exit_code != 0

    def test_generate_command_parameter_types(self, sample_config, sample_generation_result):
        """Test that parameters are correctly typed."""
        with patch("ymago.cli.load_config") as mock_load_config, patch(
            "ymago.cli.process_generation_job"
        ) as mock_process_job:

            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_generation_result

            result = self.runner.invoke(
                app,
                [
                    "image",
                    "generate",
                    "test prompt",
                    "--seed",
                    "123",
                    "--filename",
                    "test_file",
                ],
            )

            assert result.exit_code == 0

            # Verify parameter types in the job
            job_arg = mock_process_job.call_args[0][0]
            assert isinstance(job_arg.seed, int)
            assert job_arg.seed == 123
            assert isinstance(job_arg.output_filename, str)
            assert job_arg.output_filename == "test_file"
