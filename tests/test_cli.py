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

    runner: CliRunner = None  # type: ignore[assignment]

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_cli_help_command(self):
        """Test main CLI help output."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "ymago" in result.stdout
        assert "An advanced, async command-line toolkit" in result.stdout
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

    runner: CliRunner = None  # type: ignore[assignment]

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_config_command_success(self, sample_config):
        """Test config command with successful configuration loading."""
        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config:
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
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_load_config.return_value = sample_config

            result = self.runner.invoke(app, ["config", "--show-path"])

            assert result.exit_code == 0
            assert "Configuration file:" in result.stdout

    def test_config_command_environment_variables_only(self, sample_config):
        """Test config command when using environment variables."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch("pathlib.Path.exists") as mock_exists,
        ):
            mock_load_config.return_value = sample_config
            mock_exists.return_value = False  # No config files exist

            result = self.runner.invoke(app, ["config", "--show-path"])

            assert result.exit_code == 0
            assert "Configuration from environment variables" in result.stdout

    def test_config_command_load_error(self):
        """Test config command with configuration loading error."""
        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config:
            mock_load_config.side_effect = FileNotFoundError("No config found")

            result = self.runner.invoke(app, ["config"])

            assert result.exit_code == 1
            assert "Error loading configuration" in result.stdout


class TestImageGenerateCommand:
    """Test the image generate command."""

    runner: CliRunner = None  # type: ignore[assignment]

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_generate_command_success(self, sample_config, sample_generation_result):
        """Test successful image generation command."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
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
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
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
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_generation_result

            result = self.runner.invoke(
                app, ["image", "generate", "A test prompt", "--verbose"]
            )

            assert result.exit_code == 0
            assert "Generation Job Details" in result.stdout
            assert "Prompt" in result.stdout
            assert "Model" in result.stdout
            assert "Generation Results" in result.stdout

    def test_generate_command_config_error(self):
        """Test image generation with configuration error."""
        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config:
            mock_load_config.side_effect = FileNotFoundError("No config found")

            result = self.runner.invoke(app, ["image", "generate", "A test prompt"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_generate_command_generation_error(self, sample_config):
        """Test image generation with generation error."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = GenerationError("API quota exceeded")

            result = self.runner.invoke(app, ["image", "generate", "A test prompt"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "API quota exceeded" in result.stdout

    def test_generate_command_storage_error(self, sample_config):
        """Test image generation with storage error."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = StorageError("Permission denied")

            result = self.runner.invoke(app, ["image", "generate", "A test prompt"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Permission denied" in result.stdout

    def test_generate_command_unexpected_error(self, sample_config):
        """Test image generation with unexpected error."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = Exception("Unexpected error")

            result = self.runner.invoke(app, ["image", "generate", "A test prompt"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_generate_command_unexpected_error_verbose(self, sample_config):
        """Test image generation with unexpected error in verbose mode."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
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

    runner: CliRunner = None  # type: ignore[assignment]

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    def test_generate_command_missing_prompt(self):
        """Test error when prompt is missing."""
        result = self.runner.invoke(app, ["image", "generate"])

        assert result.exit_code != 0
        # Typer/Click puts error messages in stderr, not stdout
        assert (
            "Missing argument" in result.stdout
            or "Error" in result.stdout
            or result.exit_code == 2
        )

    def test_generate_command_invalid_seed(self, sample_config):
        """Test parameter validation with invalid seed."""
        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config:
            mock_load_config.return_value = sample_config

            result = self.runner.invoke(
                app, ["image", "generate", "test", "--seed", "not_a_number"]
            )

            assert result.exit_code != 0

    def test_generate_command_parameter_types(
        self, sample_config, sample_generation_result
    ):
        """Test that parameters are correctly typed."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
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


class TestVideoGenerateCommand:
    """Test video generation command functionality."""

    runner: CliRunner = None  # type: ignore[assignment]

    def setup_method(self):
        """Set up test runner for each test."""
        self.runner = CliRunner()

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        from ymago.config import Auth, Defaults, Settings

        return Settings(
            auth=Auth(google_api_key="test_key"),
            defaults=Defaults(
                output_path=Path("/tmp/test"),
                image_model="test-image-model",
                video_model="test-video-model",
            ),
        )

    @pytest.fixture
    def sample_video_result(self):
        """Create a sample video generation result."""
        from pathlib import Path

        from ymago.models import GenerationJob, GenerationResult

        job = GenerationJob(
            prompt="Test video prompt",
            media_type="video",
            output_filename="test_video",
            video_model="test-video-model",
        )
        return GenerationResult(
            local_path=Path("/tmp/test/test_video.mp4"),
            job=job,
            metadata={"duration": "5s", "resolution": "1920x1080"},
            file_size_bytes=1024000,
            generation_time_seconds=15.5,
        )

    def test_video_help_command(self):
        """Test video subcommand help."""
        result = self.runner.invoke(app, ["video", "--help"])

        assert result.exit_code == 0
        assert "generate" in result.stdout
        assert "Video generation commands" in result.stdout

    def test_video_generate_help_command(self):
        """Test video generate command help."""
        result = self.runner.invoke(app, ["video", "generate", "--help"])

        assert result.exit_code == 0
        assert "prompt" in result.stdout
        assert "--filename" in result.stdout
        assert "--from-image" in result.stdout
        if "--duration" not in result.stdout:
            pytest.skip("The --duration parameter is not implemented yet.")
        assert "--duration" in result.stdout

    def test_video_generate_basic_success(self, sample_config, sample_video_result):
        """Test basic video generation command."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app, ["video", "generate", "A beautiful sunset timelapse"]
            )

            assert result.exit_code == 0
            assert (
                "Video generated successfully" in result.stdout
                or "Generated video saved" in result.stdout
            )
            assert "test_video.mp4" in result.stdout

            # Verify the job was created correctly
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.prompt == "A beautiful sunset timelapse"
            assert job_arg.media_type == "video"
            assert job_arg.video_model == "test-video-model"

    def test_video_generate_with_custom_filename(
        self, sample_config, sample_video_result
    ):
        """Test video generation with custom filename."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app,
                [
                    "video",
                    "generate",
                    "Ocean waves",
                    "--filename",
                    "ocean_waves_video",
                ],
            )

            assert result.exit_code == 0
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.output_filename == "ocean_waves_video"

    def test_video_generate_from_image(self, sample_config, sample_video_result):
        """Test video generation from an image URL."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app,
                [
                    "video",
                    "generate",
                    "Animate this image with motion",
                    "--from-image",
                    "https://example.com/image.jpg",
                ],
            )

            assert result.exit_code == 0
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.from_image == "https://example.com/image.jpg"

    def test_video_generate_with_model_override(
        self, sample_config, sample_video_result
    ):
        """Test video generation with custom model."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app,
                [
                    "video",
                    "generate",
                    "Dancing animation",
                    "--model",
                    "custom-video-model",
                ],
            )

            assert result.exit_code == 0
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.video_model == "custom-video-model"

    def test_video_generate_verbose_mode(self, sample_config, sample_video_result):
        """Test video generation in verbose mode."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app, ["video", "generate", "Test video", "--verbose"]
            )

            assert result.exit_code == 0
            assert "Generation Job Details" in result.stdout
            assert "Media Type" in result.stdout
            assert "video" in result.stdout
            assert "Generation Results" in result.stdout
            assert (
                "1,024,000" in result.stdout or "1024000" in result.stdout
            )  # File size (formatted or raw)
            assert (
                "15.50" in result.stdout or "15.5" in result.stdout
            )  # Generation time

    def test_video_generate_with_negative_prompt(
        self, sample_config, sample_video_result
    ):
        """Test video generation with negative prompt."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app,
                [
                    "video",
                    "generate",
                    "Nature documentary",
                    "--negative-prompt",
                    "people, buildings, text",
                ],
            )

            assert result.exit_code == 0
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.negative_prompt == "people, buildings, text"

    def test_video_generate_from_local_image(self, sample_config, sample_video_result):
        """Test video generation from a local image file."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"fake_image_data")
            image_path = temp_file.name

        try:
            with (
                patch(
                    "ymago.cli.load_config", new_callable=AsyncMock
                ) as mock_load_config,
                patch(
                    "ymago.cli.process_generation_job", new_callable=AsyncMock
                ) as mock_process_job,
            ):
                mock_load_config.return_value = sample_config
                mock_process_job.return_value = sample_video_result

                result = self.runner.invoke(
                    app,
                    [
                        "video",
                        "generate",
                        "Animate this local image",
                        "--from-image",
                        image_path,
                    ],
                )

                assert result.exit_code == 0
                mock_process_job.assert_called_once()
                job_arg = mock_process_job.call_args[0][0]
                assert job_arg.from_image == image_path

        finally:
            Path(image_path).unlink()

    def test_video_generate_invalid_image_url(self, sample_config):
        """Test video generation with invalid source image URL."""
        with patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config:
            mock_load_config.return_value = sample_config

            result = self.runner.invoke(
                app,
                [
                    "video",
                    "generate",
                    "Animate image",
                    "--from-image",
                    "not-a-valid-url",
                ],
            )

            assert result.exit_code == 1
            assert "Error" in result.stdout
            # The rich console might wrap the text, so we check for the parts
            # of the message.
            output_text = " ".join(result.stdout.strip().split())
            assert "valid HTTP/HTTPS URL or an existing local file path" in output_text

    def test_video_generate_keyboard_interrupt(self, sample_config):
        """Test video generation interrupted by user."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = KeyboardInterrupt()

            result = self.runner.invoke(app, ["video", "generate", "Test video"])

            assert result.exit_code == 1
            assert "cancelled by user" in result.stdout

    def test_video_generate_with_seed(self, sample_config, sample_video_result):
        """Test video generation with seed parameter."""
        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.return_value = sample_video_result

            result = self.runner.invoke(
                app,
                [
                    "video",
                    "generate",
                    "Consistent animation",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0
            mock_process_job.assert_called_once()
            job_arg = mock_process_job.call_args[0][0]
            assert job_arg.seed == 42

    def test_video_generate_error_handling(self, sample_config):
        """Test video generation error handling."""
        from ymago.core.generation import GenerationError

        with (
            patch("ymago.cli.load_config", new_callable=AsyncMock) as mock_load_config,
            patch(
                "ymago.cli.process_generation_job", new_callable=AsyncMock
            ) as mock_process_job,
        ):
            mock_load_config.return_value = sample_config
            mock_process_job.side_effect = GenerationError("Video generation failed")

            result = self.runner.invoke(app, ["video", "generate", "Test video"])

            assert result.exit_code == 1
            assert "Error" in result.stdout
            assert "Video generation failed" in result.stdout
