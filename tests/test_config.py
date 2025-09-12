"""
Tests for ymago configuration management.

This module tests the configuration loading, TOML parsing, environment variable
handling, and Pydantic model validation.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import tomli
from pydantic import ValidationError

from ymago.config import Auth, Defaults, Settings, load_config


class TestAuthModel:
    """Test the Auth Pydantic model."""

    def test_auth_valid_api_key(self):
        """Test Auth model with valid API key."""
        auth = Auth(google_api_key="valid_api_key_123")
        assert auth.google_api_key == "valid_api_key_123"

    def test_auth_strips_whitespace(self):
        """Test Auth model strips whitespace from API key."""
        auth = Auth(google_api_key="  api_key_with_spaces  ")
        assert auth.google_api_key == "api_key_with_spaces"

    def test_auth_empty_api_key_raises_error(self):
        """Test Auth model raises error for empty API key."""
        with pytest.raises(ValidationError, match="Google API key cannot be empty"):
            Auth(google_api_key="")

    def test_auth_whitespace_only_api_key_raises_error(self):
        """Test Auth model raises error for whitespace-only API key."""
        with pytest.raises(ValidationError, match="Google API key cannot be empty"):
            Auth(google_api_key="   ")


class TestDefaultsModel:
    """Test the Defaults Pydantic model."""

    def test_defaults_with_valid_values(self):
        """Test Defaults model with valid values."""
        defaults = Defaults(image_model="test-model", output_path=Path("/test/path"))
        assert defaults.image_model == "test-model"
        assert defaults.output_path == Path("/test/path").resolve()

    def test_defaults_with_default_values(self):
        """Test Defaults model uses correct default values."""
        defaults = Defaults()
        assert defaults.image_model == "gemini-2.5-flash-image-preview"
        assert defaults.video_model == "veo-3.0-generate-001"
        assert defaults.output_path == (Path.cwd() / "generated_media").resolve()
        assert defaults.enable_metadata is True

    def test_defaults_path_resolution(self):
        """Test Defaults model resolves relative paths to absolute."""
        defaults = Defaults(output_path=Path("./relative/path"))
        assert defaults.output_path.is_absolute()


class TestSettingsModel:
    """Test the Settings Pydantic model."""

    def test_settings_with_valid_data(self, sample_auth, sample_defaults):
        """Test Settings model with valid auth and defaults."""
        settings = Settings(auth=sample_auth, defaults=sample_defaults)
        assert settings.auth == sample_auth
        assert settings.defaults == sample_defaults

    def test_settings_with_default_defaults(self, sample_auth):
        """Test Settings model creates default Defaults when not provided."""
        settings = Settings(auth=sample_auth)
        assert settings.auth == sample_auth
        assert isinstance(settings.defaults, Defaults)


class TestLoadConfig:
    """Test the load_config async function."""

    @pytest.mark.asyncio
    async def test_load_config_from_current_directory_toml(self, mock_toml_config):
        """Test loading configuration from ./ymago.toml file."""
        with (
            patch("ymago.config.Path.cwd") as mock_cwd,
            patch("ymago.config.Path.home") as mock_home,
            patch("builtins.open", mock_open()),
            patch("ymago.config.tomli.load") as mock_tomli_load,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # Mock path creation and existence
            mock_current_path = MagicMock()
            mock_current_path.exists.return_value = True
            mock_cwd.return_value.__truediv__.return_value = mock_current_path

            mock_home_path = MagicMock()
            mock_home_path.exists.return_value = False
            mock_home.return_value.__truediv__.return_value = mock_home_path
            mock_tomli_load.return_value = mock_toml_config
            mock_getenv.return_value = None

            config = await load_config()

            assert config.auth.google_api_key == "toml_api_key_67890"
            assert config.defaults.image_model == "gemini-2.5-flash-image-preview"
            assert str(config.defaults.output_path) == "/home/user/images"

    @pytest.mark.asyncio
    async def test_load_config_from_home_directory_toml(self, mock_toml_config):
        """Test loading configuration from ~/.ymago.toml file."""
        with (
            patch("ymago.config.Path.cwd") as mock_cwd,
            patch("ymago.config.Path.home") as mock_home,
            patch("builtins.open", mock_open()),
            patch("ymago.config.tomli.load") as mock_tomli_load,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # Mock path creation and existence - only home directory file exists
            mock_current_path = MagicMock()
            mock_current_path.exists.return_value = False
            mock_cwd.return_value.__truediv__.return_value = mock_current_path

            mock_home_path = MagicMock()
            mock_home_path.exists.return_value = True
            mock_home.return_value.__truediv__.return_value = mock_home_path
            mock_tomli_load.return_value = mock_toml_config
            mock_getenv.return_value = None

            config = await load_config()

            assert config.auth.google_api_key == "toml_api_key_67890"

    @pytest.mark.asyncio
    async def test_load_config_environment_variable_override(self, mock_toml_config):
        """Test environment variables override TOML file values."""
        with (
            patch("ymago.config.Path.cwd") as mock_cwd,
            patch("ymago.config.Path.home") as mock_home,
            patch("builtins.open", mock_open()),
            patch("ymago.config.tomli.load") as mock_tomli_load,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # Mock path creation and existence
            mock_current_path = MagicMock()
            mock_current_path.exists.return_value = True
            mock_cwd.return_value.__truediv__.return_value = mock_current_path

            mock_home_path = MagicMock()
            mock_home_path.exists.return_value = False
            mock_home.return_value.__truediv__.return_value = mock_home_path
            mock_tomli_load.return_value = mock_toml_config

            # Mock environment variables
            def getenv_side_effect(key):
                env_vars = {
                    "GOOGLE_API_KEY": "env_override_key",
                    "YMAGO_OUTPUT_PATH": "/env/override/path",
                    "YMAGO_IMAGE_MODEL": "env-override-model",
                }
                return env_vars.get(key)

            mock_getenv.side_effect = getenv_side_effect

            config = await load_config()

            # Environment variables should override TOML values
            assert config.auth.google_api_key == "env_override_key"
            assert str(config.defaults.output_path) == "/env/override/path"
            assert config.defaults.image_model == "env-override-model"

    @pytest.mark.asyncio
    async def test_load_config_environment_only(self):
        """Test loading configuration from environment variables only."""
        with (
            patch("ymago.config.Path.exists") as mock_exists,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # No config files exist
            mock_exists.return_value = False

            # Mock environment variables
            def getenv_side_effect(key):
                env_vars = {
                    "GOOGLE_API_KEY": "env_only_key",
                    "YMAGO_OUTPUT_PATH": "/env/only/path",
                }
                return env_vars.get(key)

            mock_getenv.side_effect = getenv_side_effect

            config = await load_config()

            assert config.auth.google_api_key == "env_only_key"
            assert str(config.defaults.output_path) == "/env/only/path"

    @pytest.mark.asyncio
    async def test_load_config_missing_configuration_raises_error(self):
        """Test FileNotFoundError when no config file or env vars exist."""
        with (
            patch("ymago.config.Path.exists") as mock_exists,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # No config files exist and no environment variables
            mock_exists.return_value = False
            mock_getenv.return_value = None

            with pytest.raises(
                FileNotFoundError,
                match=(
                    "No configuration file found.*"
                    "GOOGLE_API_KEY environment variable is not set"
                ),
            ):
                await load_config()

    @pytest.mark.asyncio
    async def test_load_config_invalid_toml_raises_error(self):
        """Test ValueError when TOML file is malformed."""
        with (
            patch("ymago.config.Path.cwd") as mock_cwd,
            patch("ymago.config.Path.home") as mock_home,
            patch("builtins.open", mock_open()),
            patch("ymago.config.tomli.load") as mock_tomli_load,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # Mock path creation and existence
            mock_current_path = MagicMock()
            mock_current_path.exists.return_value = True
            mock_cwd.return_value.__truediv__.return_value = mock_current_path

            mock_home_path = MagicMock()
            mock_home_path.exists.return_value = False
            mock_home.return_value.__truediv__.return_value = mock_home_path
            mock_tomli_load.side_effect = tomli.TOMLDecodeError(
                msg="Invalid TOML", doc="[invalid", pos=1
            )
            mock_getenv.return_value = None

            with pytest.raises(ValueError, match="Invalid TOML syntax"):
                await load_config()

    @pytest.mark.asyncio
    async def test_load_config_validation_error(self):
        """Test ValueError when configuration validation fails."""
        invalid_config = {"auth": {"google_api_key": ""}}  # Empty API key

        with (
            patch("ymago.config.Path.cwd") as mock_cwd,
            patch("ymago.config.Path.home") as mock_home,
            patch("builtins.open", mock_open()),
            patch("ymago.config.tomli.load") as mock_tomli_load,
            patch("ymago.config.os.getenv") as mock_getenv,
        ):
            # Mock path creation and existence
            mock_current_path = MagicMock()
            mock_current_path.exists.return_value = True
            mock_cwd.return_value.__truediv__.return_value = mock_current_path

            mock_home_path = MagicMock()
            mock_home_path.exists.return_value = False
            mock_home.return_value.__truediv__.return_value = mock_home_path
            mock_tomli_load.return_value = invalid_config
            mock_getenv.return_value = None

            with pytest.raises(ValueError, match="Configuration validation failed"):
                await load_config()
