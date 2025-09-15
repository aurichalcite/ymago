"""
Tests for extended configuration models.

This module tests the new cloud storage and webhook configuration
models added in Milestone 4.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ymago.config import CloudStorageConfig, Settings, WebhookConfig, load_config


class TestCloudStorageConfig:
    """Test cloud storage configuration model."""

    def test_default_values(self):
        """Test cloud storage config with default values."""
        config = CloudStorageConfig()

        assert config.aws_access_key_id is None
        assert config.aws_secret_access_key is None
        assert config.aws_region == "us-east-1"
        assert config.gcp_service_account_path is None
        assert config.r2_account_id is None
        assert config.r2_access_key_id is None
        assert config.r2_secret_access_key is None

    def test_aws_configuration(self):
        """Test AWS S3 configuration."""
        config = CloudStorageConfig(
            aws_access_key_id="AKIATEST123",
            aws_secret_access_key="secret123",
            aws_region="us-west-2",
        )

        assert config.aws_access_key_id == "AKIATEST123"
        assert config.aws_secret_access_key == "secret123"
        assert config.aws_region == "us-west-2"

    def test_gcp_configuration_valid_path(self):
        """Test GCP configuration with valid service account path."""
        # Create a temporary service account file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file.write('{"type": "service_account"}')
            tmp_path = Path(tmp_file.name)

        try:
            config = CloudStorageConfig(gcp_service_account_path=tmp_path)

            assert config.gcp_service_account_path == tmp_path
        finally:
            tmp_path.unlink()

    def test_gcp_configuration_invalid_path(self):
        """Test GCP configuration with invalid service account path."""
        invalid_path = Path("/nonexistent/service-account.json")

        with pytest.raises(ValueError, match="GCP service account file not found"):
            CloudStorageConfig(gcp_service_account_path=invalid_path)

    def test_r2_configuration(self):
        """Test Cloudflare R2 configuration."""
        config = CloudStorageConfig(
            r2_account_id="test-account-id",
            r2_access_key_id="r2-access-key",
            r2_secret_access_key="r2-secret-key",
        )

        assert config.r2_account_id == "test-account-id"
        assert config.r2_access_key_id == "r2-access-key"
        assert config.r2_secret_access_key == "r2-secret-key"


class TestWebhookConfig:
    """Test webhook configuration model."""

    def test_default_values(self):
        """Test webhook config with default values."""
        config = WebhookConfig()

        assert config.enabled is False
        assert config.timeout_seconds == 30
        assert config.retry_attempts == 3
        assert config.retry_backoff_factor == 2.0

    def test_custom_values(self):
        """Test webhook config with custom values."""
        config = WebhookConfig(
            enabled=True, timeout_seconds=60, retry_attempts=5, retry_backoff_factor=1.5
        )

        assert config.enabled is True
        assert config.timeout_seconds == 60
        assert config.retry_attempts == 5
        assert config.retry_backoff_factor == 1.5

    def test_validation_timeout_range(self):
        """Test webhook timeout validation."""
        # Valid timeout
        config = WebhookConfig(timeout_seconds=30)
        assert config.timeout_seconds == 30

        # Invalid timeout (too low)
        with pytest.raises(ValueError):
            WebhookConfig(timeout_seconds=0)

        # Invalid timeout (too high)
        with pytest.raises(ValueError):
            WebhookConfig(timeout_seconds=400)

    def test_validation_retry_attempts_range(self):
        """Test webhook retry attempts validation."""
        # Valid retry attempts
        config = WebhookConfig(retry_attempts=3)
        assert config.retry_attempts == 3

        # Invalid retry attempts (too low)
        with pytest.raises(ValueError):
            WebhookConfig(retry_attempts=0)

        # Invalid retry attempts (too high)
        with pytest.raises(ValueError):
            WebhookConfig(retry_attempts=15)

    def test_validation_backoff_factor_range(self):
        """Test webhook backoff factor validation."""
        # Valid backoff factor
        config = WebhookConfig(retry_backoff_factor=2.0)
        assert config.retry_backoff_factor == 2.0

        # Invalid backoff factor (too low)
        with pytest.raises(ValueError):
            WebhookConfig(retry_backoff_factor=0.5)

        # Invalid backoff factor (too high)
        with pytest.raises(ValueError):
            WebhookConfig(retry_backoff_factor=15.0)


class TestExtendedSettings:
    """Test extended settings model with cloud storage and webhooks."""

    def test_settings_with_cloud_storage_and_webhooks(self):
        """Test settings model with cloud storage and webhook configs."""
        from ymago.config import Auth, CloudStorageConfig, WebhookConfig

        settings = Settings(
            auth=Auth(google_api_key="test-key"),
            cloud_storage=CloudStorageConfig(
                aws_access_key_id="AKIATEST123",
                aws_secret_access_key="secret123",
                aws_region="us-west-2",
            ),
            webhooks=WebhookConfig(enabled=True, timeout_seconds=45, retry_attempts=4),
        )

        assert settings.auth.google_api_key == "test-key"
        assert settings.cloud_storage.aws_access_key_id == "AKIATEST123"
        assert settings.cloud_storage.aws_secret_access_key == "secret123"
        assert settings.cloud_storage.aws_region == "us-west-2"
        assert settings.webhooks.enabled is True
        assert settings.webhooks.timeout_seconds == 45
        assert settings.webhooks.retry_attempts == 4

    def test_settings_with_defaults(self):
        """Test settings model with default cloud storage and webhook configs."""
        from ymago.config import Auth

        settings = Settings(auth=Auth(google_api_key="test-key"))

        # Should have default cloud storage config
        assert settings.cloud_storage.aws_access_key_id is None
        assert settings.cloud_storage.aws_region == "us-east-1"

        # Should have default webhook config
        assert settings.webhooks.enabled is False
        assert settings.webhooks.timeout_seconds == 30


class TestConfigEnvironmentVariables:
    """Test configuration loading with environment variables."""

    @pytest.mark.asyncio
    async def test_aws_env_vars(self):
        """Test loading AWS configuration from environment variables."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "AWS_ACCESS_KEY_ID": "AKIATEST123",
            "AWS_SECRET_ACCESS_KEY": "secret123",
            "AWS_DEFAULT_REGION": "us-west-2",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = await load_config()

            assert config.cloud_storage.aws_access_key_id == "AKIATEST123"
            assert config.cloud_storage.aws_secret_access_key == "secret123"
            assert config.cloud_storage.aws_region == "us-west-2"

    @pytest.mark.asyncio
    async def test_gcp_env_vars(self):
        """Test loading GCP configuration from environment variables."""
        # Create a temporary service account file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file.write('{"type": "service_account"}')
            tmp_path = tmp_file.name

        try:
            env_vars = {
                "GOOGLE_API_KEY": "test-key",
                "GOOGLE_APPLICATION_CREDENTIALS": tmp_path,
            }

            with patch.dict(os.environ, env_vars, clear=True):
                config = await load_config()

                assert str(config.cloud_storage.gcp_service_account_path) == tmp_path
        finally:
            Path(tmp_path).unlink()

    @pytest.mark.asyncio
    async def test_r2_env_vars(self):
        """Test loading R2 configuration from environment variables."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "R2_ACCOUNT_ID": "test-account",
            "R2_ACCESS_KEY_ID": "r2-access-key",
            "R2_SECRET_ACCESS_KEY": "r2-secret-key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = await load_config()

            assert config.cloud_storage.r2_account_id == "test-account"
            assert config.cloud_storage.r2_access_key_id == "r2-access-key"
            assert config.cloud_storage.r2_secret_access_key == "r2-secret-key"

    @pytest.mark.asyncio
    async def test_webhook_env_vars(self):
        """Test loading webhook configuration from environment variables."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "YMAGO_WEBHOOK_ENABLED": "true",
            "YMAGO_WEBHOOK_TIMEOUT": "60",
            "YMAGO_WEBHOOK_RETRIES": "5",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = await load_config()

            assert config.webhooks.enabled is True
            assert config.webhooks.timeout_seconds == 60
            assert config.webhooks.retry_attempts == 5

    @pytest.mark.asyncio
    async def test_webhook_env_vars_false(self):
        """Test loading webhook configuration with false values."""
        env_vars = {"GOOGLE_API_KEY": "test-key", "YMAGO_WEBHOOK_ENABLED": "false"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = await load_config()

            assert config.webhooks.enabled is False

    @pytest.mark.asyncio
    async def test_invalid_webhook_env_vars(self):
        """Test loading webhook configuration with invalid environment values."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "YMAGO_WEBHOOK_TIMEOUT": "invalid",
            "YMAGO_WEBHOOK_RETRIES": "not-a-number",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise an error, just ignore invalid values
            config = await load_config()

            # Should use default values
            assert config.webhooks.timeout_seconds == 30
            assert config.webhooks.retry_attempts == 3
