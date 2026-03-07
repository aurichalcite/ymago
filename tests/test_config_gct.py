"""
Tests for Google Cloud Tasks configuration.
"""

import os
import pytest
from ymago.config import Settings, load_config
from unittest.mock import patch

try:
    from ymago.config import CloudTasksConfig
except ImportError:
    CloudTasksConfig = None


class TestCloudTasksConfig:
    """Test Google Cloud Tasks configuration model."""

    def test_cloud_tasks_config_exists(self):
        """Test that CloudTasksConfig is defined."""
        assert CloudTasksConfig is not None

    def test_default_values(self):
        """Test cloud tasks config with default values."""
        config = CloudTasksConfig()
        assert config.gct_project_id is None
        assert config.gct_location == "us-central1"
        assert config.gct_queue_name == "ymago-tasks"
        assert config.worker_url is None
        assert config.service_account_email is None

    def test_custom_values(self):
        """Test cloud tasks config with custom values."""
        config = CloudTasksConfig(
            gct_project_id="test-project",
            gct_location="europe-west1",
            gct_queue_name="custom-queue",
            worker_url="https://worker.example.com",
            service_account_email="worker@test-project.iam.gserviceaccount.com"
        )
        assert config.gct_project_id == "test-project"
        assert config.gct_location == "europe-west1"
        assert config.gct_queue_name == "custom-queue"
        assert config.worker_url == "https://worker.example.com"
        assert config.service_account_email == "worker@test-project.iam.gserviceaccount.com"


class TestGCTSettings:
    """Test settings model with Google Cloud Tasks."""

    def test_settings_has_cloud_tasks(self):
        """Test that Settings model has a cloud_tasks field."""
        from ymago.config import Auth
        settings = Settings(auth=Auth(google_api_key="test-key"))
        assert hasattr(settings, "cloud_tasks")
        assert isinstance(settings.cloud_tasks, CloudTasksConfig)


class TestGCTConfigEnvironmentVariables:
    """Test configuration loading with environment variables for GCT."""

    @pytest.mark.asyncio
    async def test_gct_env_vars(self):
        """Test loading GCT configuration from environment variables."""
        env_vars = {
            "GOOGLE_API_KEY": "test-key",
            "GCT_PROJECT_ID": "env-project",
            "GCT_LOCATION": "env-location",
            "GCT_QUEUE_NAME": "env-queue",
            "GCT_WORKER_URL": "https://env-worker.com",
            "GCT_SERVICE_ACCOUNT_EMAIL": "env-sa@example.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = await load_config()
            assert config.cloud_tasks.gct_project_id == "env-project"
            assert config.cloud_tasks.gct_location == "env-location"
            assert config.cloud_tasks.gct_queue_name == "env-queue"
            assert config.cloud_tasks.worker_url == "https://env-worker.com"
            assert config.cloud_tasks.service_account_email == "env-sa@example.com"
