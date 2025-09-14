"""
Tests for cloud storage backend implementations.

This module tests the S3, GCS, and R2 storage backends with mocked
network calls to ensure they work correctly without making actual
cloud API calls.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ymago.core.cloud_storage import (
    GCSStorageBackend,
    R2StorageBackend,
    S3StorageBackend,
)
from ymago.core.storage import StorageBackendRegistry, StorageError


class TestS3StorageBackend:
    """Test S3 storage backend implementation."""

    def test_init_valid_url(self):
        """Test S3 backend initialization with valid URL."""
        backend = S3StorageBackend(
            destination_url="s3://test-bucket/path/",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_region="us-west-2",
        )

        assert backend.bucket_name == "test-bucket"
        assert backend.base_path == "path/"
        assert backend.aws_access_key_id == "test-key"
        assert backend.aws_secret_access_key == "test-secret"
        assert backend.aws_region == "us-west-2"

    def test_init_invalid_scheme(self):
        """Test S3 backend initialization with invalid URL scheme."""
        with pytest.raises(
            ValueError, match="S3StorageBackend only supports 's3://' URLs"
        ):
            S3StorageBackend(destination_url="gs://test-bucket/path/")

    def test_init_missing_bucket(self):
        """Test S3 backend initialization with missing bucket name."""
        with pytest.raises(ValueError, match="S3 URL must include bucket name"):
            S3StorageBackend(destination_url="s3:///path/")

    def test_init_missing_aioboto3(self):
        """Test S3 backend initialization when aioboto3 is not available."""
        with patch.dict("sys.modules", {"aioboto3": None}):
            with pytest.raises(ImportError, match="AWS S3 support requires 'aioboto3'"):
                S3StorageBackend(destination_url="s3://test-bucket/path/")

    @pytest.mark.asyncio
    async def test_upload_file_success(self):
        """Test successful file upload to S3."""
        with patch("ymago.core.cloud_storage.aioboto3") as mock_aioboto3:
            # Setup mocks
            mock_session = AsyncMock()
            mock_client = AsyncMock()
            mock_session.client.return_value.__aenter__.return_value = mock_client
            mock_aioboto3.Session.return_value = mock_session

            backend = S3StorageBackend(
                destination_url="s3://test-bucket/uploads/",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
            )

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(b"test image data")
                tmp_path = Path(tmp_file.name)

            try:
                # Test upload
                result = await backend.upload(tmp_path, "test-image.jpg")

                # Verify result
                assert result == "s3://test-bucket/uploads/test-image.jpg"

                # Verify S3 client was called correctly
                mock_client.upload_file.assert_called_once()
                call_args = mock_client.upload_file.call_args
                assert call_args[0][0] == str(tmp_path)  # source file
                assert call_args[0][1] == "test-bucket"  # bucket
                assert call_args[0][2] == "uploads/test-image.jpg"  # key
                assert "ContentType" in call_args[1]["ExtraArgs"]

            finally:
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_upload_bytes_success(self):
        """Test successful bytes upload to S3."""
        with patch("ymago.core.cloud_storage.aioboto3") as mock_aioboto3:
            # Setup mocks
            mock_session = AsyncMock()
            mock_client = AsyncMock()
            mock_session.client.return_value.__aenter__.return_value = mock_client
            mock_aioboto3.Session.return_value = mock_session

            backend = S3StorageBackend(
                destination_url="s3://test-bucket/uploads/",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
            )

            # Test upload
            test_data = b"test image data"
            result = await backend.upload_bytes(
                test_data, "test-image.jpg", "image/jpeg"
            )

            # Verify result
            assert result == "s3://test-bucket/uploads/test-image.jpg"

            # Verify S3 client was called correctly
            mock_client.put_object.assert_called_once_with(
                Bucket="test-bucket",
                Key="uploads/test-image.jpg",
                Body=test_data,
                ContentType="image/jpeg",
            )

    @pytest.mark.asyncio
    async def test_upload_with_s3_error(self):
        """Test upload failure due to S3 error."""
        with patch("ymago.core.cloud_storage.aioboto3") as mock_aioboto3:
            # Setup mocks to raise an exception
            mock_session = AsyncMock()
            mock_client = AsyncMock()
            mock_client.upload_file.side_effect = Exception("S3 error")
            mock_session.client.return_value.__aenter__.return_value = mock_client
            mock_aioboto3.Session.return_value = mock_session

            backend = S3StorageBackend(
                destination_url="s3://test-bucket/uploads/",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
            )

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(b"test image data")
                tmp_path = Path(tmp_file.name)

            try:
                # Test upload failure
                with pytest.raises(StorageError, match="Failed to upload to S3"):
                    await backend.upload(tmp_path, "test-image.jpg")

            finally:
                tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Test exists method when file exists."""
        with patch("ymago.core.cloud_storage.aioboto3") as mock_aioboto3:
            # Setup mocks
            mock_session = AsyncMock()
            mock_client = AsyncMock()
            mock_session.client.return_value.__aenter__.return_value = mock_client
            mock_aioboto3.Session.return_value = mock_session

            backend = S3StorageBackend(
                destination_url="s3://test-bucket/uploads/",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
            )

            # Test exists
            result = await backend.exists("test-image.jpg")

            assert result is True
            mock_client.head_object.assert_called_once_with(
                Bucket="test-bucket", Key="uploads/test-image.jpg"
            )

    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Test exists method when file doesn't exist."""
        with patch("ymago.core.cloud_storage.aioboto3") as mock_aioboto3:
            # Setup mocks to raise an exception (file not found)
            mock_session = AsyncMock()
            mock_client = AsyncMock()
            mock_client.head_object.side_effect = Exception("Not found")
            mock_session.client.return_value.__aenter__.return_value = mock_client
            mock_aioboto3.Session.return_value = mock_session

            backend = S3StorageBackend(
                destination_url="s3://test-bucket/uploads/",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
            )

            # Test exists
            result = await backend.exists("test-image.jpg")

            assert result is False


class TestGCSStorageBackend:
    """Test Google Cloud Storage backend implementation."""

    def test_init_valid_url(self):
        """Test GCS backend initialization with valid URL."""
        backend = GCSStorageBackend(
            destination_url="gs://test-bucket/path/", service_account_path=None
        )

        assert backend.bucket_name == "test-bucket"
        assert backend.base_path == "path/"
        assert backend.service_account_path is None

    def test_init_invalid_scheme(self):
        """Test GCS backend initialization with invalid URL scheme."""
        with pytest.raises(
            ValueError, match="GCSStorageBackend only supports 'gs://' URLs"
        ):
            GCSStorageBackend(destination_url="s3://test-bucket/path/")

    def test_init_missing_bucket(self):
        """Test GCS backend initialization with missing bucket name."""
        with pytest.raises(ValueError, match="GCS URL must include bucket name"):
            GCSStorageBackend(destination_url="gs:///path/")

    def test_init_missing_gcloud_aio(self):
        """Test GCS backend initialization when gcloud-aio-storage is not available."""
        with patch.dict("sys.modules", {"gcloud.aio.storage": None}):
            with pytest.raises(
                ImportError, match="Google Cloud Storage support requires"
            ):
                GCSStorageBackend(destination_url="gs://test-bucket/path/")

    @pytest.mark.asyncio
    async def test_upload_bytes_success(self):
        """Test successful bytes upload to GCS."""
        with patch("ymago.core.cloud_storage.Storage") as mock_storage_class:
            # Setup mocks
            mock_storage = AsyncMock()
            mock_storage_class.return_value.__aenter__.return_value = mock_storage

            backend = GCSStorageBackend(
                destination_url="gs://test-bucket/uploads/", service_account_path=None
            )

            # Test upload
            test_data = b"test image data"
            result = await backend.upload_bytes(
                test_data, "test-image.jpg", "image/jpeg"
            )

            # Verify result
            assert result == "gs://test-bucket/uploads/test-image.jpg"

            # Verify GCS client was called correctly
            mock_storage.upload.assert_called_once_with(
                bucket="test-bucket",
                object_name="uploads/test-image.jpg",
                file_data=test_data,
                content_type="image/jpeg",
            )


class TestR2StorageBackend:
    """Test Cloudflare R2 storage backend implementation."""

    def test_init_valid_url(self):
        """Test R2 backend initialization with valid URL."""
        backend = R2StorageBackend(
            destination_url="r2://test-bucket/path/",
            r2_account_id="test-account",
            r2_access_key_id="test-key",
            r2_secret_access_key="test-secret",
        )

        assert backend.bucket_name == "test-bucket"
        assert backend.base_path == "path/"
        assert backend.r2_account_id == "test-account"
        assert backend.endpoint_url == "https://test-account.r2.cloudflarestorage.com"

    def test_init_invalid_scheme(self):
        """Test R2 backend initialization with invalid URL scheme."""
        with pytest.raises(
            ValueError, match="R2StorageBackend only supports 'r2://' URLs"
        ):
            R2StorageBackend(
                destination_url="s3://test-bucket/path/",
                r2_account_id="test-account",
                r2_access_key_id="test-key",
                r2_secret_access_key="test-secret",
            )


class TestStorageBackendRegistry:
    """Test storage backend registry functionality."""

    def test_registry_has_backends(self):
        """Test that registry has expected backends registered."""
        schemes = StorageBackendRegistry.list_schemes()

        # Should have at least file, s3, gs, r2
        assert "file" in schemes
        assert "s3" in schemes
        assert "gs" in schemes
        assert "r2" in schemes

    def test_create_backend_s3(self):
        """Test creating S3 backend through registry."""
        backend = StorageBackendRegistry.create_backend(
            "s3://test-bucket/path/",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
        )

        assert isinstance(backend, S3StorageBackend)
        assert backend.bucket_name == "test-bucket"

    def test_create_backend_unsupported_scheme(self):
        """Test creating backend with unsupported scheme."""
        with pytest.raises(ValueError, match="Unsupported storage scheme 'ftp'"):
            StorageBackendRegistry.create_backend("ftp://example.com/path/")
