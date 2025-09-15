"""
Cloud storage backend implementations for ymago package.

This module provides concrete implementations of the StorageUploader interface
for various cloud storage providers including AWS S3, Google Cloud Storage,
and Cloudflare R2.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiofiles
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .storage import StorageBackendRegistry, StorageError, StorageUploader

logger = logging.getLogger(__name__)

try:
    import aioboto3
except ImportError:
    aioboto3 = None

try:
    from gcloud.aio.storage import Storage
except ImportError:
    Storage = None


class S3StorageBackend(StorageUploader):
    """
    AWS S3 storage backend implementation.

    This implementation uses aioboto3 for async S3 operations with built-in
    retry logic and proper error handling.
    """

    def __init__(
        self,
        destination_url: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
    ):
        """
        Initialize S3 storage backend.

        Args:
            destination_url: S3 URL (e.g., 's3://bucket-name/path/')
            aws_access_key_id: AWS access key (optional, uses IAM if not provided)
            aws_secret_access_key: AWS secret key (optional, uses IAM if not provided)
            aws_region: AWS region
        """
        super().__init__(destination_url=destination_url)
        if aioboto3 is None:
            raise ImportError(
                "AWS S3 support requires 'aioboto3'. "
                "Install with: pip install 'ymago[aws]'"
            )

        parsed = urlparse(destination_url)
        if parsed.scheme != "s3":
            raise ValueError(
                f"S3StorageBackend only supports 's3://' URLs, got: {parsed.scheme}"
            )

        self.bucket_name = parsed.netloc
        self.base_path = parsed.path.lstrip("/")
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region

        if not self.bucket_name:
            raise ValueError("S3 URL must include bucket name: s3://bucket-name/path/")

    def _get_session_kwargs(self) -> Dict[str, Any]:
        """Get session configuration for aioboto3."""
        kwargs: Dict[str, Any] = {"region_name": self.aws_region}

        if self.aws_access_key_id and self.aws_secret_access_key:
            kwargs.update(
                {
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key,
                }
            )

        return kwargs

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload(self, file_path: Path, destination_key: str) -> str:
        """
        Upload a file to S3.

        Args:
            file_path: Local path to the file to upload
            destination_key: S3 key for the uploaded file

        Returns:
            str: S3 URL of the uploaded file
        """
        if aioboto3 is None:
            raise StorageError("aioboto3 not available")

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"

        # Construct full S3 key
        s3_key = (Path(self.base_path) / destination_key).as_posix()

        session = aioboto3.Session(**self._get_session_kwargs())

        try:
            async with session.client("s3") as s3_client:
                # Upload file with streaming
                await s3_client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={"ContentType": content_type},
                )

                # Return the S3 URL
                return f"s3://{self.bucket_name}/{s3_key}"

        except Exception as e:
            raise StorageError(f"Failed to upload to S3: {e}") from e

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload_bytes(
        self, data: bytes, destination_key: str, content_type: str
    ) -> str:
        """
        Upload raw bytes to S3.

        Args:
            data: Raw bytes to upload
            destination_key: S3 key for the uploaded file
            content_type: MIME type of the content

        Returns:
            str: S3 URL of the uploaded file
        """
        if aioboto3 is None:
            raise StorageError("aioboto3 not available")

        # Construct full S3 key
        s3_key = (Path(self.base_path) / destination_key).as_posix()

        session = aioboto3.Session(**self._get_session_kwargs())

        try:
            async with session.client("s3") as s3_client:
                # Upload bytes
                await s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=data,
                    ContentType=content_type,
                )

                # Return the S3 URL
                return f"s3://{self.bucket_name}/{s3_key}"

        except Exception as e:
            raise StorageError(f"Failed to upload bytes to S3: {e}") from e

    async def exists(self, destination_key: str) -> bool:
        """Check if a file exists in S3."""
        s3_key = (Path(self.base_path) / destination_key).as_posix()

        if aioboto3 is None:
            return False

        session = aioboto3.Session(**self._get_session_kwargs())

        try:
            async with session.client("s3") as s3_client:
                await s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                return True
        except Exception:
            return False

    async def delete(self, destination_key: str) -> bool:
        """Delete a file from S3."""
        s3_key = (Path(self.base_path) / destination_key).as_posix()

        if aioboto3 is None:
            return False

        session = aioboto3.Session(**self._get_session_kwargs())

        try:
            async with session.client("s3") as s3_client:
                await s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
                return True
        except Exception:
            return False


class R2StorageBackend(S3StorageBackend):
    """
    Cloudflare R2 storage backend implementation.

    R2 is S3-compatible, so we inherit from S3StorageBackend and override
    the endpoint configuration.
    """

    def __init__(
        self,
        destination_url: str,
        r2_account_id: str,
        r2_access_key_id: str,
        r2_secret_access_key: str,
    ):
        """
        Initialize R2 storage backend.

        Args:
            destination_url: R2 URL (e.g., 'r2://bucket-name/path/')
            r2_account_id: Cloudflare R2 account ID
            r2_access_key_id: R2 access key ID
            r2_secret_access_key: R2 secret access key
        """
        # Parse R2 URL and convert to S3 format for parent class
        parsed = urlparse(destination_url)
        if parsed.scheme != "r2":
            raise ValueError(
                f"R2StorageBackend only supports 'r2://' URLs, got: {parsed.scheme}"
            )

        # Convert to S3 URL format for parent class
        s3_url = f"s3://{parsed.netloc}{parsed.path}"

        # Initialize parent with R2 credentials
        super().__init__(
            destination_url=s3_url,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            aws_region="auto",  # R2 uses "auto" region
        )

        self.r2_account_id = r2_account_id
        self.endpoint_url = f"https://{r2_account_id}.r2.cloudflarestorage.com"

    def _get_session_kwargs(self) -> Dict[str, Any]:
        """Get session configuration for R2."""
        kwargs = super()._get_session_kwargs()
        kwargs["endpoint_url"] = self.endpoint_url
        return kwargs


class GCSStorageBackend(StorageUploader):
    """
    Google Cloud Storage backend implementation.

    This implementation uses gcloud-aio-storage for native async GCS operations
    with built-in retry logic and proper error handling.
    """

    def __init__(
        self,
        destination_url: str,
        service_account_path: Optional[Path] = None,
        **kwargs: Any,
    ):
        """
        Initialize GCS storage backend.

        Args:
            destination_url: GCS URL (e.g., 'gs://bucket-name/path/')
            service_account_path: Path to service account JSON file (optional)
        """
        super().__init__(destination_url=destination_url, **kwargs)
        if Storage is None:
            raise ImportError(
                "Google Cloud Storage support requires 'gcloud-aio-storage'. "
                "Install with: pip install 'ymago[gcp]'"
            )

        parsed = urlparse(destination_url)
        if parsed.scheme != "gs":
            raise ValueError(
                f"GCSStorageBackend only supports 'gs://' URLs, got: {parsed.scheme}"
            )

        self.bucket_name = parsed.netloc
        self.base_path = parsed.path.lstrip("/")
        self.service_account_path = service_account_path

        if not self.bucket_name:
            raise ValueError("GCS URL must include bucket name: gs://bucket-name/path/")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload(self, file_path: Path, destination_key: str) -> str:
        """
        Upload a file to GCS.

        Args:
            file_path: Local path to the file to upload
            destination_key: GCS object name for the uploaded file

        Returns:
            str: GCS URL of the uploaded file
        """
        if Storage is None:
            raise StorageError("gcloud-aio-storage not available")

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"

        # Construct full GCS object name
        object_name = (Path(self.base_path) / destination_key).as_posix()

        # Read file data
        async with aiofiles.open(file_path, "rb") as f:
            data = await f.read()

        try:
            # Initialize storage client
            storage_kwargs: Dict[str, Any] = {}
            if self.service_account_path:
                storage_kwargs["service_file"] = str(self.service_account_path)

            async with Storage(**storage_kwargs) as storage:
                # Upload file
                await storage.upload(
                    bucket=self.bucket_name,
                    object_name=object_name,
                    file_data=data,
                    content_type=content_type,
                )

                # Return the GCS URL
                return f"gs://{self.bucket_name}/{object_name}"

        except Exception as e:
            raise StorageError(f"Failed to upload to GCS: {e}") from e

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def upload_bytes(
        self, data: bytes, destination_key: str, content_type: str
    ) -> str:
        """
        Upload raw bytes to GCS.

        Args:
            data: Raw bytes to upload
            destination_key: GCS object name for the uploaded file
            content_type: MIME type of the content

        Returns:
            str: GCS URL of the uploaded file
        """
        if Storage is None:
            raise StorageError("gcloud-aio-storage not available")

        # Construct full GCS object name
        object_name = (Path(self.base_path) / destination_key).as_posix()

        try:
            # Initialize storage client
            storage_kwargs: Dict[str, Any] = {}
            if self.service_account_path:
                storage_kwargs["service_file"] = str(self.service_account_path)

            async with Storage(**storage_kwargs) as storage:
                # Upload bytes
                await storage.upload(
                    bucket=self.bucket_name,
                    object_name=object_name,
                    file_data=data,
                    content_type=content_type,
                )

                # Return the GCS URL
                return f"gs://{self.bucket_name}/{object_name}"

        except Exception as e:
            raise StorageError(f"Failed to upload bytes to GCS: {e}") from e

    async def exists(self, destination_key: str) -> bool:
        """Check if a file exists in GCS."""
        object_name = (Path(self.base_path) / destination_key).as_posix()

        if Storage is None:
            return False

        try:
            storage_kwargs: Dict[str, Any] = {}
            if self.service_account_path:
                storage_kwargs["service_file"] = str(self.service_account_path)

            async with Storage(**storage_kwargs) as storage:
                # Try to get object metadata
                await storage.download_metadata(
                    bucket=self.bucket_name, object_name=object_name
                )
                return True
        except Exception:
            return False

    async def delete(self, destination_key: str) -> bool:
        """Delete a file from GCS."""
        object_name = (Path(self.base_path) / destination_key).as_posix()

        if Storage is None:
            return False

        try:
            storage_kwargs: Dict[str, Any] = {}
            if self.service_account_path:
                storage_kwargs["service_file"] = str(self.service_account_path)

            async with Storage(**storage_kwargs) as storage:
                await storage.delete(bucket=self.bucket_name, object_name=object_name)
                return True
        except Exception:
            return False


# Register cloud storage backends
StorageBackendRegistry.register("s3", S3StorageBackend)
StorageBackendRegistry.register("gs", GCSStorageBackend)
StorageBackendRegistry.register("r2", R2StorageBackend)
