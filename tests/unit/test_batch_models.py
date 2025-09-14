"""
Unit tests for batch processing data models.

This module tests the new Pydantic models for batch processing:
GenerationRequest, BatchResult, and BatchSummary.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from ymago.models import BatchResult, BatchSummary, GenerationJob, GenerationRequest


class TestGenerationRequest:
    """Test GenerationRequest model functionality."""

    def test_generation_request_basic(self):
        """Test basic GenerationRequest creation."""
        request = GenerationRequest(
            prompt="A beautiful sunset", output_filename="sunset_image"
        )

        assert request.prompt == "A beautiful sunset"
        assert request.output_filename == "sunset_image"
        assert request.media_type == "image"  # default
        assert request.id is not None
        assert len(request.id) > 0

    def test_generation_request_with_all_fields(self):
        """Test GenerationRequest with all optional fields."""
        request = GenerationRequest(
            id="custom-id-123",
            prompt="A mountain landscape",
            media_type="video",
            output_filename="mountain_video",
            seed=42,
            negative_prompt="no buildings",
            from_image="https://example.com/image.jpg",
            quality="high",
            aspect_ratio="16:9",
            image_model="custom-image-model",
            video_model="custom-video-model",
            row_number=5,
        )

        assert request.id == "custom-id-123"
        assert request.prompt == "A mountain landscape"
        assert request.media_type == "video"
        assert request.output_filename == "mountain_video"
        assert request.seed == 42
        assert request.negative_prompt == "no buildings"
        assert request.from_image == "https://example.com/image.jpg"
        assert request.quality == "high"
        assert request.aspect_ratio == "16:9"
        assert request.image_model == "custom-image-model"
        assert request.video_model == "custom-video-model"
        assert request.row_number == 5

    def test_generation_request_auto_id(self):
        """Test automatic ID generation."""
        request1 = GenerationRequest(prompt="Test 1")
        request2 = GenerationRequest(prompt="Test 2")

        assert request1.id != request2.id
        assert len(request1.id) > 10  # Should be a UUID
        assert len(request2.id) > 10

    def test_generation_request_validation_empty_prompt(self):
        """Test validation of empty prompt."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="")

        errors = exc_info.value.errors()
        assert any("at least 1 character" in str(error) for error in errors)

    def test_generation_request_validation_long_prompt(self):
        """Test validation of overly long prompt."""
        long_prompt = "A" * 2001  # Exceeds 2000 character limit

        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt=long_prompt)

        errors = exc_info.value.errors()
        assert any("at most 2000 character" in str(error) for error in errors)

    def test_generation_request_validation_invalid_seed(self):
        """Test validation of invalid seed values."""
        # Seed too low
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test", seed=-2)

        # Seed too high
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test", seed=2**32)

    def test_generation_request_validation_invalid_quality(self):
        """Test validation of invalid quality values."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="Test", quality="invalid")

        errors = exc_info.value.errors()
        assert any("quality" in str(error).lower() for error in errors)

    def test_generation_request_validation_invalid_aspect_ratio(self):
        """Test validation of invalid aspect ratio."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(prompt="Test", aspect_ratio="invalid")

        errors = exc_info.value.errors()
        assert any("aspect_ratio" in str(error).lower() for error in errors)

    def test_generation_request_to_generation_job(self):
        """Test conversion to GenerationJob."""
        request = GenerationRequest(
            prompt="Test prompt",
            media_type="video",
            output_filename="test_video",
            seed=123,
            negative_prompt="no cars",
            quality="high",
            aspect_ratio="16:9",
        )

        job = request.to_generation_job()

        assert isinstance(job, GenerationJob)
        assert job.prompt == "Test prompt"
        assert job.media_type == "video"
        assert job.output_filename == "test_video"
        assert job.seed == 123
        assert job.negative_prompt == "no cars"
        assert job.quality == "high"
        assert job.aspect_ratio == "16:9"


class TestBatchResult:
    """Test BatchResult model functionality."""

    def test_batch_result_success(self):
        """Test BatchResult for successful processing."""
        result = BatchResult(
            request_id="req-123",
            status="success",
            output_path="/path/to/output.png",
            processing_time_seconds=2.5,
            file_size_bytes=1024,
        )

        assert result.request_id == "req-123"
        assert result.status == "success"
        assert result.output_path == "/path/to/output.png"
        assert result.processing_time_seconds == 2.5
        assert result.file_size_bytes == 1024
        assert result.error_message is None
        assert result.timestamp is not None

    def test_batch_result_failure(self):
        """Test BatchResult for failed processing."""
        result = BatchResult(
            request_id="req-456",
            status="failure",
            error_message="Network timeout",
            processing_time_seconds=10.0,
        )

        assert result.request_id == "req-456"
        assert result.status == "failure"
        assert result.error_message == "Network timeout"
        assert result.processing_time_seconds == 10.0
        assert result.output_path is None
        assert result.file_size_bytes is None

    def test_batch_result_skipped(self):
        """Test BatchResult for skipped processing."""
        result = BatchResult(request_id="req-789", status="skipped")

        assert result.request_id == "req-789"
        assert result.status == "skipped"
        assert result.output_path is None
        assert result.error_message is None
        assert result.processing_time_seconds is None

    def test_batch_result_auto_timestamp(self):
        """Test automatic timestamp generation."""
        result1 = BatchResult(request_id="req1", status="success")
        result2 = BatchResult(request_id="req2", status="success")

        assert result1.timestamp != result2.timestamp
        # Should be valid ISO format
        datetime.fromisoformat(result1.timestamp.replace("Z", "+00:00"))

    def test_batch_result_validation_negative_time(self):
        """Test validation of negative processing time."""
        with pytest.raises(ValidationError):
            BatchResult(
                request_id="req", status="success", processing_time_seconds=-1.0
            )

    def test_batch_result_validation_negative_file_size(self):
        """Test validation of negative file size."""
        with pytest.raises(ValidationError):
            BatchResult(request_id="req", status="success", file_size_bytes=-100)

    def test_batch_result_with_metadata(self):
        """Test BatchResult with metadata."""
        metadata = {"model": "test-model", "version": "1.0"}
        result = BatchResult(request_id="req", status="success", metadata=metadata)

        assert result.metadata == metadata


class TestBatchSummary:
    """Test BatchSummary model functionality."""

    def test_batch_summary_basic(self):
        """Test basic BatchSummary creation."""
        summary = BatchSummary(
            total_requests=100,
            successful=85,
            failed=10,
            skipped=5,
            processing_time_seconds=300.0,
            results_log_path="/path/to/results.jsonl",
            throughput_requests_per_minute=20.0,
        )

        assert summary.total_requests == 100
        assert summary.successful == 85
        assert summary.failed == 10
        assert summary.skipped == 5
        assert summary.processing_time_seconds == 300.0
        assert summary.results_log_path == "/path/to/results.jsonl"
        assert summary.throughput_requests_per_minute == 20.0
        assert summary.rejected_rows_path is None

    def test_batch_summary_with_rejected_rows(self):
        """Test BatchSummary with rejected rows file."""
        summary = BatchSummary(
            total_requests=50,
            successful=40,
            failed=5,
            skipped=5,
            processing_time_seconds=150.0,
            results_log_path="/path/to/results.jsonl",
            rejected_rows_path="/path/to/rejected.csv",
            throughput_requests_per_minute=20.0,
        )

        assert summary.rejected_rows_path == "/path/to/rejected.csv"

    def test_batch_summary_success_rate(self):
        """Test success rate calculation."""
        summary = BatchSummary(
            total_requests=100,
            successful=75,
            failed=20,
            skipped=5,
            processing_time_seconds=300.0,
            results_log_path="/path/to/results.jsonl",
            throughput_requests_per_minute=20.0,
        )

        assert summary.success_rate == 75.0

    def test_batch_summary_success_rate_zero_requests(self):
        """Test success rate with zero requests."""
        summary = BatchSummary(
            total_requests=0,
            successful=0,
            failed=0,
            skipped=0,
            processing_time_seconds=0.0,
            results_log_path="/path/to/results.jsonl",
            throughput_requests_per_minute=0.0,
        )

        assert summary.success_rate == 0.0

    def test_batch_summary_validation_negative_counts(self):
        """Test validation of negative count values."""
        with pytest.raises(ValidationError):
            BatchSummary(
                total_requests=10,
                successful=-1,  # Invalid
                failed=5,
                skipped=5,
                processing_time_seconds=100.0,
                results_log_path="/path/to/results.jsonl",
                throughput_requests_per_minute=6.0,
            )

    def test_batch_summary_validation_negative_time(self):
        """Test validation of negative processing time."""
        with pytest.raises(ValidationError):
            BatchSummary(
                total_requests=10,
                successful=5,
                failed=3,
                skipped=2,
                processing_time_seconds=-10.0,  # Invalid
                results_log_path="/path/to/results.jsonl",
                throughput_requests_per_minute=6.0,
            )

    def test_batch_summary_validation_negative_throughput(self):
        """Test validation of negative throughput."""
        with pytest.raises(ValidationError):
            BatchSummary(
                total_requests=10,
                successful=5,
                failed=3,
                skipped=2,
                processing_time_seconds=100.0,
                results_log_path="/path/to/results.jsonl",
                throughput_requests_per_minute=-5.0,  # Invalid
            )

    def test_batch_summary_auto_timestamps(self):
        """Test automatic timestamp generation."""
        summary1 = BatchSummary(
            total_requests=10,
            successful=5,
            failed=3,
            skipped=2,
            processing_time_seconds=100.0,
            results_log_path="/path/to/results.jsonl",
            throughput_requests_per_minute=6.0,
        )

        summary2 = BatchSummary(
            total_requests=20,
            successful=15,
            failed=3,
            skipped=2,
            processing_time_seconds=200.0,
            results_log_path="/path/to/results2.jsonl",
            throughput_requests_per_minute=6.0,
        )

        assert summary1.start_time != summary2.start_time
        assert summary1.end_time != summary2.end_time

        # Should be valid ISO format
        datetime.fromisoformat(summary1.start_time.replace("Z", "+00:00"))
        datetime.fromisoformat(summary1.end_time.replace("Z", "+00:00"))
