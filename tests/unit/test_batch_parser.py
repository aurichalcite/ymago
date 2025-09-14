"""
Unit tests for batch input parser functionality.

This module tests the streaming parser for CSV and JSONL files with
comprehensive coverage of error handling and validation scenarios.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from ymago.core.batch_parser import (
    BatchParseError,
    RejectedRow,
    _clean_row_data,
    _detect_format,
    _format_validation_error,
    parse_batch_input,
)
from ymago.models import GenerationRequest


class TestBatchParseError:
    """Test BatchParseError exception."""

    def test_batch_parse_error_creation(self):
        """Test creating BatchParseError with message."""
        error = BatchParseError("Test error message")
        assert str(error) == "Test error message"


class TestRejectedRow:
    """Test RejectedRow data structure."""

    def test_rejected_row_creation(self):
        """Test creating RejectedRow with all fields."""
        row = RejectedRow(
            row_number=5,
            raw_data={"prompt": "test", "invalid_field": "value"},
            error_message="Validation failed",
            error_type="validation_error",
        )

        assert row.row_number == 5
        assert row.raw_data == {"prompt": "test", "invalid_field": "value"}
        assert row.error_message == "Validation failed"
        assert row.error_type == "validation_error"

    def test_rejected_row_default_error_type(self):
        """Test RejectedRow with default error type."""
        row = RejectedRow(row_number=1, raw_data={}, error_message="Error")

        assert row.error_type == "validation_error"


class TestFormatDetection:
    """Test format detection functionality."""

    def test_detect_format_csv_extension(self):
        """Test format detection with .csv extension."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            file_path = Path(f.name)

        try:
            format_type = _detect_format(file_path)
            assert format_type == "csv"
        finally:
            file_path.unlink()

    def test_detect_format_jsonl_extension(self):
        """Test format detection with .jsonl extension."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            file_path = Path(f.name)

        try:
            format_type = _detect_format(file_path)
            assert format_type == "jsonl"
        finally:
            file_path.unlink()

    def test_detect_format_json_extension(self):
        """Test format detection with .json extension."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = Path(f.name)

        try:
            format_type = _detect_format(file_path)
            assert format_type == "jsonl"
        finally:
            file_path.unlink()

    def test_detect_format_content_based_json(self):
        """Test format detection based on JSON content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write('{"prompt": "test"}\n')
            file_path = Path(f.name)

        try:
            format_type = _detect_format(file_path)
            assert format_type == "jsonl"
        finally:
            file_path.unlink()

    def test_detect_format_content_based_csv(self):
        """Test format detection based on CSV content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write('prompt,output_name\n"test prompt","test_output"\n')
            file_path = Path(f.name)

        try:
            format_type = _detect_format(file_path)
            assert format_type == "csv"
        finally:
            file_path.unlink()

    def test_detect_format_unknown(self):
        """Test format detection failure."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".unknown", delete=False
        ) as f:
            f.write("unknown content format\n")
            file_path = Path(f.name)

        try:
            with pytest.raises(BatchParseError, match="Cannot determine format"):
                _detect_format(file_path)
        finally:
            file_path.unlink()


class TestRowDataCleaning:
    """Test row data cleaning and field mapping."""

    def test_clean_row_data_basic(self):
        """Test basic row data cleaning."""
        raw_data = {
            "prompt": "A beautiful sunset",
            "output_name": "sunset_image",
            "seed": "42",
        }

        cleaned = _clean_row_data(raw_data)

        assert cleaned["prompt"] == "A beautiful sunset"
        assert cleaned["output_filename"] == "sunset_image"
        assert cleaned["seed"] == 42

    def test_clean_row_data_aliases(self):
        """Test field alias mapping."""
        raw_data = {
            "text": "Test prompt",  # alias for prompt
            "filename": "test_file",  # alias for output_filename
            "random_seed": "123",  # alias for seed
            "negative": "no cars",  # alias for negative_prompt
        }

        cleaned = _clean_row_data(raw_data)

        assert cleaned["prompt"] == "Test prompt"
        assert cleaned["output_filename"] == "test_file"
        assert cleaned["seed"] == 123
        assert cleaned["negative_prompt"] == "no cars"

    def test_clean_row_data_empty_values(self):
        """Test handling of empty and None values."""
        raw_data = {
            "prompt": "Valid prompt",
            "output_name": "",  # empty string
            "seed": None,  # None value
            "quality": "   ",  # whitespace only
        }

        cleaned = _clean_row_data(raw_data)

        assert cleaned["prompt"] == "Valid prompt"
        assert "output_filename" not in cleaned
        assert "seed" not in cleaned
        assert "quality" not in cleaned

    def test_clean_row_data_invalid_seed(self):
        """Test handling of invalid seed values."""
        raw_data = {"prompt": "Test prompt", "seed": "not_a_number"}

        # Invalid seed should raise ValueError
        with pytest.raises(ValueError, match="Invalid seed value: not_a_number"):
            _clean_row_data(raw_data)


class TestValidationErrorFormatting:
    """Test Pydantic validation error formatting."""

    def test_format_validation_error_single(self):
        """Test formatting single validation error."""
        try:
            GenerationRequest(prompt="")  # Empty prompt should fail
        except ValidationError as e:
            formatted = _format_validation_error(e)
            assert "prompt" in formatted
            assert "at least 1 character" in formatted.lower()

    def test_format_validation_error_multiple(self):
        """Test formatting multiple validation errors."""
        try:
            GenerationRequest(
                prompt="",  # Empty prompt
                seed=-2,  # Invalid seed
            )
        except ValidationError as e:
            formatted = _format_validation_error(e)
            assert "prompt" in formatted
            assert "seed" in formatted
            assert ";" in formatted  # Multiple errors separated by semicolon


@pytest.mark.asyncio
class TestParseBatchInput:
    """Test the main parse_batch_input function."""

    async def test_parse_csv_file_valid(self):
        """Test parsing valid CSV file."""
        csv_content = """prompt,output_name,seed
"A beautiful sunset","sunset",42
"A mountain landscape","mountain",123
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            input_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                requests = []
                async for request in parse_batch_input(input_file, output_dir, "csv"):
                    requests.append(request)

                assert len(requests) == 2
                assert requests[0].prompt == "A beautiful sunset"
                assert requests[0].output_filename == "sunset"
                assert requests[0].seed == 42
                assert requests[1].prompt == "A mountain landscape"
                assert requests[1].output_filename == "mountain"
                assert requests[1].seed == 123

            finally:
                input_file.unlink()

    async def test_parse_jsonl_file_valid(self):
        """Test parsing valid JSONL file."""
        jsonl_content = """{"prompt": "A beautiful sunset", "output_filename": "sunset", "seed": 42}
{"prompt": "A mountain landscape", "output_filename": "mountain", "seed": 123}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            input_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                requests = []
                async for request in parse_batch_input(input_file, output_dir, "jsonl"):
                    requests.append(request)

                assert len(requests) == 2
                assert requests[0].prompt == "A beautiful sunset"
                assert requests[0].output_filename == "sunset"
                assert requests[0].seed == 42

            finally:
                input_file.unlink()

    async def test_parse_file_not_found(self):
        """Test handling of non-existent input file."""
        non_existent_file = Path("/non/existent/file.csv")
        output_dir = Path("/tmp")

        with pytest.raises(FileNotFoundError):
            async for _ in parse_batch_input(non_existent_file, output_dir):
                pass

    async def test_parse_unsupported_format(self):
        """Test handling of unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            input_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                with pytest.raises(BatchParseError, match="Unsupported format"):
                    async for _ in parse_batch_input(
                        input_file, output_dir, "unsupported"
                    ):
                        pass
            finally:
                input_file.unlink()

    async def test_parse_csv_with_invalid_rows(self):
        """Test CSV parsing with some invalid rows."""
        csv_content = """prompt,output_name,seed
"Valid prompt","valid_output",42
"","invalid_empty_prompt",123
"Another valid prompt","valid_output2",not_a_number
"Third valid prompt","valid_output3",456
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            input_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                requests = []
                async for request in parse_batch_input(input_file, output_dir, "csv"):
                    requests.append(request)

                # Should get 2 valid requests (rows 1 and 4)
                assert len(requests) == 2
                assert requests[0].prompt == "Valid prompt"
                assert requests[1].prompt == "Third valid prompt"

                # Check that rejected file was created
                rejected_file = output_dir / f"{input_file.stem}.rejected.csv"
                assert rejected_file.exists()

            finally:
                input_file.unlink()

    async def test_parse_jsonl_with_invalid_json(self):
        """Test JSONL parsing with invalid JSON lines."""
        jsonl_content = """{"prompt": "Valid prompt", "output_filename": "valid"}
invalid json line
{"prompt": "Another valid prompt", "output_filename": "valid2"}
{"prompt": "", "output_filename": "invalid_empty"}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            input_file = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            try:
                requests = []
                async for request in parse_batch_input(input_file, output_dir, "jsonl"):
                    requests.append(request)

                # Should get 2 valid requests (rows 1 and 3)
                assert len(requests) == 2
                assert requests[0].prompt == "Valid prompt"
                assert requests[1].prompt == "Another valid prompt"

                # Check that rejected file was created
                rejected_file = output_dir / f"{input_file.stem}.rejected.csv"
                assert rejected_file.exists()

            finally:
                input_file.unlink()
