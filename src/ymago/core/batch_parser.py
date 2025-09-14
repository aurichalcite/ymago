"""
Asynchronous batch input parser for ymago package.

This module provides streaming parsers for CSV and JSONL input files with
robust error handling, memory efficiency, and detailed rejection tracking.
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiocsv
import aiofiles
from pydantic import ValidationError

from ..models import GenerationRequest

logger = logging.getLogger(__name__)


class BatchParseError(Exception):
    """Exception raised for batch parsing errors."""

    pass


class RejectedRow:
    """Represents a rejected input row with error details."""

    def __init__(
        self,
        row_number: int,
        raw_data: Dict[str, str],
        error_message: str,
        error_type: str = "validation_error",
    ):
        self.row_number = row_number
        self.raw_data = raw_data
        self.error_message = error_message
        self.error_type = error_type


async def parse_batch_input(
    input_file: Path, output_dir: Path, format_hint: Optional[str] = None
) -> AsyncGenerator[GenerationRequest, None]:
    """
    Parse batch input file and yield GenerationRequest objects.

    This function streams through the input file, validates each row,
    and yields valid GenerationRequest objects while tracking rejected
    rows for later reporting.

    Args:
        input_file: Path to the input CSV or JSONL file
        output_dir: Directory where rejection files will be written
        format_hint: Optional format hint ("csv", "jsonl", or None for auto-detect)

    Yields:
        GenerationRequest: Valid generation requests from the input file

    Raises:
        BatchParseError: For file access or format detection errors
        FileNotFoundError: If input file doesn't exist
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Auto-detect format if not provided
    if format_hint is None:
        format_hint = _detect_format(input_file)

    logger.info(f"Parsing {format_hint.upper()} file: {input_file}")

    rejected_rows: List[RejectedRow] = []

    try:
        if format_hint.lower() == "csv":
            async for request in _parse_csv_file(input_file, rejected_rows):
                yield request
        elif format_hint.lower() in ("jsonl", "json"):
            async for request in _parse_jsonl_file(input_file, rejected_rows):
                yield request
        else:
            raise BatchParseError(f"Unsupported format: {format_hint}")

    finally:
        # Write rejected rows to file if any exist
        if rejected_rows:
            await _write_rejected_rows(rejected_rows, input_file, output_dir)


def _detect_format(input_file: Path) -> str:
    """
    Auto-detect input file format based on extension and content.

    Args:
        input_file: Path to the input file

    Returns:
        str: Detected format ("csv" or "jsonl")

    Raises:
        BatchParseError: If format cannot be determined
    """
    extension = input_file.suffix.lower()

    if extension == ".csv":
        return "csv"
    elif extension in (".jsonl", ".json"):
        return "jsonl"
    else:
        # Try to detect by examining first few lines
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("{") and first_line.endswith("}"):
                    return "jsonl"
                elif "," in first_line:
                    return "csv"
        except Exception as e:
            logger.warning(f"Failed to auto-detect format: {e}")

    raise BatchParseError(
        f"Cannot determine format for file: {input_file}. "
        f"Please specify format explicitly or use .csv/.jsonl extension."
    )


async def _parse_csv_file(
    input_file: Path, rejected_rows: List[RejectedRow]
) -> AsyncGenerator[GenerationRequest, None]:
    """Parse CSV file and yield valid GenerationRequest objects."""
    row_number = 0

    async with aiofiles.open(input_file, mode="r", encoding="utf-8", newline="") as f:
        async for row in aiocsv.AsyncDictReader(f):
            row_number += 1

            try:
                # Clean and validate row data
                cleaned_row = _clean_row_data(row)

                # Create GenerationRequest with row number for tracking
                request = GenerationRequest(**cleaned_row, row_number=row_number)
                yield request

            except ValidationError as e:
                error_msg = _format_validation_error(e)
                rejected_row = RejectedRow(
                    row_number=row_number,
                    raw_data=dict(row),
                    error_message=error_msg,
                    error_type="validation_error",
                )
                rejected_rows.append(rejected_row)
                logger.warning(f"Row {row_number} rejected: {error_msg}")

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                rejected_row = RejectedRow(
                    row_number=row_number,
                    raw_data=dict(row),
                    error_message=error_msg,
                    error_type="parsing_error",
                )
                rejected_rows.append(rejected_row)
                logger.error(f"Row {row_number} failed: {error_msg}")


async def _parse_jsonl_file(
    input_file: Path, rejected_rows: List[RejectedRow]
) -> AsyncGenerator[GenerationRequest, None]:
    """Parse JSONL file and yield valid GenerationRequest objects."""
    row_number = 0

    async with aiofiles.open(input_file, mode="r", encoding="utf-8") as f:
        async for line in f:
            row_number += 1
            line = line.strip()

            if not line:  # Skip empty lines
                continue

            row_data: Dict[str, Any] = {
                "raw_line": line
            }  # Initialize for error handling
            try:
                # Parse JSON line
                row_data = json.loads(line)

                # Clean and validate row data
                cleaned_row = _clean_row_data(row_data)

                # Create GenerationRequest with row number for tracking
                request = GenerationRequest(**cleaned_row, row_number=row_number)
                yield request

            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {str(e)}"
                rejected_row = RejectedRow(
                    row_number=row_number,
                    raw_data={"raw_line": line},
                    error_message=error_msg,
                    error_type="json_error",
                )
                rejected_rows.append(rejected_row)
                logger.warning(f"Row {row_number} rejected: {error_msg}")

            except ValidationError as e:
                error_msg = _format_validation_error(e)
                rejected_row = RejectedRow(
                    row_number=row_number,
                    raw_data=row_data,
                    error_message=error_msg,
                    error_type="validation_error",
                )
                rejected_rows.append(rejected_row)
                logger.warning(f"Row {row_number} rejected: {error_msg}")

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                rejected_row = RejectedRow(
                    row_number=row_number,
                    raw_data=row_data,
                    error_message=error_msg,
                    error_type="parsing_error",
                )
                rejected_rows.append(rejected_row)
                logger.error(f"Row {row_number} failed: {error_msg}")


def _clean_row_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize row data for GenerationRequest creation.

    Args:
        row: Raw row data from CSV or JSON

    Returns:
        Dict[str, Any]: Cleaned row data with properly typed values
    """
    cleaned: Dict[str, Any] = {}

    # Define field mappings and aliases
    field_mappings = {
        "prompt": ["prompt", "text", "description"],
        "output_filename": ["output_filename", "filename", "name", "output_name"],
        "media_type": ["media_type", "type"],
        "seed": ["seed", "random_seed"],
        "negative_prompt": ["negative_prompt", "negative", "exclude"],
        "from_image": ["from_image", "source_image", "input_image", "image_url"],
        "quality": ["quality"],
        "aspect_ratio": ["aspect_ratio", "ratio"],
        "image_model": ["image_model", "model"],
        "video_model": ["video_model", "model"],
    }

    # Map fields using aliases
    for target_field, aliases in field_mappings.items():
        for alias in aliases:
            if alias in row and row[alias] is not None and str(row[alias]).strip():
                value = str(row[alias]).strip()

                # Type conversion for specific fields
                if target_field == "seed" and value:
                    try:
                        cleaned[target_field] = int(value)
                    except ValueError as e:
                        # Invalid seed should cause validation error
                        raise ValueError(f"Invalid seed value: {value}") from e
                elif target_field == "media_type":
                    # Ensure media_type is properly typed as Literal
                    if value not in ["image", "video"]:
                        raise ValueError(
                            f"Invalid media_type: {value}. Must be 'image' or 'video'"
                        )
                    cleaned[target_field] = value
                else:
                    cleaned[target_field] = value
                break

    return cleaned


def _format_validation_error(error: ValidationError) -> str:
    """Format Pydantic validation error for user-friendly display."""
    errors = []
    for err in error.errors():
        field = ".".join(str(loc) for loc in err["loc"])
        message = err["msg"]
        errors.append(f"{field}: {message}")
    return "; ".join(errors)


async def _write_rejected_rows(
    rejected_rows: List[RejectedRow], input_file: Path, output_dir: Path
) -> None:
    """Write rejected rows to a CSV file for user review."""
    if not rejected_rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_file = output_dir / f"{input_file.stem}.rejected.csv"

    logger.info(f"Writing {len(rejected_rows)} rejected rows to: {rejected_file}")

    async with aiofiles.open(
        rejected_file, mode="w", encoding="utf-8", newline=""
    ) as f:
        # Write header
        await f.write("row_number,error_type,error_message,raw_data\n")

        # Write rejected rows
        for row in rejected_rows:
            # Escape raw data for CSV
            raw_data_str = json.dumps(row.raw_data).replace('"', '""')
            error_msg_escaped = row.error_message.replace('"', '""')

            line = (
                f'{row.row_number},"{row.error_type}",'
                f'"{error_msg_escaped}","{raw_data_str}"\n'
            )
            await f.write(line)
