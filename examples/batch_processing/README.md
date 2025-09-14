# Batch Processing Examples

This directory contains example input files and usage patterns for ymago's batch processing functionality.

## Example Files

### `sample_prompts.csv`
A comprehensive CSV example with 20 diverse image generation prompts including:
- Various artistic styles and subjects
- Different quality settings and aspect ratios
- Negative prompts for better control
- Consistent naming conventions

### `sample_requests.jsonl`
A JSONL example with 15 creative prompts showcasing:
- Fantasy and sci-fi themes
- Mixed quality and aspect ratio settings
- Optional negative prompts
- Structured JSON format

## Usage Examples

### Basic Batch Processing

```bash
# Process the CSV example with default settings
ymago batch run sample_prompts.csv --output-dir ./csv_results/

# Process the JSONL example with higher concurrency
ymago batch run sample_requests.jsonl --output-dir ./jsonl_results/ --concurrency 15
```

### Advanced Configuration

```bash
# High-throughput processing with custom settings
ymago batch run sample_prompts.csv \
  --output-dir ./high_throughput_results/ \
  --concurrency 20 \
  --rate-limit 180 \
  --verbose

# Conservative processing for testing
ymago batch run sample_requests.jsonl \
  --output-dir ./test_results/ \
  --concurrency 5 \
  --rate-limit 30 \
  --dry-run
```

### Resume Interrupted Processing

```bash
# Start processing
ymago batch run sample_prompts.csv --output-dir ./results/

# If interrupted, resume from checkpoint
ymago batch run sample_prompts.csv --output-dir ./results/ --resume
```

## Creating Your Own Input Files

### CSV Format Guidelines

1. **Required column**: `prompt`
2. **Recommended columns**: `output_name`, `seed`, `quality`, `aspect_ratio`
3. **Optional columns**: `negative_prompt`, `media_type`, `from_image`

Example CSV structure:
```csv
prompt,output_name,seed,quality,aspect_ratio,negative_prompt
"Your creative prompt here","output_filename",42,"high","16:9","things to avoid"
```

### JSONL Format Guidelines

Each line should be a valid JSON object with at least a `prompt` field:

```jsonl
{"prompt": "Your prompt", "output_filename": "name", "seed": 42}
{"prompt": "Another prompt", "quality": "high", "aspect_ratio": "16:9"}
```

## Field Reference

| Field | CSV Aliases | Description | Values |
|-------|-------------|-------------|---------|
| `prompt` | `text`, `description` | Generation prompt (required) | Any text |
| `output_filename` | `output_name`, `filename`, `name` | Output file name | String without extension |
| `seed` | `random_seed` | Random seed | Integer (-1 for random) |
| `quality` | | Image quality | "draft", "standard", "high" |
| `aspect_ratio` | `ratio` | Image dimensions | "1:1", "16:9", "9:16", "4:3", "3:4" |
| `negative_prompt` | `negative`, `exclude` | What to avoid | Any text |
| `media_type` | `type` | Output type | "image", "video" |
| `from_image` | `source_image`, `input_image` | Source image URL/path | URL or file path |
| `image_model` | `model` | AI model for images | Model name |
| `video_model` | | AI model for videos | Model name |

## Tips for Large Batches

### Performance Optimization

1. **Start small**: Test with 10-20 requests first
2. **Increase gradually**: Scale up concurrency and rate limits
3. **Monitor resources**: Watch CPU, memory, and network usage
4. **Use checkpoints**: Enable resume for long-running batches

### Error Prevention

1. **Validate prompts**: Ensure all prompts are meaningful
2. **Check file paths**: Verify source image URLs/paths exist
3. **Test format**: Use `--dry-run` to validate input format
4. **Monitor quotas**: Check API usage limits

### Batch Size Recommendations

- **Development/Testing**: 10-50 requests
- **Small Production**: 100-500 requests  
- **Medium Production**: 500-2,000 requests
- **Large Production**: 2,000+ requests (consider chunking)

## Troubleshooting Common Issues

### "No valid requests found"
- Check CSV headers match expected field names
- Verify JSONL has valid JSON on each line
- Ensure at least one row has a valid prompt

### "Rate limit exceeded"
- Reduce `--rate-limit` value
- Check your API quota and usage
- Consider processing during off-peak hours

### "Permission denied" errors
- Verify output directory is writable
- Check file system permissions
- Ensure adequate disk space

### Processing appears stuck
- Enable `--verbose` for detailed progress
- Check network connectivity
- Monitor system resources (CPU, memory)

## Integration Examples

### Python API Usage

```python
import asyncio
from pathlib import Path
from ymago.core.backends import LocalExecutionBackend
from ymago.core.batch_parser import parse_batch_input

async def custom_batch_processing():
    backend = LocalExecutionBackend(max_concurrent_jobs=10)
    
    requests = parse_batch_input(
        input_file=Path("sample_prompts.csv"),
        output_dir=Path("./api_results/"),
        format_hint="csv"
    )
    
    summary = await backend.process_batch(
        requests=requests,
        output_dir=Path("./api_results/"),
        concurrency=10,
        rate_limit=60,
        resume=False
    )
    
    print(f"Completed: {summary.successful}/{summary.total_requests}")
    return summary

# Run the batch
summary = asyncio.run(custom_batch_processing())
```

### Shell Script Automation

```bash
#!/bin/bash
# batch_process.sh - Automated batch processing script

INPUT_FILE="$1"
OUTPUT_DIR="$2"
CONCURRENCY="${3:-10}"
RATE_LIMIT="${4:-60}"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_file> <output_dir> [concurrency] [rate_limit]"
    exit 1
fi

echo "Starting batch processing..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Concurrency: $CONCURRENCY"
echo "Rate Limit: $RATE_LIMIT"

ymago batch run "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --concurrency "$CONCURRENCY" \
    --rate-limit "$RATE_LIMIT" \
    --verbose

echo "Batch processing completed!"
```

## Next Steps

1. Try the example files with your ymago installation
2. Create your own input files based on these templates
3. Experiment with different concurrency and rate limit settings
4. Explore the full [Batch Processing Guide](../../docs/batch-processing.md)
5. Check out [Performance Tuning](../../docs/performance.md) for optimization tips
