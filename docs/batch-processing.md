# Batch Processing Guide

Ymago's batch processing system enables you to generate hundreds or thousands of images and videos from a single input file with resilient, high-throughput execution. This guide covers everything you need to know about using batch processing effectively.

## Overview

The batch processing system provides:

- **Resilient Execution**: Automatic retry logic and checkpoint-based resumption
- **High Throughput**: Configurable concurrency and intelligent rate limiting
- **Memory Efficiency**: Streaming input processing for files of any size
- **Comprehensive Reporting**: Detailed logs and statistics for every batch
- **Error Handling**: Graceful handling of invalid inputs and processing failures

## Quick Start

### Basic Usage

```bash
# Process a CSV file with default settings
ymago batch run prompts.csv --output-dir ./results/

# Process with custom concurrency and rate limiting
ymago batch run requests.jsonl --output-dir ./batch_output/ \
  --concurrency 20 --rate-limit 120

# Resume interrupted batch processing
ymago batch run prompts.csv --output-dir ./results/ --resume
```

### Input File Formats

#### CSV Format

Create a CSV file with generation parameters:

```csv
prompt,output_name,seed,quality,aspect_ratio
"A beautiful sunset over mountains","sunset_mountain",42,"high","16:9"
"A cat wearing a wizard hat","wizard_cat",123,"standard","1:1"
"Abstract geometric patterns","abstract_geo",456,"draft","4:3"
```

**Supported CSV columns:**
- `prompt` (required): Text prompt for generation
- `output_name` / `output_filename` / `filename`: Custom output filename
- `seed` / `random_seed`: Random seed for reproducible generation
- `quality`: Image quality ("draft", "standard", "high")
- `aspect_ratio` / `ratio`: Aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4")
- `negative_prompt` / `negative`: What to exclude from generation
- `from_image` / `source_image`: URL or path to source image
- `media_type` / `type`: "image" or "video"
- `image_model` / `model`: AI model for image generation
- `video_model`: AI model for video generation

#### JSONL Format

Create a JSON Lines file with one JSON object per line:

```jsonl
{"prompt": "A beautiful sunset", "output_filename": "sunset", "seed": 42}
{"prompt": "A mountain landscape", "quality": "high", "aspect_ratio": "16:9"}
{"prompt": "A forest scene", "negative_prompt": "buildings, cars"}
```

## Command Reference

### `ymago batch run`

Process a batch of generation requests from an input file.

```bash
ymago batch run INPUT_FILE --output-dir OUTPUT_DIR [OPTIONS]
```

#### Required Arguments

- `INPUT_FILE`: Path to CSV or JSONL input file

#### Required Options

- `--output-dir`, `-o`: Directory for storing results, logs, and state files

#### Optional Parameters

- `--concurrency`, `-c`: Maximum parallel requests (1-50, default: 10)
- `--rate-limit`, `-r`: Maximum requests per minute (1-300, default: 60)
- `--resume/--no-resume`: Resume from checkpoint (default: false)
- `--format`, `-f`: Input format ("csv", "jsonl", or auto-detect)
- `--dry-run`: Validate input and show plan without execution
- `--verbose`, `-v`: Enable verbose output

#### Examples

```bash
# Basic batch processing
ymago batch run prompts.csv --output-dir ./results/

# High-throughput processing
ymago batch run large_batch.csv --output-dir ./output/ \
  --concurrency 25 --rate-limit 180

# Resume interrupted batch
ymago batch run prompts.csv --output-dir ./results/ --resume

# Validate input without processing
ymago batch run prompts.csv --output-dir ./test/ --dry-run

# Process JSONL with verbose output
ymago batch run requests.jsonl --output-dir ./output/ \
  --format jsonl --verbose
```

## Output Files and Structure

When you run a batch, ymago creates several files in the output directory:

```
output_dir/
├── _batch_state.jsonl          # Checkpoint file for resumption
├── generated_image_001.png     # Generated media files
├── generated_image_002.png
├── ...
├── prompts.rejected.csv        # Invalid input rows (if any)
└── metadata files...           # Sidecar metadata (if enabled)
```

### Checkpoint File (`_batch_state.jsonl`)

The checkpoint file contains one JSON object per processed request:

```jsonl
{"request_id": "req1", "status": "success", "output_path": "/path/to/image.png", "processing_time_seconds": 2.5}
{"request_id": "req2", "status": "failure", "error_message": "Invalid prompt", "processing_time_seconds": 0.1}
```

### Rejected Rows File

If any input rows are invalid, they're saved to `{input_filename}.rejected.csv`:

```csv
row_number,error_type,error_message,raw_data
2,validation_error,"prompt: at least 1 character required","{""prompt"": """", ""output_name"": ""test""}"
5,json_error,"Invalid JSON: Expecting ',' delimiter","{""prompt"": ""test"" ""invalid""}"
```

## Resilient Processing

### Automatic Retry Logic

The batch processor automatically retries failed requests with exponential backoff:

- **Retry Count**: Up to 3 attempts per request
- **Backoff Strategy**: 1s, 2s, 4s, 8s delays
- **Retry Conditions**: Network errors, timeouts, rate limit errors
- **Permanent Failures**: Validation errors, authentication failures

### Checkpoint-Based Resumption

Batch processing can be safely interrupted and resumed:

```bash
# Start batch processing
ymago batch run large_batch.csv --output-dir ./results/

# If interrupted (Ctrl+C, system crash, etc.), resume with:
ymago batch run large_batch.csv --output-dir ./results/ --resume
```

**Resume behavior:**
- Skips already successful requests
- Retries previously failed requests
- Continues from the exact interruption point
- Preserves all progress and statistics

### Error Handling

The system gracefully handles various error conditions:

- **Invalid Input Rows**: Logged and saved to rejected file, processing continues
- **Network Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Intelligent rate limiting prevents API throttling
- **Disk Space**: Clear error messages for storage issues
- **Permission Errors**: Detailed error reporting for access issues

## Performance Optimization

### Concurrency Settings

Choose concurrency based on your system and API limits:

```bash
# Conservative (good for development)
--concurrency 5

# Balanced (good for most use cases)
--concurrency 10

# Aggressive (for high-performance systems)
--concurrency 25
```

**Guidelines:**
- Start with 10 concurrent requests
- Increase gradually while monitoring system resources
- Consider API rate limits and quotas
- Monitor memory usage with large batches

### Rate Limiting

Configure rate limiting to maximize throughput without hitting API limits:

```bash
# Conservative rate limiting
--rate-limit 60    # 1 request per second

# Balanced rate limiting
--rate-limit 120   # 2 requests per second

# Aggressive rate limiting
--rate-limit 300   # 5 requests per second
```

**Best Practices:**
- Check your API quota and limits
- Start conservatively and increase gradually
- Monitor for rate limit errors in logs
- Consider time-of-day variations in API performance

### Memory Management

For very large batches (10,000+ requests):

- Use streaming processing (automatic)
- Monitor system memory usage
- Consider processing in smaller chunks
- Ensure adequate disk space for outputs

## Troubleshooting

### Common Issues

#### "No valid requests found"

**Cause**: All input rows failed validation
**Solution**: Check the rejected rows file for specific errors

```bash
# Check rejected rows
cat output_dir/input_file.rejected.csv
```

#### "Cannot write to output directory"

**Cause**: Permission or disk space issues
**Solution**: Verify directory permissions and available space

```bash
# Check permissions
ls -la output_dir/
# Check disk space
df -h output_dir/
```

#### "Rate limit exceeded"

**Cause**: Too many requests per minute
**Solution**: Reduce rate limit or increase delays

```bash
# Reduce rate limiting
ymago batch run input.csv --output-dir ./output/ --rate-limit 30
```

#### Processing seems stuck

**Cause**: Network issues or very slow requests
**Solution**: Check logs and consider interrupting/resuming

```bash
# Interrupt with Ctrl+C, then resume
ymago batch run input.csv --output-dir ./output/ --resume --verbose
```

### Debug Mode

Enable verbose output for detailed debugging:

```bash
ymago batch run input.csv --output-dir ./output/ --verbose
```

This shows:
- Configuration details
- Progress information
- Error details
- Performance statistics

### Log Analysis

Check the checkpoint file for detailed request status:

```bash
# Count successful requests
grep '"status": "success"' output_dir/_batch_state.jsonl | wc -l

# Find failed requests
grep '"status": "failure"' output_dir/_batch_state.jsonl

# Calculate average processing time
grep '"processing_time_seconds"' output_dir/_batch_state.jsonl | \
  jq '.processing_time_seconds' | awk '{sum+=$1; count++} END {print sum/count}'
```

## Best Practices

### Input File Preparation

1. **Validate prompts**: Ensure all prompts are meaningful and within length limits
2. **Use consistent naming**: Follow a consistent pattern for output filenames
3. **Test with small batches**: Validate your input format with a small subset first
4. **Include variety**: Mix different types of prompts to test system robustness

### Batch Size Recommendations

- **Small batches (1-100 requests)**: Good for testing and development
- **Medium batches (100-1,000 requests)**: Optimal for most production use cases
- **Large batches (1,000+ requests)**: Ensure adequate system resources

### Monitoring and Maintenance

1. **Monitor progress**: Use `--verbose` for long-running batches
2. **Check disk space**: Ensure adequate space for all outputs
3. **Review rejected rows**: Address validation issues in input data
4. **Archive results**: Move completed batches to long-term storage

### Production Deployment

1. **Use configuration files**: Set default models and API keys in `ymago.toml`
2. **Implement monitoring**: Track batch completion and error rates
3. **Set up alerts**: Monitor for failed batches or system issues
4. **Plan for scale**: Consider distributed processing for very large workloads

## API Integration

You can also use batch processing programmatically:

```python
import asyncio
from pathlib import Path
from ymago.core.backends import LocalExecutionBackend
from ymago.core.batch_parser import parse_batch_input

async def process_batch_programmatically():
    backend = LocalExecutionBackend(max_concurrent_jobs=10)
    
    # Parse input file
    requests = parse_batch_input(
        input_file=Path("prompts.csv"),
        output_dir=Path("./output/"),
        format_hint="csv"
    )
    
    # Process batch
    summary = await backend.process_batch(
        requests=requests,
        output_dir=Path("./output/"),
        concurrency=10,
        rate_limit=60,
        resume=False
    )
    
    print(f"Processed {summary.total_requests} requests")
    print(f"Success rate: {summary.success_rate:.1f}%")

# Run the batch
asyncio.run(process_batch_programmatically())
```

## Next Steps

- Explore the [Configuration Guide](configuration.md) for advanced settings
- Learn about [Cloud Storage Integration](cloud-storage.md) for scalable deployments
- Check the [API Reference](api-reference.md) for programmatic usage
- See [Performance Tuning](performance.md) for optimization tips
