"""
Command-line interface for ymago package.

This module provides the main CLI application using Typer, with rich UI components
for progress indication and user feedback.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.status import Status
from rich.table import Table

from .config import load_config
from .core.backends import LocalExecutionBackend
from .core.batch_parser import parse_batch_input
from .core.generation import process_generation_job
from .models import BatchSummary, GenerationJob, GenerationResult

# Create the main Typer application
app = typer.Typer(
    name="ymago",
    help="An advanced, async command-line toolkit for generative AI media.",
    no_args_is_help=True,
)

# Create a sub-application for image commands
image_app = typer.Typer(
    name="image",
    help="Image generation commands",
    no_args_is_help=True,
)

# Create a sub-application for video commands
video_app = typer.Typer(
    name="video",
    help="Video generation commands",
    no_args_is_help=True,
)

# Create a sub-application for batch commands
batch_app = typer.Typer(
    name="batch",
    help="Batch processing commands",
    no_args_is_help=True,
)

# Add the sub-applications to the main app
app.add_typer(image_app, name="image")
app.add_typer(video_app, name="video")
app.add_typer(batch_app, name="batch")

# Create console for rich output
console = Console()


def _validate_aspect_ratio(aspect_ratio: str) -> bool:
    """Validate aspect ratio format."""
    import re

    pattern = r"^\d+:\d+$"
    return bool(re.match(pattern, aspect_ratio))


def _validate_url(url: str) -> bool:
    """Validate URL format."""
    import re

    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(url_pattern, url))


def _validate_url_or_path(value: str) -> bool:
    """Validate that value is a valid URL or an existing file path."""
    if _validate_url(value):
        return True
    return Path(value).expanduser().is_file()


def _validate_seed(seed: int) -> bool:
    """Validate seed value."""
    return seed == -1 or seed >= 0


@image_app.command("generate")  # type: ignore[misc]
def generate_image_command(
    prompt: Annotated[str, typer.Argument(help="Text prompt for image generation")],
    output_filename: Annotated[
        Optional[str],
        typer.Option(
            "--filename",
            "-f",
            help="Custom filename for the generated image (without extension)",
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducible generation (-1 for random)",
        ),
    ] = None,
    quality: Annotated[
        Optional[str],
        typer.Option(
            "--quality",
            "-q",
            help="Image quality setting (draft, standard, high)",
        ),
    ] = "standard",
    aspect_ratio: Annotated[
        Optional[str],
        typer.Option(
            "--aspect-ratio",
            "-a",
            help="Aspect ratio for the image (1:1, 16:9, 9:16, 4:3, 3:4)",
        ),
    ] = "1:1",
    negative_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--negative-prompt",
            "-n",
            help="Text describing what to avoid in the generated image",
        ),
    ] = None,
    from_image: Annotated[
        Optional[str],
        typer.Option(
            "--from-image",
            help="URL of source image for image-to-image generation",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="AI model to use for generation"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Generate an image from a text prompt.

    This command generates an image using Google's Generative AI and saves it
    to the configured output directory. It supports advanced features like
    negative prompts and image-to-image generation.

    Examples:
        ymago image generate "A beautiful sunset over mountains"
        ymago image generate "A cat wearing a hat" --filename "cat_hat" -s 42
        ymago image generate "Abstract art" -q high -a 16:9
        ymago image generate "A forest scene" -n "buildings, cars"
        ymago image generate "Transform this image" --from-image "https://.../image.jpg"
    """

    async def _async_generate() -> None:
        try:
            # Validate inputs
            if seed is not None and not _validate_seed(seed):
                console.print(
                    "[red]Error: Seed must be -1 (for random) "
                    "or a non-negative integer[/red]"
                )
                sys.exit(1)

            if aspect_ratio and not _validate_aspect_ratio(aspect_ratio):
                console.print(
                    "[red]Error: Aspect ratio must be in format 'width:height' "
                    "(e.g., '16:9')[/red]"
                )
                sys.exit(1)

            if from_image and not _validate_url(from_image):
                console.print(
                    "[red]Error: Source image must be a valid HTTP/HTTPS URL[/red]"
                )
                sys.exit(1)

            # Load configuration
            with Status("Loading configuration...", console=console) as status:
                config = await load_config()
                if verbose:
                    console.print(
                        f"✓ Configuration loaded from {config.defaults.output_path}"
                    )

            # Create generation job
            job = GenerationJob(
                prompt=prompt,
                output_filename=output_filename,
                seed=seed,
                quality=quality,
                aspect_ratio=aspect_ratio,
                negative_prompt=negative_prompt,
                from_image=from_image,
                image_model=model or config.defaults.image_model,
            )

            if verbose:
                _display_job_info(job)

            # Generate image with progress indication
            with Status("Generating image...", console=console) as status:
                result = await process_generation_job(job, config)
                status.update("Saving image...")

            # Display success message
            _display_success(result, verbose)

        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)

    # Run the async function
    asyncio.run(_async_generate())


@video_app.command("generate")  # type: ignore[misc]
def generate_video_command(
    prompt: Annotated[str, typer.Argument(help="Text prompt for video generation")],
    output_filename: Annotated[
        Optional[str],
        typer.Option(
            "--filename",
            "-f",
            help="Custom filename for the generated video (without extension)",
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducible generation (-1 for random)",
        ),
    ] = None,
    aspect_ratio: Annotated[
        Optional[str],
        typer.Option(
            "--aspect-ratio",
            "-a",
            help="Aspect ratio for the video (1:1, 16:9, 9:16, 4:3, 3:4)",
        ),
    ] = "16:9",
    negative_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--negative-prompt",
            "-n",
            help="Text describing what to avoid in the generated video",
        ),
    ] = None,
    from_image: Annotated[
        Optional[str],
        typer.Option(
            "--from-image",
            help="URL or local path of source image for image-to-video generation",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="AI model to use for video generation"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Generate a video from a text prompt.

    This command generates a video using Google's Veo model and saves it
    to the configured output directory. It supports advanced features like
    negative prompts and image-to-video generation.

    Examples:
        ymago video generate "A cat playing in a garden"
        ymago video generate "Ocean waves" --filename "waves" -s 42
        ymago video generate "Dancing" -a 9:16 -n "static"
        ymago video generate "Animate this image" --from-image "https://.../image.jpg"
    """

    async def _async_generate_video() -> None:
        try:
            # Validate inputs
            if seed is not None and not _validate_seed(seed):
                console.print(
                    "[red]Error: Seed must be -1 (for random) "
                    "or a non-negative integer[/red]"
                )
                sys.exit(1)

            if aspect_ratio and not _validate_aspect_ratio(aspect_ratio):
                console.print(
                    "[red]Error: Aspect ratio must be in format 'width:height' "
                    "(e.g., '16:9')[/red]"
                )
                sys.exit(1)

            if from_image and not _validate_url_or_path(from_image):
                console.print(
                    "[red]Error: Source image must be a valid "
                    "HTTP/HTTPS URL or an existing local file path[/red]"
                )
                sys.exit(1)

            # Load configuration
            with Status("Loading configuration...", console=console) as status:
                config = await load_config()
                if verbose:
                    console.print(
                        f"✓ Configuration loaded from {config.defaults.output_path}"
                    )

            # Create generation job for video
            job = GenerationJob(
                prompt=prompt,
                media_type="video",
                output_filename=output_filename,
                seed=seed,
                aspect_ratio=aspect_ratio,
                negative_prompt=negative_prompt,
                from_image=from_image,
                video_model=model or config.defaults.video_model,
            )

            if verbose:
                _display_job_info(job)

            # Generate video with progress indication
            with Status(
                "Generating video (this may take several minutes)...", console=console
            ) as status:
                result = await process_generation_job(job, config)
                status.update("Saving video...")

            # Display success message
            _display_video_success(result, verbose)

        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)

    # Run the async function
    asyncio.run(_async_generate_video())


def _display_job_info(job: GenerationJob) -> None:
    """Display information about the generation job."""
    table = Table(title="Generation Job Details", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    prompt_display = job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt
    table.add_row("Prompt", prompt_display)
    table.add_row("Media Type", job.media_type.title())
    table.add_row("Model", job.model_name)
    table.add_row("Quality", job.quality or "standard")
    table.add_row("Aspect Ratio", job.aspect_ratio or "1:1")

    if job.seed is not None:
        table.add_row("Seed", str(job.seed))
    if job.negative_prompt:
        neg_display = (
            job.negative_prompt[:50] + "..."
            if len(job.negative_prompt) > 50
            else job.negative_prompt
        )
        table.add_row("Negative Prompt", neg_display)
    if job.from_image:
        table.add_row("Source Image", job.from_image)
    if job.output_filename:
        table.add_row("Custom Filename", job.output_filename)

    console.print(table)
    console.print()


def _display_success(result: "GenerationResult", verbose: bool = False) -> None:
    """Display success message with result information."""
    # Main success message
    console.print("[green]✓ Image generated successfully![/green]")
    console.print(f"[blue]Saved to:[/blue] {result.local_path}")

    if verbose:
        # Detailed information table
        table = Table(title="Generation Results", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("File Size", f"{result.file_size_bytes:,} bytes")
        table.add_row(
            "Generation Time", f"{result.generation_time_seconds:.2f} seconds"
        )
        table.add_row("Model Used", result.get_metadata("api_model", "unknown"))

        if result.get_metadata("seed"):
            table.add_row("Seed", str(result.get_metadata("seed")))

        console.print()
        console.print(table)


def _display_video_success(result: "GenerationResult", verbose: bool = False) -> None:
    """Display success message for video generation with result information."""
    # Main success message
    console.print("[green]✓ Video generated successfully![/green]")
    console.print(f"[blue]Saved to:[/blue] {result.local_path}")

    if verbose:
        # Detailed information table
        table = Table(title="Video Generation Results", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("File Size", f"{result.file_size_bytes:,} bytes")
        table.add_row(
            "Generation Time", f"{result.generation_time_seconds:.2f} seconds"
        )
        table.add_row("Model Used", result.get_metadata("api_model", "unknown"))
        table.add_row("Media Type", result.get_metadata("media_type", "video").title())

        if result.get_metadata("seed"):
            table.add_row("Seed", str(result.get_metadata("seed")))
        if result.get_metadata("negative_prompt"):
            table.add_row(
                "Negative Prompt", str(result.get_metadata("negative_prompt"))
            )
        if result.get_metadata("source_image_url"):
            table.add_row("Source Image", str(result.get_metadata("source_image_url")))

        console.print()
        console.print(table)


@app.command("version")  # type: ignore[misc]
def version_command() -> None:
    """Display version information."""
    from . import __version__

    console.print(f"ymago version {__version__}")


@app.command("config")  # type: ignore[misc]
def config_command(
    show_path: Annotated[
        bool,
        typer.Option("--show-path", help="Show the configuration file path"),
    ] = False,
) -> None:
    """Display current configuration."""

    async def _async_config() -> None:
        try:
            config = await load_config()

            if show_path:
                # Try to find which config file was used
                config_paths = [Path.cwd() / "ymago.toml", Path.home() / ".ymago.toml"]

                config_file = None
                for path in config_paths:
                    if path.exists():
                        config_file = path
                        break

                if config_file:
                    console.print(f"Configuration file: {config_file}")
                else:
                    console.print("Configuration from environment variables")
                console.print()

            # Display configuration (without sensitive data)
            table = Table(title="Current Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Image Model", config.defaults.image_model)
            table.add_row("Video Model", config.defaults.video_model)
            table.add_row("Output Path", str(config.defaults.output_path))
            table.add_row(
                "Metadata Enabled", "Yes" if config.defaults.enable_metadata else "No"
            )
            api_key_display = (
                "***" + config.auth.google_api_key[-4:]
                if len(config.auth.google_api_key) > 4
                else "***"
            )
            table.add_row("API Key", api_key_display)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            sys.exit(1)

    # Run the async function
    asyncio.run(_async_config())


@batch_app.command("run")  # type: ignore[misc]
def run_batch_command(
    input_file: Annotated[
        Path,
        typer.Argument(..., help="Path to CSV or JSONL file with generation requests"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            ...,
            "--output-dir",
            "-o",
            help="Directory for storing results, logs, and state",
        ),
    ],
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-c",
            help="Maximum parallel requests (1-50)",
            min=1,
            max=50,
        ),
    ] = 10,
    rate_limit: Annotated[
        int,
        typer.Option(
            "--rate-limit", "-r", help="Maximum requests per minute", min=1, max=300
        ),
    ] = 60,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume", help="Resume from checkpoint in output dir"
        ),
    ] = False,
    format_hint: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            "-f",
            help="Input format (csv, jsonl, or auto-detect if not specified)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Validate input and show plan without execution"
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """
    Process a batch of generation requests from a CSV or JSONL file.

    This command processes multiple generation requests concurrently with
    resilient execution, rate limiting, and checkpoint-based resumption.

    Examples:
        ymago batch run prompts.csv -o ./results/
        ymago batch run reqs.jsonl -o ./out/ -c 20
        ymago batch run data.csv -o ./res/ --resume -r 120
        ymago batch run prompts.csv -o ./test/ --dry-run
    """
    asyncio.run(
        _async_run_batch(
            input_file=input_file,
            output_dir=output_dir,
            concurrency=concurrency,
            rate_limit=rate_limit,
            resume=resume,
            format_hint=format_hint,
            dry_run=dry_run,
            verbose=verbose,
        )
    )


async def _async_run_batch(
    input_file: Path,
    output_dir: Path,
    concurrency: int,
    rate_limit: int,
    resume: bool,
    format_hint: Optional[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Async implementation of batch processing command."""
    try:
        # Validate input file exists
        if not input_file.exists():
            console.print(f"[red]Error: Input file not found: {input_file}[/red]")
            sys.exit(1)

        # Validate output directory is writable
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            console.print(
                f"[red]Error: Cannot write to output directory {output_dir}: {e}[/red]"
            )
            sys.exit(1)

        # Load configuration
        with Status("Loading configuration...", console=console):
            await load_config()

        if verbose:
            console.print("✓ Configuration loaded")
            console.print(f"✓ Input file: {input_file}")
            console.print(f"✓ Output directory: {output_dir}")
            console.print(f"✓ Concurrency: {concurrency}")
            console.print(f"✓ Rate limit: {rate_limit} requests/minute")

        # Parse and validate input
        console.print("\n[bold blue]Parsing input file...[/bold blue]")

        request_count = 0

        # Count requests and validate (for dry run and progress tracking)
        async for request in parse_batch_input(input_file, output_dir, format_hint):
            request_count += 1
            if dry_run and request_count <= 5:  # Show first 5 requests in dry run
                console.print(f"  Request {request_count}: {request.prompt[:50]}...")

        if request_count == 0:
            console.print("[yellow]No valid requests found in input file[/yellow]")
            sys.exit(0)

        console.print(f"✓ Found {request_count} valid requests")

        # Check for rejected rows file
        rejected_file = output_dir / f"{input_file.stem}.rejected.csv"
        if rejected_file.exists():
            console.print(
                f"[yellow]⚠ Rejected rows file exists: {rejected_file}[/yellow]"
            )

        if dry_run:
            console.print("\n[green]Dry run completed successfully![/green]")
            console.print(f"Would process {request_count} requests with:")
            console.print(f"  • Concurrency: {concurrency}")
            console.print(f"  • Rate limit: {rate_limit} requests/minute")
            estimated_time = _estimate_processing_time(request_count, rate_limit)
            console.print(f"  • Estimated time: {estimated_time}")
            return

        # Initialize backend and start processing
        backend = LocalExecutionBackend(max_concurrent_jobs=concurrency)

        console.print("\n[bold green]Starting batch processing...[/bold green]")

        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            # Add progress task
            task = progress.add_task(
                f"Processing {request_count} requests...", total=request_count
            )

            # Process batch

            # Re-parse requests for processing
            requests_generator = parse_batch_input(input_file, output_dir, format_hint)

            summary = await backend.process_batch(
                requests=requests_generator,
                output_dir=output_dir,
                concurrency=concurrency,
                rate_limit=rate_limit,
                resume=resume,
            )

            progress.update(task, completed=request_count)

        # Display results
        _display_batch_summary(summary, verbose)

    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing cancelled by user[/yellow]")
        console.print("Progress has been saved. Use --resume to continue.")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _estimate_processing_time(request_count: int, rate_limit: int) -> str:
    """Estimate processing time based on rate limit."""
    minutes = request_count / rate_limit
    if minutes < 1:
        return f"{int(minutes * 60)} seconds"
    elif minutes < 60:
        return f"{minutes:.1f} minutes"
    else:
        hours = minutes / 60
        return f"{hours:.1f} hours"


def _display_batch_summary(summary: BatchSummary, verbose: bool) -> None:
    """Display batch processing summary with rich formatting."""
    console.print("\n[bold green]Batch Processing Complete![/bold green]")

    # Create summary table
    table = Table(
        title="Batch Processing Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Total Requests", str(summary.total_requests))
    table.add_row("Successful", f"[green]{summary.successful}[/green]")
    table.add_row("Failed", f"[red]{summary.failed}[/red]")
    table.add_row("Skipped", f"[yellow]{summary.skipped}[/yellow]")
    table.add_row("Success Rate", f"{summary.success_rate:.1f}%")
    table.add_row("Processing Time", f"{summary.processing_time_seconds:.1f} seconds")
    table.add_row("Throughput", f"{summary.throughput_requests_per_minute:.1f} req/min")

    console.print(table)

    # Show file locations
    console.print("\n[bold]Output Files:[/bold]")
    console.print(f"  • Results log: {summary.results_log_path}")
    if summary.rejected_rows_path:
        console.print(f"  • Rejected rows: {summary.rejected_rows_path}")

    if verbose and summary.failed > 0:
        console.print(
            "\n[yellow]Check the results log for detailed error information[/yellow]"
        )


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
