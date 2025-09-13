"""
Command-line interface for ymago package.

This module provides the main CLI application using Typer, with rich UI components
for progress indication and user feedback.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from .config import load_config
from .core.generation import process_generation_job
from .models import GenerationJob, GenerationResult

# Create the main Typer application
app = typer.Typer(
    name="ymago",
    help="An advanced, asynchronous command-line toolkit for generative AI media.",
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

# Add the sub-applications to the main app
app.add_typer(image_app, name="image")
app.add_typer(video_app, name="video")

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


@image_app.command("generate")
def generate_image_command(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    output_filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-f",
        help="Custom filename for the generated image (without extension)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducible generation (-1 for random)",
    ),
    quality: Optional[str] = typer.Option(
        "standard",
        "--quality",
        "-q",
        help="Image quality setting (draft, standard, high)",
    ),
    aspect_ratio: Optional[str] = typer.Option(
        "1:1",
        "--aspect-ratio",
        "-a",
        help="Aspect ratio for the image (1:1, 16:9, 9:16, 4:3, 3:4)",
    ),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        "-n",
        help="Text describing what to avoid in the generated image",
    ),
    from_image: Optional[str] = typer.Option(
        None,
        "--from-image",
        help="URL of source image for image-to-image generation",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="AI model to use for generation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """
    Generate an image from a text prompt.

    This command generates an image using Google's Generative AI and saves it
    to the configured output directory. Supports advanced features like negative
    prompts and image-to-image generation.

    Examples:
        ymago image generate "A beautiful sunset over mountains"
        ymago image generate "A cat wearing a hat" --filename "cat_hat" --seed 42
        ymago image generate "Abstract art" --quality high --aspect-ratio 16:9
        ymago image generate "A forest scene" --negative-prompt "buildings, cars"
        ymago image generate "Transform this image" --from-image "https://example.com/image.jpg"
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


@video_app.command("generate")
def generate_video_command(
    prompt: str = typer.Argument(..., help="Text prompt for video generation"),
    output_filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-f",
        help="Custom filename for the generated video (without extension)",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducible generation (-1 for random)",
    ),
    aspect_ratio: Optional[str] = typer.Option(
        "16:9",
        "--aspect-ratio",
        "-a",
        help="Aspect ratio for the video (1:1, 16:9, 9:16, 4:3, 3:4)",
    ),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        "-n",
        help="Text describing what to avoid in the generated video",
    ),
    from_image: Optional[str] = typer.Option(
        None,
        "--from-image",
        help="URL or local path of source image for image-to-video generation",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="AI model to use for video generation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """
    Generate a video from a text prompt.

    This command generates a video using Google's Veo model and saves it
    to the configured output directory. Supports advanced features like negative
    prompts and image-to-video generation.

    Examples:
        ymago video generate "A cat playing in a garden"
        ymago video generate "Ocean waves" --filename "waves" --seed 42
        ymago video generate "Dancing" --aspect-ratio 9:16 --negative-prompt "static"
        ymago video generate "Animate this image" --from-image "https://example.com/image.jpg"
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


@app.command("version")
def version_command() -> None:
    """Display version information."""
    from . import __version__

    console.print(f"ymago version {__version__}")


@app.command("config")
def config_command(
    show_path: bool = typer.Option(
        False, "--show-path", help="Show the configuration file path"
    ),
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


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
