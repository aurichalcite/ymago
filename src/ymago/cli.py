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

# Add the image sub-application to the main app
app.add_typer(image_app, name="image")

# Create console for rich output
console = Console()


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
        None, "--seed", "-s", help="Random seed for reproducible generation"
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
    to the configured output directory.

    Examples:
        ymago image generate "A beautiful sunset over mountains"
        ymago image generate "A cat wearing a hat" --filename "cat_hat" --seed 42
        ymago image generate "Abstract art" --quality high --aspect-ratio 16:9
    """

    async def _async_generate() -> None:
        try:
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


def _display_job_info(job: GenerationJob) -> None:
    """Display information about the generation job."""
    table = Table(title="Generation Job Details", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    prompt_display = job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt
    table.add_row("Prompt", prompt_display)
    table.add_row("Model", job.image_model)
    table.add_row("Quality", job.quality or "standard")
    table.add_row("Aspect Ratio", job.aspect_ratio or "1:1")

    if job.seed is not None:
        table.add_row("Seed", str(job.seed))
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
            table.add_row("Output Path", str(config.defaults.output_path))
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
