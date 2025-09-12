"""
Ymago - An advanced, asynchronous command-line toolkit and Python library for
generative AI media.

This package provides tools for generating images from text prompts using
various AI backends, with support for local and cloud storage, distributed
execution, and comprehensive configuration management.
"""

# Runtime guard to ensure Pydantic v2 is installed
import pydantic

# Essential package-level exports for public API
from .config import Settings, load_config
from .models import GenerationJob, GenerationResult

assert pydantic.VERSION.startswith("2."), (
    f"Pydantic v2 or greater is required, but found version {pydantic.VERSION}. "
    "Please upgrade with: uv add 'pydantic>=2.0,<3.0'"
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "load_config",
    "GenerationJob",
    "GenerationResult",
]
