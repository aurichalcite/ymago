"""
Ymago - An advanced, asynchronous command-line toolkit and Python library for
generative AI media.

This package provides tools for generating images from text prompts using
various AI backends, with support for local and cloud storage, distributed
execution, and comprehensive configuration management.
"""

# Runtime guard to ensure Pydantic v2 is installed
import sys

import pydantic

from ._version import __version__

# Essential package-level exports for public API
from .config import Settings, load_config
from .models import GenerationJob, GenerationResult

# Use runtime check instead of assert to work in optimized mode
if not pydantic.VERSION.startswith("2."):
    error_msg = (
        f"Error: Pydantic v2 or greater is required,"
        f"but found version {pydantic.VERSION}. "
        "Please upgrade with: uv add 'pydantic>=2.0,<3.0'"
    )
    print(error_msg, file=sys.stderr)
    sys.exit(1)


__all__ = [
    "__version__",
    "Settings",
    "load_config",
    "GenerationJob",
    "GenerationResult",
]
