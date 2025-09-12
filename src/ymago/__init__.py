"""
Ymago - An advanced, asynchronous command-line toolkit and Python library for
generative AI media.

This package provides tools for generating images from text prompts using
various AI backends, with support for local and cloud storage, distributed
execution, and comprehensive configuration management.
"""

__version__ = "0.1.0"

# Essential package-level exports for public API
from .config import Settings, load_config
from .models import GenerationJob, GenerationResult

__all__ = [
    "__version__",
    "Settings",
    "load_config",
    "GenerationJob",
    "GenerationResult",
]
