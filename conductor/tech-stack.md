# Technology Stack: ymago

## Core Language
- **Python (>=3.10):** The project targets modern Python environments, utilizing the latest asynchronous features and type-hinting capabilities.

## Frameworks & Libraries
- **CLI Framework:** **Typer** for building modular, type-safe command-line interfaces.
- **UI & Formatting:** **Rich** for creating beautiful terminal output, including progress bars, spinners, and formatted panels.
- **Asynchronous HTTP:** **aiohttp** for non-blocking, concurrent API interactions.
- **Generative AI:** **google-genai** for direct integration with Google's Gemini models (Nano Banana, Veo).
- **Resilience:** **Tenacity** for automated retries with exponential backoff for handling transient network and API failures.

## Infrastructure & Cloud
- **Cloud Storage Integration:**
  - **AWS S3** and **Cloudflare R2** via **aioboto3** and **botocore**.
  - **Google Cloud Storage** via **gcloud-aio-storage**.
- **Asynchronous Core:** Non-blocking I/O and task management throughout the system.

## Tooling & Package Management
- **Package Management:** **uv** for high-speed, reproducible environment and dependency management.
- **Build System:** **Hatchling** and **Hatch-VCS** for versioning and building distributable packages.
- **Documentation:** **Sphinx** with **myst-parser** for Markdown support.

## Quality Assurance & Testing
- **Testing:** **pytest**, **pytest-cov**, and **pytest-asyncio** for unit and integration tests.
- **Property-Based Testing:** **Hypothesis** for robust edge-case discovery.
- **Linting & Formatting:** **Ruff** for extremely fast linting and formatting.
- **Static Typing:** **mypy** and **basedpyright** for strict, project-wide type checking.
