# Initial Concept
ymago is a high-performance, asynchronous command-line toolkit and Python library for generative AI media. It leverages Google Gemini models to provide a professional-grade workflow for large-scale, reproducible, and integrable media generation, supporting both image and video synthesis with built-in resilience and cloud storage integration.

---

# Product Guide: ymago

## Introduction
**ymago** is a high-performance, asynchronous command-line toolkit and Python library designed for the professional-grade synthesis of generative AI media. It serves as a streamlined interface for Google’s state-of-the-art Gemini models (e.g., Nano Banana, Veo), enabling scalable, reproducible, and integrable workflows for both image and video generation.

## Target Audience
- **Python Developers:** Engineers who need a robust, asynchronous library to integrate generative AI capabilities into their own applications.
- **Content Creators:** Users who require a powerful CLI for high-volume, professional-quality image and video synthesis.
- **DevOps/Systems Engineers:** Infrastructure specialists building scalable, cloud-native media processing pipelines.

## Goals
- **High Performance/Scalability:** An asynchronous-first architecture (aiohttp, asyncio) that maximizes throughput and efficiency.
- **Reliability & Reproducibility:** Consistent generation results through metadata embedding and sidecar files, coupled with robust error handling.
- **Seamless Cloud Integration:** Native support for uploading generated assets directly to AWS S3, Google Cloud Storage, and Cloudflare R2, with webhook notification support.
- **Distributed Execution:** Instantly offload thousands of jobs to a managed cloud queue (Google Cloud Tasks) for massive scalability.

## Core Features
- **Advanced Batch Processing:** Support for managing and executing hundreds of generation jobs concurrently via CSV or JSONL input files.
- **Built-in Resilience:** Automatic retries with exponential backoff powered by the `tenacity` library to handle transient API failures.
- **Composable Python Library:** A well-structured API that allows the core generation engine to be easily imported and used in custom Python projects.
- **Multi-Modal Generation:** Native support for both image and video creation with fine-grained control over model parameters (seed, negative prompts, aspect ratio, etc.).
- **Rich Interactive UI:** A modern CLI experience featuring progress bars, status spinners, and formatted terminal output.
- **Pluggable Backends:** Support for multiple execution backends, including local execution and Google Cloud Tasks for distributed generation.

## Technical Constraints
- **Python Version Compatibility:** The project strictly targets Python 3.10 and newer environments.
- **Asynchronous-Only Core:** All core network and I/O operations must remain non-blocking to ensure maximum concurrency.
- **Strict Static Typing:** Full adherence to strict `mypy` and `basedpyright` type-checking rules to ensure code quality and maintainability.

## Future Roadmap
- **Serverless Workers:** Execution of generation tasks via scalable, serverless platforms like Google Cloud Run or AWS Lambda.
- **Decoupled Architecture:** A fully distributed, resilient pipeline integrating task queues, workers, cloud storage, and webhooks.
