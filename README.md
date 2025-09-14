[![CI](https://github.com/aurichalcite/ymago/actions/workflows/ci.yml/badge.svg)](https://github.com/aurichalcite/ymago/actions/workflows/ci.yml)
# ymago: Advanced Generative AI Media Toolkit


ymago is a powerful, asynchronous command-line interface (CLI) and Python library designed for developers and creators. It provides a sophisticated, streamlined workflow to harness Google's cutting-edge generative AI models, including Nano Banana for image synthesis and Veo for video generation.
Engineered to the highest professional standards, ymago is more than just a CLI wrapper. It's a complete toolkit for scalable, reproducible, and integrable media generation.
________________


## ✨ Key Features


ymago is packed with features designed for power, flexibility, and a superior user experience.
* Multi-Modal Generation: Natively supports both image and video creation using state-of-the-art Gemini models.
* Rich Interactive CLI: Enjoy a modern terminal experience with beautiful progress bars, status spinners, and formatted output powered by rich.
* Asynchronous Core: Built with aiohttp for non-blocking, concurrent API requests, enabling high throughput for batch operations.
* Flexible Inputs: Generate from text prompts or image URLs. ymago automatically downloads and processes remote images for you.
* Powerful Batch Processing: Submit hundreds of generation jobs from a single CSV or JSON file, perfect for large-scale asset creation.
* Advanced Generation Control: Fine-tune your creations with direct CLI access to model parameters like --seed, --negative-prompt, --aspect-ratio, and more.
* Composable Python API: Beyond the CLI, ymago is a well-structured library. Import and use its core functions in your own Python applications for programmatic generation.
* Cloud Integration: Save generated media directly to Cloudflare R2,  Google Cloud Storage or AWS S3, returning the public URL for seamless integration into cloud-native workflows.
* Webhook Notifications: For long-running jobs, configure a webhook URL to receive a notification the moment your media is ready.
* Configuration Files: Set default models, output paths, and API keys in a ymago.toml file for a cleaner, faster workflow.
* Built-in Resilience: Automatically retries failed API requests with exponential backoff using tenacity.
* Metadata Embedding: Automatically saves the generation prompt and key parameters to a JSON sidecar file, ensuring your creations are always reproducible.
________________


## 🛠️ Technical Stack


ymago is built with a modern, high-performance Python stack:
* CLI Framework: typer
* Interactive UI: rich
* Package Management: uv
* Linting & Formatting: ruff
* Testing: pytest & pytest-cov
* HTTP Client: aiohttp
* Static Typing: mypy
* Documentation: sphinx
* Resilience: tenacity
* Core AI Integration: google-genai
________________


## 🚀 Installation


Ensure you have Python 3.10+ and uv installed.
### 1. Clone the repository:
```bash
git clone https://github.com/aurichalcite/ymago.git
cd ymago
````

### 2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Configure your Google AI API Key:
Export your API key as an environment variable or set it in a ymago.toml file.
Bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"

________________


## 💻 Usage Examples




### 1. Basic Image Generation


Generate an image with a simple prompt and see the interactive progress.



```bash
ymago image "A photorealistic portrait of an astronaut on Mars" --output-path ./astro.png
```


### 2. Advanced Image Generation


Use advanced parameters and upload the result directly to a Google Cloud Storage bucket with webhook notification.


```bash
ymago image "A majestic wolf howling at a full moon, fantasy art" \
 --negative-prompt "daylight, clouds, cartoon" \
 --seed 42 \
 --destination gs://my-art-bucket/wolves/ \
 --webhook-url https://api.example.com/webhook
```


### 3. Batch Processing from a File


Define multiple jobs in a prompts.csv file and run them all at once.


```csv
prompt,output_name
"A sunny beach with a palm tree","beach_day"
"A dark, rainy city street at night","rainy_night"
```


```bash
ymago batch --from-file ./prompts.csv --output-dir ./generated_images/
```


### 4. Using ymago as a Python Library


The core logic is easily importable into your own projects.


```python
import asyncio
from ymago.core import generate_image
from ymago.config import settings
```

# Ensure API key is configured in the environment or a config file
settings.load_config()

async def main():
   image_data = await generate_image(
       prompt="A vibrant coral reef teeming with life",
       model="gemini-nano-banana"
   )
   with open("coral_reef.png", "wb") as f:
       f.write(image_data)

if __name__ == "__main__":
   asyncio.run(main())

________________


## ☁️ Cloud Storage & Webhook Configuration


ymago supports direct upload to cloud storage providers and webhook notifications for production workflows.

### Cloud Storage Providers

#### AWS S3
```bash
# Set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Generate and upload to S3
ymago image "A sunset over mountains" --destination s3://my-bucket/images/
```

#### Google Cloud Storage
```bash
# Set service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Generate and upload to GCS
ymago image "A forest in autumn" --destination gs://my-bucket/images/
```

#### Cloudflare R2
```bash
# Set R2 credentials
export R2_ACCOUNT_ID="your-account-id"
export R2_ACCESS_KEY_ID="your-access-key"
export R2_SECRET_ACCESS_KEY="your-secret-key"

# Generate and upload to R2
ymago image "A city skyline" --destination r2://my-bucket/images/
```

### Webhook Notifications

Configure webhooks to receive notifications when generation jobs complete:

```bash
ymago image "A space station" \
  --destination s3://my-bucket/images/ \
  --webhook-url https://api.example.com/webhook
```

#### Webhook Payload

Your webhook endpoint will receive a JSON payload:

```json
{
  "job_id": "unique-job-identifier",
  "job_status": "success",
  "output_url": "s3://my-bucket/images/generated-image.png",
  "processing_time_seconds": 5.2,
  "file_size_bytes": 1048576,
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "model": "gemini-nano-banana",
    "prompt": "A space station",
    "storage_backend": "cloud"
  }
}
```

For failed jobs:
```json
{
  "job_id": "unique-job-identifier",
  "job_status": "failure",
  "error_message": "API quota exceeded",
  "processing_time_seconds": 1.2,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Configuration File

Add cloud storage and webhook settings to your `ymago.toml`:

```toml
[cloud_storage]
aws_access_key_id = "your-access-key"
aws_secret_access_key = "your-secret-key"
aws_region = "us-east-1"

gcp_service_account_path = "/path/to/service-account.json"

r2_account_id = "your-account-id"
r2_access_key_id = "your-access-key"
r2_secret_access_key = "your-secret-key"

[webhooks]
enabled = true
timeout_seconds = 30
retry_attempts = 3
retry_backoff_factor = 2.0
```

### Installation with Cloud Support

Install ymago with cloud storage dependencies:

```bash
# For AWS S3 support
pip install "ymago[aws]"

# For Google Cloud Storage support
pip install "ymago[gcp]"

# For Cloudflare R2 support
pip install "ymago[r2]"

# For all cloud providers
pip install "ymago[cloud]"
```

________________


## 🏗️ Architecture & Future Roadmap


ymago is architected for scalability and extensibility, with a clean separation between the CLI interface and the core generation service.
The future roadmap focuses on building a robust, distributed generation system:
   * Pluggable Backends: The current "local" execution backend will be complemented by a Google Cloud Tasks backend. This will allow the CLI to instantly offload thousands of tasks to a managed queue, with a separate fleet of cloud workers handling the generation. The architecture is designed to easily support other backends like Celery in the future.
   * Serverless Workers: Generation tasks will be executed by scalable, serverless workers (e.g., Google Cloud Run or AWS Lambda), ensuring you only pay for what you use.
   * Decoupled Workflow: The combination of a task queue, serverless workers, cloud storage, and webhooks creates a fully decoupled, resilient, and scalable media generation pipeline suitable for production use.