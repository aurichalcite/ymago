# Specification: Google Cloud Tasks Backend

## Objective
Implement a new `ExecutionBackend` for `ymago` that leverages Google Cloud Tasks (GCT) for distributed, asynchronous generation. This allows the CLI to offload generation jobs to a managed queue, which can then be processed by a scalable fleet of workers (e.g., on Cloud Run or GKE).

## Requirements
1. **New Backend Class:** Implement `CloudTasksExecutionBackend` in `src/ymago/core/backends.py`.
2. **Configuration:** Add settings for:
    - `gct_project_id`
    - `gct_location`
    - `gct_queue_name`
    - `worker_url` (The endpoint where the worker receives the task)
    - `service_account_email` (For OIDC token authentication)
3. **Task Dispatching:**
    - Use the `google-cloud-tasks` Python library.
    - Serialize `GenerationJob` into a JSON payload.
    - Set up OIDC authentication for the HTTP target.
4. **Resilience:**
    - Handle GCT API errors with retries.
    - Ensure tasks are created with unique IDs to prevent duplicate dispatching.
5. **CLI Integration:**
    - Allow users to specify `--backend cloud-tasks` (or via config).
6. **Documentation:**
    - Update documentation with instructions for setting up the GCT queue and workers.

## Technical Design
- **`submit(jobs)`**: Iterates through the provided jobs and creates a task in the specified GCT queue for each.
- **`process_batch(...)`**: Similar to `submit`, but operates on a generator of requests. It will offload the rate limiting and concurrency to the GCT queue configuration.
- **Payload Structure:**
    ```json
    {
      "request_id": "...",
      "prompt": "...",
      "model": "...",
      "parameters": {...},
      "destination": "...",
      "webhook_url": "..."
    }
    ```
- **Authentication:** Use `oidc_token` in the `http_request` of the GCT task to securely call the worker endpoint.

## Dependencies
- `google-cloud-tasks`
