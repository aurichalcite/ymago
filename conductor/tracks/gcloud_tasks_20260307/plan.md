# Implementation Plan: Google Cloud Tasks Backend

## Phase 1: Infrastructure & Configuration [checkpoint: c3ef52a]
Implement the configuration schema and the basic skeleton of the `CloudTasksExecutionBackend`.

- [x] Task: Define `CloudTasksConfig` in `src/ymago/config.py`. [f95084e]
    - [ ] Add `gct_project_id`, `gct_location`, `gct_queue_name`, `worker_url`, `service_account_email` fields.
- [x] Task: Implement `CloudTasksExecutionBackend` skeleton in `src/ymago/core/backends.py`. [e7e2521]
    - [ ] Inherit from `ExecutionBackend`.
    - [ ] Implement `__init__` and stub methods for `submit`, `process_batch`, and `get_status`.
- [x] Task: Conductor - User Manual Verification 'Infrastructure & Configuration' (Protocol in workflow.md) [79e0ea5]

## Phase 2: Task Dispatching Implementation [checkpoint: 0a4969b]
Implement the logic to create tasks in the GCT queue and serialize the job payload.

- [x] Task: Integrate `google-cloud-tasks` client. [48afbf2]
    - [x] Write tests for task payload serialization.
    - [x] Implement `_create_gct_task` helper method in `CloudTasksExecutionBackend`.
- [x] Task: Implement `submit(jobs)` method. [fe61689]
    - [x] Write tests for submitting multiple jobs.
    - [x] Implement dispatching logic with OIDC authentication support.
- [x] Task: Implement `process_batch(...)` method. [28f8483]
    - [x] Write tests for batch processing with the cloud backend.
    - [x] Implement asynchronous generator consumption and task dispatching.
- [x] Task: Conductor - User Manual Verification 'Task Dispatching Implementation' (Protocol in workflow.md) [913e4b2]

## Phase 3: CLI Integration & Final Verification
Update the CLI to allow selecting the new backend and perform final validation.

- [x] Task: Update CLI to support `--backend cloud-tasks`. [44f8e1f]
    - [ ] Modify `src/ymago/cli.py` to handle backend selection.
    - [ ] Update `src/ymago/core/backends.py` factory function (if one exists) or logic.
- [ ] Task: Integration tests with Mock GCT Client.
    - [ ] Create `tests/core/test_gct_backend.py`.
    - [ ] Verify full flow from CLI to mock task creation.
- [ ] Task: Update documentation.
    - [ ] Add GCT setup guide to `README.md` or a new documentation file.
- [ ] Task: Conductor - User Manual Verification 'CLI Integration & Final Verification' (Protocol in workflow.md)
