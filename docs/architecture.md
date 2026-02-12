# Architecture

## Services
1. Frontend (`frontend`): React client for run setup and visualization.
2. API (`api`): FastAPI service exposing run/upload/stream endpoints.
3. Worker (`worker`): RQ consumer that executes training jobs.
4. Redis: queue backend + ephemeral run/upload/event storage.

## Data flow
1. Frontend calls `POST /api/v1/runs`.
2. API validates limits and enqueues `worker.jobs.train_run_job`.
3. Worker trains tiny GPT and appends ordered events to Redis.
4. Frontend opens SSE stream `GET /api/v1/runs/{run_id}/events`.
5. Frontend renders live token, attention, gradient, and op-graph views.

## Storage model (ephemeral)
- `run:{id}:meta` hash
- `run:{id}:events` list
- `run:{id}:seq` integer
- `upload:{id}:text` string
- `upload:{id}:meta` hash
- All keys expire after 24h.
