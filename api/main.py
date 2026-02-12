from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import sentry_sdk
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from redis import Redis

from api.queue import RedisRunQueue, RunQueue
from shared.constants import BUILTIN_PACK_IDS, RUN_LIMITS, TERMINAL_STATUSES
from shared.packs import build_pack_descriptors
from shared.rate_limit import allow_request
from shared.redis_client import get_redis
from shared.settings import API_RATE_LIMIT_PER_MINUTE, CORS_ALLOW_ORIGINS, SENTRY_DSN
from shared.store import RunStore
from shared.types import RunCreateRequest, RunSummary, UploadResponse
from shared.validation import UploadValidationError, validate_upload


if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.0)


def create_app(store: RunStore | None = None, queue: RunQueue | None = None) -> FastAPI:
    app = FastAPI(title="TinyLLM Lab API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    redis = store.redis if store else get_redis()
    app.state.redis = redis
    app.state.store = store or RunStore(redis)
    app.state.queue = queue or RedisRunQueue(redis)

    def _store(request: Request) -> RunStore:
        return request.app.state.store

    def _redis(request: Request) -> Redis:
        return request.app.state.redis

    def _queue(request: Request) -> RunQueue:
        return request.app.state.queue

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/packs")
    def list_packs() -> list[dict]:
        return [pack.model_dump(mode="json") for pack in build_pack_descriptors()]

    @app.post("/api/v1/uploads", response_model=UploadResponse)
    async def create_upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
        redis_client = _redis(request)
        ip = request.client.host if request.client else "unknown"
        if not allow_request(
            redis_client,
            key=f"rl:upload:{ip}",
            limit=API_RATE_LIMIT_PER_MINUTE,
            window_seconds=60,
        ):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        filename = file.filename or "upload.txt"
        content = await file.read()
        try:
            text = validate_upload(filename, content)
        except UploadValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        store_client = _store(request)
        upload_id, expires_at, doc_count, char_count = store_client.create_upload(text)
        return UploadResponse(
            upload_id=upload_id,
            document_count=doc_count,
            character_count=char_count,
            expires_at=expires_at,
        )

    @app.post("/api/v1/runs", response_model=RunSummary)
    def create_run(request: Request, body: RunCreateRequest) -> RunSummary:
        redis_client = _redis(request)
        ip = request.client.host if request.client else "unknown"
        if not allow_request(
            redis_client,
            key=f"rl:runs:{ip}",
            limit=API_RATE_LIMIT_PER_MINUTE,
            window_seconds=60,
        ):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        store_client = _store(request)
        if store_client.count_active_runs() >= RUN_LIMITS["concurrent_runs_max"]:
            raise HTTPException(
                status_code=429,
                detail=f"Max concurrent runs is {RUN_LIMITS['concurrent_runs_max']}",
            )

        pack_id = body.pack_id
        if pack_id not in BUILTIN_PACK_IDS and not pack_id.startswith("upload:"):
            raise HTTPException(status_code=400, detail=f"Unsupported pack_id: {pack_id}")

        if pack_id.startswith("upload:"):
            upload_id = pack_id.split("upload:", 1)[1]
            if not store_client.get_upload_text(upload_id):
                raise HTTPException(status_code=404, detail="Upload not found or expired")

        run = store_client.create_run(pack_id=pack_id, config=body.config)
        queue_client = _queue(request)
        queue_client.enqueue_run(run.run_id)
        return run

    @app.get("/api/v1/runs/{run_id}", response_model=RunSummary)
    def get_run(run_id: str, request: Request) -> RunSummary:
        run = _store(request).get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel_run(run_id: str, request: Request) -> dict[str, str]:
        store_client = _store(request)
        run = store_client.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status in TERMINAL_STATUSES:
            return {"status": run.status}

        store_client.request_cancel(run_id)
        store_client.append_event(
            run_id,
            "run.canceled",
            {"requested_at": datetime.now(UTC).isoformat()},
        )
        return {"status": "cancel_requested"}

    @app.get("/api/v1/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        from_seq: int = Query(default=1, ge=1),
    ) -> StreamingResponse:
        store_client = _store(request)
        run = store_client.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        last_event_id = request.headers.get("last-event-id")
        if last_event_id:
            try:
                from_seq = max(from_seq, int(last_event_id) + 1)
            except ValueError:
                pass

        async def event_generator():
            cursor = from_seq
            while True:
                if await request.is_disconnected():
                    return

                events = store_client.list_events(run_id, from_seq=cursor)
                for event in events:
                    cursor = event["seq"] + 1
                    payload = json.dumps(event)
                    yield f"id: {event['seq']}\nevent: {event['type']}\ndata: {payload}\n\n"

                run_state = store_client.get_run(run_id)
                if run_state and run_state.status in TERMINAL_STATUSES and not events:
                    return

                if not events:
                    yield ": ping\n\n"
                    await asyncio.sleep(0.5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
