from __future__ import annotations

import logging

import sentry_sdk

from shared.packs import resolve_docs
from shared.redis_client import get_redis
from shared.store import RunStore
from worker.trainer import train_tiny_gpt

logger = logging.getLogger(__name__)


def train_run_job(run_id: str) -> None:
    redis = get_redis()
    store = RunStore(redis)
    run = store.get_run(run_id)
    if run is None:
        logger.error("run_not_found", extra={"run_id": run_id})
        return

    if store.is_cancel_requested(run_id):
        store.update_run_status(run_id, "canceled")
        return

    try:
        store.update_run_status(run_id, "running")

        upload_text = None
        if run.pack_id.startswith("upload:"):
            upload_id = run.pack_id.split("upload:", 1)[1]
            upload_text = store.get_upload_text(upload_id)

        docs = resolve_docs(run.pack_id, upload_text)

        def emit(event_type: str, payload: dict) -> None:
            store.append_event(run_id, event_type, payload)

        result = train_tiny_gpt(
            docs=docs,
            config=run.config,
            emit_event=emit,
            is_cancel_requested=lambda: store.is_cancel_requested(run_id),
        )

        if result.status == "canceled":
            store.update_run_status(run_id, "canceled")
            return

        store.append_event(
            run_id,
            "run.completed",
            {
                "steps_completed": result.steps_completed,
                "final_loss": round(result.final_loss, 6),
                "vocab_size": result.vocab_size,
            },
        )
        store.update_run_status(run_id, "completed")
    except Exception as exc:  # pragma: no cover - fail-safe path
        logger.exception("run_failed", extra={"run_id": run_id})
        sentry_sdk.capture_exception(exc)
        store.append_event(run_id, "run.failed", {"error": str(exc)})
        store.update_run_status(run_id, "failed", error=str(exc))
