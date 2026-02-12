from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from redis import Redis

from shared.constants import ACTIVE_STATUSES, RUN_LIMITS
from shared.types import RunConfig, RunSummary


class RunStore:
    def __init__(self, redis: Redis):
        self.redis = redis

    def _meta_key(self, run_id: str) -> str:
        return f"run:{run_id}:meta"

    def _events_key(self, run_id: str) -> str:
        return f"run:{run_id}:events"

    def _seq_key(self, run_id: str) -> str:
        return f"run:{run_id}:seq"

    def _cancel_key(self, run_id: str) -> str:
        return f"run:{run_id}:cancel"

    def _upload_key(self, upload_id: str) -> str:
        return f"upload:{upload_id}:text"

    def _upload_meta_key(self, upload_id: str) -> str:
        return f"upload:{upload_id}:meta"

    def _set_expiry(self, *keys: str) -> None:
        for key in keys:
            self.redis.expire(key, RUN_LIMITS["ttl_seconds"])

    def create_run(self, pack_id: str, config: RunConfig) -> RunSummary:
        run_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        meta = {
            "run_id": run_id,
            "status": "queued",
            "pack_id": pack_id,
            "config_json": config.model_dump_json(),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "error": "",
        }
        self.redis.sadd("runs:index", run_id)
        self.redis.hset(self._meta_key(run_id), mapping=meta)
        self.redis.set(self._seq_key(run_id), 0)
        self._set_expiry("runs:index", self._meta_key(run_id), self._seq_key(run_id))
        return RunSummary(
            run_id=run_id,
            status="queued",
            pack_id=pack_id,
            config=config,
            created_at=now,
            updated_at=now,
            error=None,
        )

    def get_run(self, run_id: str) -> RunSummary | None:
        raw = self.redis.hgetall(self._meta_key(run_id))
        if not raw:
            return None
        error = raw.get("error") or None
        return RunSummary(
            run_id=raw["run_id"],
            status=raw["status"],
            pack_id=raw["pack_id"],
            config=RunConfig.model_validate_json(raw["config_json"]),
            created_at=datetime.fromisoformat(raw["created_at"]),
            updated_at=datetime.fromisoformat(raw["updated_at"]),
            error=error,
        )

    def update_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        mapping: dict[str, Any] = {
            "status": status,
            "updated_at": now,
            "error": error or "",
        }
        self.redis.hset(self._meta_key(run_id), mapping=mapping)
        self._set_expiry(self._meta_key(run_id), self._events_key(run_id), self._seq_key(run_id))

    def list_runs(self) -> list[RunSummary]:
        run_ids = sorted(self.redis.smembers("runs:index"))
        runs: list[RunSummary] = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is not None:
                runs.append(run)
        return runs

    def count_active_runs(self) -> int:
        return sum(1 for run in self.list_runs() if run.status in ACTIVE_STATUSES)

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        seq = int(self.redis.incr(self._seq_key(run_id)))
        event = {
            "seq": seq,
            "type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        self.redis.rpush(self._events_key(run_id), json.dumps(event))
        self._set_expiry(self._events_key(run_id), self._seq_key(run_id))
        return event

    def list_events(self, run_id: str, from_seq: int = 1) -> list[dict[str, Any]]:
        start_index = max(from_seq - 1, 0)
        raw_events = self.redis.lrange(self._events_key(run_id), start_index, -1)
        events: list[dict[str, Any]] = []
        for item in raw_events:
            event = json.loads(item)
            if event["seq"] >= from_seq:
                events.append(event)
        return events

    def request_cancel(self, run_id: str) -> None:
        self.redis.set(self._cancel_key(run_id), "1", ex=RUN_LIMITS["ttl_seconds"])

    def is_cancel_requested(self, run_id: str) -> bool:
        return self.redis.get(self._cancel_key(run_id)) == "1"

    def create_upload(self, text: str) -> tuple[str, datetime, int, int]:
        upload_id = uuid.uuid4().hex
        docs = [line for line in text.splitlines() if line.strip()]
        now = datetime.now(UTC)
        expires_at = now.timestamp() + RUN_LIMITS["ttl_seconds"]
        self.redis.set(self._upload_key(upload_id), text, ex=RUN_LIMITS["ttl_seconds"])
        self.redis.hset(
            self._upload_meta_key(upload_id),
            mapping={
                "document_count": str(len(docs)),
                "character_count": str(len(text)),
                "expires_at": str(expires_at),
            },
        )
        self._set_expiry(self._upload_meta_key(upload_id))
        return upload_id, datetime.fromtimestamp(expires_at, UTC), len(docs), len(text)

    def get_upload_text(self, upload_id: str) -> str | None:
        text = self.redis.get(self._upload_key(upload_id))
        return text if text else None
