from __future__ import annotations

from typing import Protocol

from rq import Queue

from shared.constants import RUN_LIMITS
from shared.redis_client import get_redis_raw


class RunQueue(Protocol):
    def enqueue_run(self, run_id: str) -> str:
        ...


class RedisRunQueue:
    def __init__(self):
        # RQ job payloads are binary-serialized; use a raw Redis client here.
        self.queue = Queue("microllm", connection=get_redis_raw())

    def enqueue_run(self, run_id: str) -> str:
        job = self.queue.enqueue(
            "worker.jobs.train_run_job",
            run_id,
            job_timeout=60 * 30,
            result_ttl=RUN_LIMITS["ttl_seconds"],
            failure_ttl=RUN_LIMITS["ttl_seconds"],
        )
        return job.id
