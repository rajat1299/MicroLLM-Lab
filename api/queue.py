from __future__ import annotations

from typing import Protocol

from redis import Redis
from rq import Queue

from shared.constants import RUN_LIMITS


class RunQueue(Protocol):
    def enqueue_run(self, run_id: str) -> str:
        ...


class RedisRunQueue:
    def __init__(self, redis: Redis):
        self.queue = Queue("microllm", connection=redis)

    def enqueue_run(self, run_id: str) -> str:
        job = self.queue.enqueue(
            "worker.jobs.train_run_job",
            run_id,
            job_timeout=60 * 30,
            result_ttl=RUN_LIMITS["ttl_seconds"],
            failure_ttl=RUN_LIMITS["ttl_seconds"],
        )
        return job.id
