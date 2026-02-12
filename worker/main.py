from __future__ import annotations

import logging

import sentry_sdk
from redis import Redis
from rq import Connection, Worker

from shared.redis_client import get_redis
from shared.settings import SENTRY_DSN


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.0)


def run_worker() -> None:
    redis: Redis = get_redis()
    with Connection(redis):
        worker = Worker(["microllm"])
        worker.work(with_scheduler=False)


if __name__ == "__main__":
    run_worker()
