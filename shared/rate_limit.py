from __future__ import annotations

from redis import Redis


def allow_request(redis: Redis, key: str, limit: int, window_seconds: int) -> bool:
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window_seconds)
    return count <= limit
