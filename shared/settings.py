from __future__ import annotations

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
API_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "30"))
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173")
