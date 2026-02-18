from __future__ import annotations

from pathlib import Path

RUN_LIMITS = {
    "upload_max_bytes": 200 * 1024,
    "upload_max_unique_chars": 256,
    "corpus_max_chars": 200_000,
    "num_steps_max": 2_000,
    "block_size_max": 64,
    "n_embd_max": 64,
    "n_layer_max": 2,
    "n_head_max": 8,
    "concurrent_runs_max": 3,
    "ttl_seconds": 24 * 60 * 60,
}

TERMINAL_STATUSES = {"completed", "failed", "canceled"}
ACTIVE_STATUSES = {"queued", "running"}

BUILTIN_PACK_IDS = ["regex", "abc_music", "chess_pgn", "sql_snippets", "arithmetic", "json"]

PACK_DIR = Path(__file__).resolve().parents[1] / "packs"

UPLOAD_ALLOWED_EXTENSIONS = {".txt"}

CONTENT_BLOCKLIST = {
    "-----BEGIN PRIVATE KEY-----",
    "<script>",
    "DROP DATABASE",
    "rm -rf /",
}
