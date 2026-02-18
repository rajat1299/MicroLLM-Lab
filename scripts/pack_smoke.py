#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.packs import resolve_docs
from shared.types import RunConfig
from worker.trainer import train_tiny_gpt

PACK_IDS = ["regex", "abc_music", "chess_pgn", "sql_snippets", "arithmetic", "json"]


def run_pack_smoke(pack_id: str, steps: int) -> None:
    docs = resolve_docs(pack_id, upload_text=None)
    losses: list[float] = []
    has_non_empty_sample = False

    config = RunConfig(
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=16,
        num_steps=steps,
        learning_rate=0.01,
        temperature=0.9,
        seed=42,
        sample_count=8,
        sample_interval=40,
        top_k=3,
        op_graph_token_index=0,
        op_graph_step_interval=20,
    )

    def emit_event(event_type: str, payload: dict) -> None:
        nonlocal has_non_empty_sample
        if event_type == "step.loss":
            losses.append(float(payload["loss"]))
        if event_type == "sample.generated":
            samples = payload.get("samples", [])
            if any(isinstance(sample, str) and sample.strip() for sample in samples):
                has_non_empty_sample = True

    result = train_tiny_gpt(
        docs=docs,
        config=config,
        emit_event=emit_event,
        is_cancel_requested=lambda: False,
    )
    if result.status != "completed":
        raise ValueError(f"{pack_id}: training did not complete")

    if len(losses) < 100:
        raise ValueError(f"{pack_id}: expected >=100 loss points, got {len(losses)}")

    start_median = statistics.median(losses[:20])
    end_median = statistics.median(losses[-20:])
    if end_median >= start_median:
        raise ValueError(
            f"{pack_id}: loss trend check failed (start_median={start_median:.4f}, end_median={end_median:.4f})"
        )

    if not has_non_empty_sample:
        raise ValueError(f"{pack_id}: no non-empty generated samples observed")

    print(
        f"{pack_id}: ok "
        f"(start_median={start_median:.4f}, end_median={end_median:.4f}, "
        f"final_loss={result.final_loss:.4f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pack smoke checks for MicroLLM Lab.")
    parser.add_argument("--steps", type=int, default=120, help="Training steps per pack (default: 120)")
    parser.add_argument(
        "--pack",
        action="append",
        choices=PACK_IDS,
        help="Optional pack ID to run (can be provided multiple times).",
    )
    args = parser.parse_args()

    selected = args.pack if args.pack else PACK_IDS
    for pack_id in selected:
        run_pack_smoke(pack_id, steps=args.steps)

    print("Pack smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
