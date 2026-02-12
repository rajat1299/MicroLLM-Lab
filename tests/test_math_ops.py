from __future__ import annotations

from shared.types import RunConfig
from worker.trainer import Value, softmax, train_tiny_gpt


def test_softmax_outputs_sum_to_one() -> None:
    logits = [Value(1.0), Value(0.0), Value(-1.0)]
    probs = softmax(logits)
    total = sum(p.data for p in probs)
    assert abs(total - 1.0) < 1e-9


def test_attention_weights_are_normalized() -> None:
    captured: list[dict] = []

    def emit(event_type: str, payload: dict) -> None:
        captured.append({"type": event_type, "payload": payload})

    config = RunConfig(
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=4,
        num_steps=1,
        sample_interval=1,
        op_graph_step_interval=1,
    )
    result = train_tiny_gpt(["ab", "ac"], config, emit, lambda: False)

    assert result.status == "completed"
    attention_events = [event for event in captured if event["type"] == "step.attention"]
    assert attention_events

    first = attention_events[0]["payload"]
    token_attention = first["token_attention"]
    for position_entry in token_attention:
        for head_weights in position_entry["heads"]:
            assert abs(sum(head_weights) - 1.0) < 1e-5
