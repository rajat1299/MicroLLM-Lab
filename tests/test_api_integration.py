from __future__ import annotations

import json

import fakeredis
from fastapi.testclient import TestClient

from api.main import create_app
from shared.packs import resolve_docs
from shared.store import RunStore
from shared.types import RunConfig
from worker.trainer import train_tiny_gpt


class InlineQueue:
    def __init__(self, store: RunStore):
        self.store = store

    def enqueue_run(self, run_id: str) -> str:
        run = self.store.get_run(run_id)
        if run is None:
            raise RuntimeError("run not found")

        self.store.update_run_status(run_id, "running")
        upload_text = None
        if run.pack_id.startswith("upload:"):
            upload_id = run.pack_id.split("upload:", 1)[1]
            upload_text = self.store.get_upload_text(upload_id)
        docs = resolve_docs(run.pack_id, upload_text)

        result = train_tiny_gpt(
            docs=docs,
            config=run.config,
            emit_event=lambda event_type, payload: self.store.append_event(
                run_id, event_type, payload
            ),
            is_cancel_requested=lambda: self.store.is_cancel_requested(run_id),
        )

        if result.status == "completed":
            self.store.append_event(
                run_id,
                "run.completed",
                {
                    "steps_completed": result.steps_completed,
                    "final_loss": round(result.final_loss, 6),
                    "vocab_size": result.vocab_size,
                },
            )
            self.store.update_run_status(run_id, "completed")
        else:
            self.store.update_run_status(run_id, "canceled")
        return "inline-job"


def _small_config() -> dict:
    return RunConfig(
        n_embd=8,
        n_head=2,
        n_layer=1,
        block_size=8,
        num_steps=2,
        sample_interval=1,
        op_graph_step_interval=1,
    ).model_dump(mode="json")


def test_run_lifecycle_and_event_stream() -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    store = RunStore(redis)
    app = create_app(store=store, queue=InlineQueue(store))
    client = TestClient(app)

    create_response = client.post(
        "/api/v1/runs",
        json={"pack_id": "regex", "config": _small_config()},
    )
    assert create_response.status_code == 200
    run_id = create_response.json()["run_id"]

    run_response = client.get(f"/api/v1/runs/{run_id}")
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "completed"

    with client.stream("GET", f"/api/v1/runs/{run_id}/events") as stream_response:
        text = "".join(chunk for chunk in stream_response.iter_text())

    assert "event: run.started" in text
    assert "event: run.completed" in text

    event_ids = []
    for line in text.splitlines():
        if line.startswith("id: "):
            event_ids.append(int(line.split("id: ", 1)[1]))
    assert event_ids == sorted(event_ids)


def test_upload_and_run_flow() -> None:
    redis = fakeredis.FakeRedis(decode_responses=True)
    store = RunStore(redis)
    app = create_app(store=store, queue=InlineQueue(store))
    client = TestClient(app)

    upload_response = client.post(
        "/api/v1/uploads",
        files={"file": ("custom.txt", b"hello\nworld\n", "text/plain")},
    )
    assert upload_response.status_code == 200
    upload_id = upload_response.json()["upload_id"]

    run_response = client.post(
        "/api/v1/runs",
        json={"pack_id": f"upload:{upload_id}", "config": _small_config()},
    )
    assert run_response.status_code == 200
    run_id = run_response.json()["run_id"]

    details = client.get(f"/api/v1/runs/{run_id}")
    assert details.status_code == 200
    assert details.json()["status"] == "completed"
