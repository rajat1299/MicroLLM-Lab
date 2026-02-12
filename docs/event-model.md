# Event Model

Each event includes:
- `seq`: monotonically increasing integer per run
- `type`: event name
- `timestamp`: ISO-8601 timestamp
- `payload`: typed event payload

## Event types
1. `run.started`
2. `step.forward`
3. `step.attention`
4. `step.loss`
5. `step.backward`
6. `step.update`
7. `sample.generated`
8. `run.completed`
9. `run.failed`
10. `run.canceled`

## Replay semantics
- SSE supports replay via `from_seq` query and `Last-Event-ID` header.
- Stream ordering is guaranteed by `seq` and Redis append order.
