# TinyLLM Lab

TinyLLM Lab is an education-first web app that trains tiny GPT-style models on small domain corpora and visualizes forward pass, attention, backward gradients, and selected-token operation graphs in real time.

## Stack
- Frontend: React + TypeScript + Vite
- API: FastAPI
- Worker: Python + RQ
- Queue/Store: Redis

## Features in this implementation
- Curated packs: `regex`, `abc_music`, `chess_pgn`, `sql_snippets`
- Run lifecycle API with queue-backed training jobs
- SSE event stream with replay support
- Token-level visualizer and selected-token op-graph panel
- Constrained uploads with strict validation
- Hard run limits and simple rate limiting

## Quickstart
See `/Users/rajattiwari/microgpt/docs/quickstart.md`.
