# MicroLLM Lab

**Train tiny transformers. Watch them think.**

A live, open-source laboratory for training and inspecting small language models from scratch. Built for learning.

---

## What This Is

Most educational ML tools either show you pre-trained models or hide the training process behind abstractions. MicroLLM Lab does neither.

You configure a tiny GPT (we're talking 2 layers, 64 dimensions—small enough to train in seconds), watch every forward pass token-by-token, then step through the backward pass operation-by-operation. See the gradients flow. Watch the attention heads specialize. Actually understand what "learning" means at the tensor level.


---

## Live Demo

**WIP*

---

## The Six Packs

We ship with six curated datasets, each designed to teach a specific structural pattern under strict, deterministic templates:

| Pack | What It Teaches | Corpus Size |
|------|----------------|-------------|
| **regex** | Pattern matching with constrained email-style regexes | 60 lines (~1.4K chars) |
| **abc_music** | Repetition and motif composition in fixed bars | 60 lines (~1.9K chars) |
| **chess_pgn** | Move-sequence dependencies in 3-move openings | 60 lines (~1.8K chars) |
| **sql_snippets** | Query form consistency (`SELECT ... WHERE ...`) | 60 lines (~2.5K chars) |
| **arithmetic** | Symbolic completion for `A±B=C` relations | 60 lines (~0.4K chars) |
| **json** | Key/value structure adherence in compact objects | 60 lines (~1.6K chars) |

Pack templates, constraints, and regeneration commands are documented in `/packs/README.md`.

---

## Quick Start (Local)

```bash
# Clone
git clone https://github.com/rajat1299/MicroLLM-Lab.git
cd MicroLLM-Lab

# Install (from project root; requires Python 3.11+, Node.js 20+, Redis)
pip install -r requirements.txt && npm install

# Start backend (terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start worker (terminal 2)
python -m worker.main

# Start frontend (terminal 3)
npm run frontend:dev
```

Visit `http://localhost:5173`. Create a run. Watch the SSE stream light up.

---

## Architecture

```
┌─────────────┐      HTTP/SSE       ┌─────────────┐
│   Vercel    │◄───────────────────►│   Railway   │
│  (React)    │                     │  (FastAPI)  │
└─────────────┘                     └──────┬──────┘
                                           │
                                    ┌──────┴──────┐
                                    │    Redis    │
                                    │  (queue +   │
                                    │   state)    │
                                    └──────┬──────┘
                                           │
                                    ┌──────┴──────┐
                                    │   Railway   │
                                    │   (Worker)  │
                                    └─────────────┘
```

- **Frontend**: React + TypeScript + Vite. Real-time visualization using canvas for attention heatmaps and SVG for operation graphs.
- **API**: FastAPI handling run lifecycle, validation, and event streaming.
- **Worker**: Pure Python training loop with instrumentation hooks. No PyTorch—just NumPy and explicit autograd for transparency.
- **State**: Redis with 24h TTL. Ephemeral by design.

---

## Visualization Modes

**Token-level** (default): Watch the model predict one token at a time. See top-k probabilities, attention patterns per head, loss curves, and generated samples.

**Operation-level** (select any token): Inspect the full computation graph for that specific forward pass. Step through backpropagation. See gradients attach to every multiply, add, and softmax. Understand *why* that token's loss changed the weights it did.

---

## Constraints (By Design)

We enforce hard limits to keep training fast, safe, and comprehensible:

| Limit | Value | Why |
|-------|-------|-----|
| Max steps | 2,000 | Minutes, not hours |
| Max block size | 64 | Fits in working memory |
| Max embedding dim | 64 | Visualizable attention |
| Max layers | 2 | Traceable backprop |
| Upload size | 200KB | Forces curation |
| Concurrent runs | 3 | Shared infra fairness |

These aren't arbitrary. They're the bounds where you can still hold the full model in your head.

---

## Uploading Your Own Data

Constrained upload is supported. We'll validate:
- UTF-8 text only, `.txt` extension
- ≤200KB raw, ≤256 unique characters
- No blocklisted content categories
- ≤200K characters after preprocessing

Uploads get a deterministic ID. Use `upload:<id>` as your `pack_id` in run creation.

---

## Testing

```bash
# Unit tests (autograd, validation, tokenization)
pytest tests/ -v

# UI tests (Vitest)
npm run frontend:test
```

We test event ordering religiously. A training run that doesn't emit `run.completed` or `run.failed` is a bug.

---

## Monitoring

- **Analytics**: PostHog (funnel events only, no PII)
- **Errors**: Sentry free tier across all three services
- **Health**: `/health` endpoints on API and worker

---

## License

MIT. Use it, fork it, break it, fix it.

---


---

## Contributing

Contributions welcome. Start with `docs/architecture.md` and `docs/event-model.md`.

Issues and PRs follow standard templates. Bug reports with reproduction steps get priority.

---

## Why This Exists

I kept watching people "learn transformers" by reading papers and running `model.generate()`. That's like learning combustion by driving a car. 

MicroLLM Lab is the engine with the hood off—stripped down, slowed down, and lit up. You see the spark. You see the fuel. You see why it moves.

Train something small. Watch it learn.
