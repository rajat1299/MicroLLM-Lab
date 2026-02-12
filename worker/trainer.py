from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from shared.types import RunConfig


class Value:
    __slots__ = ("id", "data", "grad", "_children", "_local_grads")
    _next_id = 1

    def __init__(self, data: float, children=(), local_grads=()):
        self.id = Value._next_id
        Value._next_id += 1
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(float(other))
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(float(other))
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self):
        exp_val = math.exp(self.data)
        return Value(exp_val, (self,), (exp_val,))

    def relu(self):
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0.0 else 0.0,))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node: Value):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                child.grad += local_grad * node.grad


@dataclass
class TrainResult:
    status: str
    final_loss: float
    steps_completed: int
    vocab_size: int


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: list[Value]) -> list[Value]:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def _matrix(nout: int, nin: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0.0, std)) for _ in range(nin)] for _ in range(nout)]


def _build_state(vocab_size: int, config: RunConfig) -> dict[str, list[list[Value]]]:
    state: dict[str, list[list[Value]]] = {
        "wte": _matrix(vocab_size, config.n_embd),
        "wpe": _matrix(config.block_size, config.n_embd),
        "lm_head": _matrix(vocab_size, config.n_embd),
    }
    for i in range(config.n_layer):
        state[f"layer{i}.attn_wq"] = _matrix(config.n_embd, config.n_embd)
        state[f"layer{i}.attn_wk"] = _matrix(config.n_embd, config.n_embd)
        state[f"layer{i}.attn_wv"] = _matrix(config.n_embd, config.n_embd)
        state[f"layer{i}.attn_wo"] = _matrix(config.n_embd, config.n_embd)
        state[f"layer{i}.mlp_fc1"] = _matrix(4 * config.n_embd, config.n_embd)
        state[f"layer{i}.mlp_fc2"] = _matrix(config.n_embd, 4 * config.n_embd)
    return state


def _flatten_params(
    state: dict[str, list[list[Value]]],
) -> tuple[list[Value], dict[str, list[Value]], dict[int, str]]:
    all_params: list[Value] = []
    groups = {"embeddings": [], "attention": [], "mlp": [], "lm_head": []}
    param_group_by_id: dict[int, str] = {}
    for name, mat in state.items():
        group_name = "lm_head"
        if name in {"wte", "wpe"}:
            group_name = "embeddings"
        elif ".attn_" in name:
            group_name = "attention"
        elif ".mlp_" in name:
            group_name = "mlp"

        for row in mat:
            for param in row:
                all_params.append(param)
                groups[group_name].append(param)
                param_group_by_id[param.id] = group_name
    return all_params, groups, param_group_by_id


def _serialize_graph(root: Value, max_nodes: int = 160) -> dict[str, list[dict[str, float | int]]]:
    topo: list[Value] = []
    visited = set()

    def build(node: Value):
        if node not in visited:
            visited.add(node)
            for child in node._children:
                build(child)
            topo.append(node)

    build(root)
    trimmed = topo[-max_nodes:]
    id_set = {node.id for node in trimmed}
    nodes = [
        {
            "id": node.id,
            "value": round(node.data, 6),
            "grad": round(node.grad, 6),
        }
        for node in trimmed
    ]
    edges: list[dict[str, int]] = []
    for node in trimmed:
        for child in node._children:
            if child.id in id_set:
                edges.append({"source": child.id, "target": node.id})
    return {"nodes": nodes, "edges": edges}


def _token_str(token_id: int, uchars: list[str], bos_id: int) -> str:
    return "<BOS>" if token_id == bos_id else uchars[token_id]


def _top_k_probs(
    probs: list[Value],
    uchars: list[str],
    bos_id: int,
    k: int,
) -> list[dict[str, float | str]]:
    ranked = sorted(enumerate(probs), key=lambda item: item[1].data, reverse=True)[:k]
    return [
        {
            "token_id": token_id,
            "token": _token_str(token_id, uchars, bos_id),
            "prob": round(prob.data, 6),
        }
        for token_id, prob in ranked
    ]


def _norm_for_values(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _sample_sequences(
    state: dict[str, list[list[Value]]],
    config: RunConfig,
    uchars: list[str],
    bos_id: int,
) -> list[str]:
    head_dim = config.n_embd // config.n_head

    def gpt(
        token_id: int,
        pos_id: int,
        keys: list[list[list[Value]]],
        values: list[list[list[Value]]],
    ) -> tuple[list[Value], list[list[float]]]:
        tok_emb = state["wte"][token_id]
        pos_emb = state["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        attention_per_head: list[list[float]] = []
        for li in range(config.n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state[f"layer{li}.attn_wq"])
            k = linear(x, state[f"layer{li}.attn_wk"])
            v = linear(x, state[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn: list[Value] = []
            for h in range(config.n_head):
                start = h * head_dim
                q_h = q[start : start + head_dim]
                k_h = [ki[start : start + head_dim] for ki in keys[li]]
                v_h = [vi[start : start + head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / math.sqrt(head_dim)
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                attention_per_head.append([round(w.data, 6) for w in attn_weights])
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, state[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state["lm_head"])
        return logits, attention_per_head

    samples: list[str] = []
    for _ in range(config.sample_count):
        keys = [[] for _ in range(config.n_layer)]
        values = [[] for _ in range(config.n_layer)]
        token_id = bos_id
        chars: list[str] = []
        for pos_id in range(config.block_size):
            logits, _ = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / config.temperature for l in logits])
            token_id = random.choices(
                range(len(uchars) + 1), weights=[p.data for p in probs], k=1
            )[0]
            if token_id == bos_id:
                break
            chars.append(uchars[token_id])
        samples.append("".join(chars))
    return samples


def train_tiny_gpt(
    docs: list[str],
    config: RunConfig,
    emit_event: Callable[[str, dict], None],
    is_cancel_requested: Callable[[], bool],
) -> TrainResult:
    random.seed(config.seed)
    docs = docs.copy()
    random.shuffle(docs)

    uchars = sorted(set("".join(docs)))
    if not uchars:
        raise ValueError("No characters available in corpus")

    bos_id = len(uchars)
    vocab_size = len(uchars) + 1
    head_dim = config.n_embd // config.n_head
    state = _build_state(vocab_size, config)
    params, param_groups, param_group_by_id = _flatten_params(state)

    learning_rate, beta1, beta2, eps_adam = config.learning_rate, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    def gpt(
        token_id: int,
        pos_id: int,
        keys: list[list[list[Value]]],
        values: list[list[list[Value]]],
    ) -> tuple[list[Value], list[list[float]]]:
        tok_emb = state["wte"][token_id]
        pos_emb = state["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        attention_per_head: list[list[float]] = []
        for li in range(config.n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state[f"layer{li}.attn_wq"])
            k = linear(x, state[f"layer{li}.attn_wk"])
            v_layer = linear(x, state[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v_layer)

            x_attn: list[Value] = []
            for h in range(config.n_head):
                start = h * head_dim
                q_h = q[start : start + head_dim]
                k_h = [ki[start : start + head_dim] for ki in keys[li]]
                v_h = [vi[start : start + head_dim] for vi in values[li]]

                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / math.sqrt(head_dim)
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                attention_per_head.append([round(w.data, 6) for w in attn_weights])
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, state[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, state[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state["lm_head"])
        return logits, attention_per_head

    emit_event(
        "run.started",
        {
            "vocab_size": vocab_size,
            "doc_count": len(docs),
            "num_params": len(params),
            "config": config.model_dump(mode="json"),
        },
    )

    last_loss = 0.0
    for step in range(config.num_steps):
        if is_cancel_requested():
            emit_event("run.canceled", {"step": step + 1})
            return TrainResult(
                status="canceled",
                final_loss=last_loss,
                steps_completed=step,
                vocab_size=vocab_size,
            )

        doc = docs[step % len(docs)]
        tokens = [bos_id] + [uchars.index(ch) for ch in doc] + [bos_id]
        n = min(config.block_size, len(tokens) - 1)
        keys = [[] for _ in range(config.n_layer)]
        values = [[] for _ in range(config.n_layer)]

        losses: list[Value] = []
        token_summaries: list[dict] = []
        attention_by_token: list[dict] = []
        selected_token_loss: Value | None = None

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits, attention_heads = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

            token_summaries.append(
                {
                    "position": pos_id,
                    "input_token": _token_str(token_id, uchars, bos_id),
                    "target_token": _token_str(target_id, uchars, bos_id),
                    "top_k": _top_k_probs(probs, uchars, bos_id, config.top_k),
                }
            )
            attention_by_token.append(
                {
                    "position": pos_id,
                    "heads": attention_heads,
                }
            )
            if pos_id == config.op_graph_token_index:
                selected_token_loss = loss_t

        loss = (1.0 / n) * sum(losses)
        last_loss = loss.data

        emit_event(
            "step.forward",
            {
                "step": step + 1,
                "token_summaries": token_summaries,
            },
        )
        emit_event(
            "step.attention",
            {
                "step": step + 1,
                "token_attention": attention_by_token,
            },
        )
        emit_event(
            "step.loss",
            {
                "step": step + 1,
                "loss": round(loss.data, 6),
            },
        )

        loss.backward()

        grad_norms = {
            group_name: round(
                _norm_for_values([param.grad for param in group_params]),
                6,
            )
            for group_name, group_params in param_groups.items()
        }

        op_graph: dict | None = None
        if (
            selected_token_loss is not None
            and step % config.op_graph_step_interval == 0
        ):
            op_graph = _serialize_graph(selected_token_loss)

        emit_event(
            "step.backward",
            {
                "step": step + 1,
                "gradient_norms": grad_norms,
                "op_graph": op_graph,
            },
        )

        lr_t = learning_rate * (1.0 - (step / config.num_steps))
        delta_by_group = {"embeddings": [], "attention": [], "mlp": [], "lm_head": []}

        for i, param in enumerate(params):
            m[i] = beta1 * m[i] + (1.0 - beta1) * param.grad
            v[i] = beta2 * v[i] + (1.0 - beta2) * (param.grad**2)
            m_hat = m[i] / (1.0 - beta1 ** (step + 1))
            v_hat = v[i] / (1.0 - beta2 ** (step + 1))
            delta = lr_t * m_hat / (math.sqrt(v_hat) + eps_adam)
            param.data -= delta

            group_name = param_group_by_id[param.id]
            delta_by_group[group_name].append(delta)
            param.grad = 0.0

        update_norms = {
            group_name: round(_norm_for_values(deltas), 6)
            for group_name, deltas in delta_by_group.items()
        }
        emit_event(
            "step.update",
            {
                "step": step + 1,
                "learning_rate": round(lr_t, 8),
                "update_norms": update_norms,
            },
        )

        if (step + 1) % config.sample_interval == 0 or step + 1 == config.num_steps:
            samples = _sample_sequences(state, config, uchars, bos_id)
            emit_event(
                "sample.generated",
                {
                    "step": step + 1,
                    "samples": samples,
                },
            )

    return TrainResult(
        status="completed",
        final_loss=last_loss,
        steps_completed=config.num_steps,
        vocab_size=vocab_size,
    )
