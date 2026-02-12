import { useEffect, useMemo, useRef, useState } from "react";

import { cancelRun, createRun, eventStreamUrl, listPacks, uploadCorpus } from "./api";
import type { PackDescriptor, RunConfig, RunEvent, RunStatus } from "./types";

const DEFAULT_CONFIG: RunConfig = {
  n_embd: 32,
  n_head: 4,
  n_layer: 1,
  block_size: 16,
  num_steps: 200,
  learning_rate: 0.01,
  temperature: 0.8,
  seed: 42,
  sample_count: 5,
  sample_interval: 50,
  top_k: 5,
  op_graph_token_index: 0,
  op_graph_step_interval: 25,
};

type LossPoint = { step: number; loss: number };
type ForwardFrame = { step: number; token_summaries: Array<Record<string, unknown>> };
type AttentionFrame = {
  step: number;
  token_attention: Array<{ position: number; heads: number[][] }>;
};
type NormFrame = { step: number; values: Record<string, number> };
type OpGraph = {
  step: number;
  nodes: Array<{ id: number; value: number; grad: number }>;
  edges: Array<{ source: number; target: number }>;
};

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function renderLossPoints(points: LossPoint[]): string {
  if (!points.length) {
    return "";
  }
  const width = 520;
  const height = 180;
  const maxStep = points[points.length - 1].step;
  const maxLoss = Math.max(...points.map((p) => p.loss));
  const minLoss = Math.min(...points.map((p) => p.loss));
  const lossSpan = Math.max(maxLoss - minLoss, 0.0001);

  return points
    .map((point) => {
      const x = (point.step / Math.max(maxStep, 1)) * width;
      const y = height - ((point.loss - minLoss) / lossSpan) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function App() {
  const [packs, setPacks] = useState<PackDescriptor[]>([]);
  const [selectedPackId, setSelectedPackId] = useState<string>("");
  const [config, setConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [runId, setRunId] = useState<string>("");
  const [status, setStatus] = useState<RunStatus | "idle">("idle");
  const [error, setError] = useState<string>("");
  const [losses, setLosses] = useState<LossPoint[]>([]);
  const [forwardFrames, setForwardFrames] = useState<ForwardFrame[]>([]);
  const [attentionFrames, setAttentionFrames] = useState<AttentionFrame[]>([]);
  const [gradientFrames, setGradientFrames] = useState<NormFrame[]>([]);
  const [updateFrames, setUpdateFrames] = useState<NormFrame[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [opGraphs, setOpGraphs] = useState<OpGraph[]>([]);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [selectedTokenPosition, setSelectedTokenPosition] = useState(0);
  const [selectedOpGraphIndex, setSelectedOpGraphIndex] = useState(0);
  const [isPlayingGraph, setIsPlayingGraph] = useState(false);
  const [highlightNodeIndex, setHighlightNodeIndex] = useState(0);
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    listPacks()
      .then((data) => {
        setPacks(data);
        if (data.length) {
          setSelectedPackId(data[0].pack_id);
        }
      })
      .catch((exc: Error) => setError(exc.message));
  }, []);

  useEffect(() => {
    if (!runId) {
      return;
    }
    const source = new EventSource(eventStreamUrl(runId));
    eventSourceRef.current = source;

    const onEvent = (event: MessageEvent<string>) => {
      const parsed = JSON.parse(event.data) as RunEvent;
      const payload = parsed.payload;

      if (parsed.type === "step.forward") {
        const frame = payload as unknown as ForwardFrame;
        setForwardFrames((prev) => [...prev, frame]);
      }
      if (parsed.type === "step.attention") {
        const frame = payload as unknown as AttentionFrame;
        setAttentionFrames((prev) => [...prev, frame]);
      }
      if (parsed.type === "step.loss") {
        const step = Number(payload.step ?? 0);
        const loss = Number(payload.loss ?? 0);
        setLosses((prev) => [...prev, { step, loss }]);
      }
      if (parsed.type === "step.backward") {
        const step = Number(payload.step ?? 0);
        const values = (payload.gradient_norms ?? {}) as Record<string, number>;
        setGradientFrames((prev) => [...prev, { step, values }]);
        const graphPayload = payload.op_graph as { nodes?: OpGraph["nodes"]; edges?: OpGraph["edges"] } | null;
        if (graphPayload && Array.isArray(graphPayload.nodes) && Array.isArray(graphPayload.edges)) {
          const graph = { step, nodes: graphPayload.nodes, edges: graphPayload.edges };
          setOpGraphs((prev) => [...prev, graph]);
        }
      }
      if (parsed.type === "step.update") {
        const step = Number(payload.step ?? 0);
        const values = (payload.update_norms ?? {}) as Record<string, number>;
        setUpdateFrames((prev) => [...prev, { step, values }]);
      }
      if (parsed.type === "sample.generated") {
        const generated = (payload.samples ?? []) as string[];
        setSamples(generated);
      }
      if (parsed.type === "run.completed") {
        setStatus("completed");
      }
      if (parsed.type === "run.failed") {
        setStatus("failed");
        setError(String(payload.error ?? "run failed"));
      }
      if (parsed.type === "run.canceled") {
        setStatus("canceled");
      }
    };

    [
      "run.started",
      "step.forward",
      "step.attention",
      "step.loss",
      "step.backward",
      "step.update",
      "sample.generated",
      "run.completed",
      "run.failed",
      "run.canceled",
    ].forEach((eventName) => source.addEventListener(eventName, onEvent));

    source.onerror = () => {
      if (status === "running") {
        setError("Event stream interrupted. Refresh run state.");
      }
    };

    return () => {
      source.close();
    };
  }, [runId, status]);

  useEffect(() => {
    if (!isPlayingGraph) {
      return;
    }
    const graph = opGraphs[selectedOpGraphIndex];
    if (!graph || graph.nodes.length === 0) {
      return;
    }
    const timer = window.setInterval(() => {
      setHighlightNodeIndex((prev) => (prev + 1) % graph.nodes.length);
    }, 350);
    return () => window.clearInterval(timer);
  }, [isPlayingGraph, opGraphs, selectedOpGraphIndex]);

  const selectedForward = forwardFrames[selectedFrameIndex];
  const selectedAttention = attentionFrames[selectedFrameIndex];
  const selectedTokenAttention = selectedAttention?.token_attention.find(
    (entry) => entry.position === selectedTokenPosition,
  );
  const selectedGraph = opGraphs[selectedOpGraphIndex];

  useEffect(() => {
    if (selectedFrameIndex >= forwardFrames.length) {
      setSelectedFrameIndex(Math.max(forwardFrames.length - 1, 0));
    }
  }, [forwardFrames.length, selectedFrameIndex]);

  useEffect(() => {
    if (selectedOpGraphIndex >= opGraphs.length) {
      setSelectedOpGraphIndex(Math.max(opGraphs.length - 1, 0));
      setHighlightNodeIndex(0);
    }
  }, [opGraphs.length, selectedOpGraphIndex]);

  const latestGradients = gradientFrames.length ? gradientFrames[gradientFrames.length - 1] : null;
  const latestUpdates = updateFrames.length ? updateFrames[updateFrames.length - 1] : null;

  const lossPolyline = useMemo(() => renderLossPoints(losses), [losses]);

  async function handleCreateRun(): Promise<void> {
    setError("");
    setLosses([]);
    setForwardFrames([]);
    setAttentionFrames([]);
    setGradientFrames([]);
    setUpdateFrames([]);
    setSamples([]);
    setOpGraphs([]);
    setSelectedFrameIndex(0);
    setSelectedTokenPosition(0);
    setSelectedOpGraphIndex(0);
    setHighlightNodeIndex(0);

    try {
      const run = await createRun(selectedPackId, config);
      setRunId(run.run_id);
      setStatus(run.status);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  async function handleCancelRun(): Promise<void> {
    if (!runId) {
      return;
    }
    try {
      await cancelRun(runId);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  async function handleUpload(): Promise<void> {
    if (!uploadFile) {
      setError("Choose a .txt file first");
      return;
    }
    try {
      const result = await uploadCorpus(uploadFile);
      const uploadPackId = `upload:${result.upload_id}`;
      const uploadPack: PackDescriptor = {
        pack_id: uploadPackId,
        title: "Uploaded Corpus",
        description: `Temporary upload: ${result.document_count} documents`,
        document_count: result.document_count,
        character_count: result.character_count,
      };
      setPacks((prev) => [uploadPack, ...prev.filter((pack) => !pack.pack_id.startsWith("upload:"))]);
      setSelectedPackId(uploadPackId);
      setError("");
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  function updateConfig<K extends keyof RunConfig>(key: K, value: number): void {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="hero-kicker">MicroLLM Lab</p>
        <h1>Train tiny domain GPTs. Inspect every step.</h1>
        <p>
          Real-time forward/backward visualization with constrained server-side training and selected-token computation graphs.
        </p>
      </header>

      <main className="layout-grid">
        <section className="panel control-panel">
          <h2>Run Setup</h2>
          <label>
            Domain Pack
            <select value={selectedPackId} onChange={(event) => setSelectedPackId(event.target.value)}>
              {packs.map((pack) => (
                <option key={pack.pack_id} value={pack.pack_id}>
                  {pack.title} ({pack.document_count} docs)
                </option>
              ))}
            </select>
          </label>

          <div className="config-grid">
            <label>
              Steps
              <input
                type="number"
                min={1}
                max={2000}
                value={config.num_steps}
                onChange={(event) => updateConfig("num_steps", clampNumber(Number(event.target.value), 1, 2000))}
              />
            </label>
            <label>
              Block Size
              <input
                type="number"
                min={4}
                max={64}
                value={config.block_size}
                onChange={(event) => updateConfig("block_size", clampNumber(Number(event.target.value), 4, 64))}
              />
            </label>
            <label>
              n_embd
              <input
                type="number"
                min={8}
                max={64}
                step={8}
                value={config.n_embd}
                onChange={(event) => updateConfig("n_embd", clampNumber(Number(event.target.value), 8, 64))}
              />
            </label>
            <label>
              n_head
              <input
                type="number"
                min={1}
                max={8}
                value={config.n_head}
                onChange={(event) => updateConfig("n_head", clampNumber(Number(event.target.value), 1, 8))}
              />
            </label>
            <label>
              n_layer
              <input
                type="number"
                min={1}
                max={2}
                value={config.n_layer}
                onChange={(event) => updateConfig("n_layer", clampNumber(Number(event.target.value), 1, 2))}
              />
            </label>
            <label>
              Learning Rate
              <input
                type="number"
                step={0.001}
                min={0.001}
                max={0.1}
                value={config.learning_rate}
                onChange={(event) => updateConfig("learning_rate", clampNumber(Number(event.target.value), 0.001, 0.1))}
              />
            </label>
            <label>
              Temperature
              <input
                type="number"
                step={0.1}
                min={0.1}
                max={1.5}
                value={config.temperature}
                onChange={(event) => updateConfig("temperature", clampNumber(Number(event.target.value), 0.1, 1.5))}
              />
            </label>
            <label>
              Op Graph Token Index
              <input
                type="number"
                min={0}
                max={63}
                value={config.op_graph_token_index}
                onChange={(event) => updateConfig("op_graph_token_index", clampNumber(Number(event.target.value), 0, 63))}
              />
            </label>
          </div>

          <div className="action-row">
            <button onClick={handleCreateRun} disabled={!selectedPackId || status === "running"}>
              Start Run
            </button>
            <button onClick={handleCancelRun} disabled={!runId || status !== "running"} className="secondary">
              Cancel
            </button>
          </div>

          <div className="upload-box">
            <h3>Constrained Upload (.txt, max 200KB)</h3>
            <input type="file" accept=".txt,text/plain" onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)} />
            <button onClick={handleUpload} className="secondary">Use Upload as Pack</button>
          </div>

          <p className="status-line">
            Status: <strong>{status}</strong>
            {runId ? ` | run_id: ${runId}` : ""}
          </p>
          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="panel chart-panel">
          <h2>Loss Curve</h2>
          <svg viewBox="0 0 520 180" className="loss-chart" aria-label="Loss curve">
            <polyline points={lossPolyline} fill="none" stroke="currentColor" strokeWidth="2" />
          </svg>
          <p className="muted">Forward frames: {forwardFrames.length} | Attention frames: {attentionFrames.length}</p>
        </section>

        <section className="panel token-panel">
          <h2>Token Timeline</h2>
          <label>
            Step Index
            <input
              type="range"
              min={0}
              max={Math.max(forwardFrames.length - 1, 0)}
              value={selectedFrameIndex}
              onChange={(event) => setSelectedFrameIndex(Number(event.target.value))}
            />
          </label>
          <label>
            Token Position
            <input
              type="number"
              min={0}
              max={63}
              value={selectedTokenPosition}
              onChange={(event) => setSelectedTokenPosition(clampNumber(Number(event.target.value), 0, 63))}
            />
          </label>
          <div className="token-list">
            {(selectedForward?.token_summaries ?? []).map((item, index) => {
              const tokenItem = item as { position: number; input_token: string; target_token: string; top_k: Array<{ token: string; prob: number }> };
              return (
                <article key={`${tokenItem.position}-${index}`} className="token-card">
                  <p>
                    pos {tokenItem.position}: <code>{tokenItem.input_token}</code> -&gt; <code>{tokenItem.target_token}</code>
                  </p>
                  <p>
                    top-k: {tokenItem.top_k.map((entry) => `${entry.token}:${entry.prob.toFixed(3)}`).join(" | ")}
                  </p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="panel attention-panel">
          <h2>Attention Heatmap</h2>
          <p className="muted">Selected token position: {selectedTokenPosition}</p>
          <div className="heatmap-grid">
            {(selectedTokenAttention?.heads ?? []).map((head, headIndex) => (
              <div key={headIndex} className="head-row">
                <span className="head-label">Head {headIndex}</span>
                <div className="head-values">
                  {head.map((value, idx) => (
                    <span
                      key={`${headIndex}-${idx}`}
                      className="cell"
                      style={{ opacity: clampNumber(value, 0.08, 1) }}
                      title={`t${idx}: ${value}`}
                    >
                      {value.toFixed(2)}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel gradient-panel">
          <h2>Gradient and Update Norms</h2>
          <div className="norm-columns">
            <div>
              <h3>Gradients (step {latestGradients?.step ?? 0})</h3>
              <ul>
                {Object.entries(latestGradients?.values ?? {}).map(([key, value]) => (
                  <li key={key}>{key}: {value.toFixed(6)}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3>Updates (step {latestUpdates?.step ?? 0})</h3>
              <ul>
                {Object.entries(latestUpdates?.values ?? {}).map(([key, value]) => (
                  <li key={key}>{key}: {value.toFixed(6)}</li>
                ))}
              </ul>
            </div>
          </div>
          <h3>Samples</h3>
          <ul className="samples">
            {samples.map((sample, index) => (
              <li key={`${sample}-${index}`}>{sample || "<empty>"}</li>
            ))}
          </ul>
        </section>

        <section className="panel op-graph-panel">
          <h2>Operation Graph (Selected Token)</h2>
          <label>
            Graph Snapshot
            <input
              type="range"
              min={0}
              max={Math.max(opGraphs.length - 1, 0)}
              value={selectedOpGraphIndex}
              onChange={(event) => {
                setSelectedOpGraphIndex(Number(event.target.value));
                setHighlightNodeIndex(0);
              }}
            />
          </label>
          <div className="action-row">
            <button className="secondary" onClick={() => setIsPlayingGraph((prev) => !prev)}>
              {isPlayingGraph ? "Pause" : "Play"}
            </button>
            <button
              className="secondary"
              onClick={() => {
                if (!selectedGraph?.nodes.length) {
                  return;
                }
                setHighlightNodeIndex((prev) => (prev + 1) % selectedGraph.nodes.length);
              }}
            >
              Step
            </button>
          </div>
          <p className="muted">
            Snapshot step: {selectedGraph?.step ?? 0} | nodes: {selectedGraph?.nodes.length ?? 0} | edges: {selectedGraph?.edges.length ?? 0}
          </p>
          <div className="graph-list">
            {(selectedGraph?.nodes ?? []).map((node, index) => (
              <div
                key={node.id}
                className={index === highlightNodeIndex ? "graph-node highlighted" : "graph-node"}
              >
                <span>id:{node.id}</span>
                <span>val:{node.value.toFixed(6)}</span>
                <span>grad:{node.grad.toFixed(6)}</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
