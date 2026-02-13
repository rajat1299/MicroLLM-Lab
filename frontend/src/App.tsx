/**
 * MicroLLM Lab — Early Mac Era UI
 * 
 * Design Philosophy (Apple HIG, 1987):
 * - "The user, not the computer, initiates and controls all actions"
 * - "People learn best when they're actively engaged"
 * - "Graphics are not merely cosmetic... they contribute to understanding"
 */

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
  if (!points.length) return "";
  const width = 520;
  const height = 160;
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

/** Classic Mac window title bar with close box */
function PanelTitleBar({ title }: { title: string }) {
  return (
    <div className="panel-titlebar">
      <div className="panel-closebox" aria-hidden="true" />
      <span className="panel-title">{title}</span>
    </div>
  );
}

/** Returns intensity level for heatmap cell dithering */
function getIntensityLevel(value: number): "low" | "medium" | "high" | "max" {
  if (value < 0.2) return "low";
  if (value < 0.5) return "medium";
  if (value < 0.8) return "high";
  return "max";
}

/** Friendly status messages - "expressed in the user's vocabulary" */
function getStatusMessage(status: RunStatus | "idle", step?: number, maxSteps?: number): string {
  switch (status) {
    case "idle":
      return "Ready to train";
    case "running":
      return step !== undefined ? `Training… step ${step} of ${maxSteps}` : "Training…";
    case "completed":
      return "Training complete ✓";
    case "failed":
      return "Training failed";
    case "canceled":
      return "Training canceled";
    default:
      return String(status);
  }
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
  const [currentStep, setCurrentStep] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);

  // Fetch available packs on mount
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

  // SSE event stream for training updates
  useEffect(() => {
    if (!runId) return;

    const source = new EventSource(eventStreamUrl(runId));
    eventSourceRef.current = source;

    const onEvent = (event: MessageEvent<string>) => {
      const parsed = JSON.parse(event.data) as RunEvent;
      const payload = parsed.payload;

      if (parsed.type === "step.forward") {
        const frame = payload as unknown as ForwardFrame;
        setForwardFrames((prev) => [...prev, frame]);
        setCurrentStep(frame.step);
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
        setError(String(payload.error ?? "Run failed"));
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
        setError("Connection lost. Training may still be running.");
      }
    };

    return () => source.close();
  }, [runId, status]);

  // Animated op-graph playback
  useEffect(() => {
    if (!isPlayingGraph) return;
    const graph = opGraphs[selectedOpGraphIndex];
    if (!graph || graph.nodes.length === 0) return;

    const timer = window.setInterval(() => {
      setHighlightNodeIndex((prev) => (prev + 1) % graph.nodes.length);
    }, 350);

    return () => window.clearInterval(timer);
  }, [isPlayingGraph, opGraphs, selectedOpGraphIndex]);

  const selectedForward = forwardFrames[selectedFrameIndex];
  const selectedAttention = attentionFrames[selectedFrameIndex];
  const selectedTokenAttention = selectedAttention?.token_attention.find(
    (entry) => entry.position === selectedTokenPosition
  );
  const selectedGraph = opGraphs[selectedOpGraphIndex];

  // Sync frame indices
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
  const progressPercent = (currentStep / config.num_steps) * 100;

  /** Start training - user initiates the action */
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
    setCurrentStep(0);

    try {
      const run = await createRun(selectedPackId, config);
      setRunId(run.run_id);
      setStatus(run.status);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  /** Cancel training - user remains in control */
  async function handleCancelRun(): Promise<void> {
    if (!runId) return;
    try {
      await cancelRun(runId);
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  /** Upload custom corpus */
  async function handleUpload(): Promise<void> {
    if (!uploadFile) {
      setError("Please choose a .txt file first");
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

  const isRunning = status === "running";

  return (
    <div className="app-shell">
      {/* ================================================
          HERO - About Dialog Style
          ================================================ */}
      <header className="hero">
        <div className="hero-titlebar">
          <div className="hero-closebox" aria-hidden="true" />
          <span className="hero-title">About MicroLLM Lab</span>
        </div>
        <div className="hero-content">
          <div className="hero-icon" aria-hidden="true" />
          <h1>MicroLLM Lab</h1>
          <p>Train tiny transformers. Watch them learn.</p>
          <p>Real-time visualization of forward pass, attention, and backpropagation.</p>
        </div>
      </header>

      <main className="layout-grid">
        {/* ================================================
            CONTROL PANEL - Run Setup
            ================================================ */}
        <section className="panel control-panel" aria-labelledby="control-title">
          <PanelTitleBar title="Run Setup" />
          <div className="panel-content">
            <label>
              <span>Domain Pack</span>
              <select
                value={selectedPackId}
                onChange={(e) => setSelectedPackId(e.target.value)}
                aria-describedby="pack-description"
              >
                {packs.map((pack) => (
                  <option key={pack.pack_id} value={pack.pack_id}>
                    {pack.title} ({pack.document_count} docs)
                  </option>
                ))}
              </select>
            </label>
            <p id="pack-description" className="muted">
              Choose training data for the model.
            </p>

            <h3>Model Configuration</h3>
            <div className="config-grid">
              <label>
                <span>Steps</span>
                <input
                  type="number"
                  min={1}
                  max={2000}
                  value={config.num_steps}
                  onChange={(e) => updateConfig("num_steps", clampNumber(Number(e.target.value), 1, 2000))}
                  aria-describedby="steps-hint"
                />
              </label>
              <label>
                <span>Block Size</span>
                <input
                  type="number"
                  min={4}
                  max={64}
                  value={config.block_size}
                  onChange={(e) => updateConfig("block_size", clampNumber(Number(e.target.value), 4, 64))}
                />
              </label>
              <label>
                <span>Embedding Dim</span>
                <input
                  type="number"
                  min={8}
                  max={64}
                  step={8}
                  value={config.n_embd}
                  onChange={(e) => updateConfig("n_embd", clampNumber(Number(e.target.value), 8, 64))}
                />
              </label>
              <label>
                <span>Heads</span>
                <input
                  type="number"
                  min={1}
                  max={8}
                  value={config.n_head}
                  onChange={(e) => updateConfig("n_head", clampNumber(Number(e.target.value), 1, 8))}
                />
              </label>
              <label>
                <span>Layers</span>
                <input
                  type="number"
                  min={1}
                  max={2}
                  value={config.n_layer}
                  onChange={(e) => updateConfig("n_layer", clampNumber(Number(e.target.value), 1, 2))}
                />
              </label>
              <label>
                <span>Learning Rate</span>
                <input
                  type="number"
                  step={0.001}
                  min={0.001}
                  max={0.1}
                  value={config.learning_rate}
                  onChange={(e) => updateConfig("learning_rate", clampNumber(Number(e.target.value), 0.001, 0.1))}
                />
              </label>
              <label>
                <span>Temperature</span>
                <input
                  type="number"
                  step={0.1}
                  min={0.1}
                  max={1.5}
                  value={config.temperature}
                  onChange={(e) => updateConfig("temperature", clampNumber(Number(e.target.value), 0.1, 1.5))}
                />
              </label>
              <label>
                <span>Graph Token</span>
                <input
                  type="number"
                  min={0}
                  max={63}
                  value={config.op_graph_token_index}
                  onChange={(e) => updateConfig("op_graph_token_index", clampNumber(Number(e.target.value), 0, 63))}
                />
              </label>
            </div>
            <p id="steps-hint" className="muted">
              More steps = longer training, finer learning.
            </p>

            {/* Action buttons - user in control */}
            <div className="action-row">
              <button
                className="primary"
                onClick={handleCreateRun}
                disabled={!selectedPackId || isRunning}
                aria-busy={isRunning}
              >
                {isRunning ? "Training…" : "Start Training"}
              </button>
              <button
                className="secondary"
                onClick={handleCancelRun}
                disabled={!runId || !isRunning}
              >
                Stop
              </button>
            </div>

            {/* Progress indicator - immediate feedback */}
            {isRunning && (
              <div className="progress-bar" role="progressbar" aria-valuenow={currentStep} aria-valuemax={config.num_steps}>
                <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
              </div>
            )}

            {/* Upload section */}
            <div className="upload-box">
              <h3>Custom Corpus</h3>
              <p className="muted">Upload .txt file (max 200KB)</p>
              <input
                type="file"
                accept=".txt,text/plain"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                aria-label="Choose text file to upload"
              />
              <button className="secondary" onClick={handleUpload} style={{ marginTop: "4px" }}>
                Use Upload
              </button>
            </div>

            {/* Status - brief, direct communication */}
            <p className={`status-line ${isRunning ? "status-running" : ""}`} aria-live="polite">
              {getStatusMessage(status, currentStep, config.num_steps)}
              {runId ? ` • ID: ${runId.slice(0, 8)}…` : ""}
            </p>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
          </div>
        </section>

        {/* ================================================
            LOSS CHART
            ================================================ */}
        <section className="panel chart-panel" aria-labelledby="loss-title">
          <PanelTitleBar title="Loss Curve" />
          <div className="panel-content">
            <svg
              viewBox="0 0 520 180"
              className="loss-chart"
              aria-label={`Loss curve chart showing ${losses.length} data points`}
              role="img"
            >
              {/* Grid lines */}
              <line className="chart-grid" x1="0" y1="45" x2="520" y2="45" />
              <line className="chart-grid" x1="0" y1="90" x2="520" y2="90" />
              <line className="chart-grid" x1="0" y1="135" x2="520" y2="135" />
              {/* Axis */}
              <line className="chart-axis" x1="0" y1="160" x2="520" y2="160" />
              {/* Loss curve */}
              <polyline points={lossPolyline} />
            </svg>
            <p className="muted">
              Steps: {losses.length} • Frames: {forwardFrames.length}
              {losses.length > 0 && ` • Latest loss: ${losses[losses.length - 1].loss.toFixed(4)}`}
            </p>
          </div>
        </section>

        {/* ================================================
            TOKEN TIMELINE
            ================================================ */}
        <section className="panel token-panel" aria-labelledby="token-title">
          <PanelTitleBar title="Token Timeline" />
          <div className="panel-content">
            <label>
              <span>Step Index ({selectedFrameIndex})</span>
              <input
                type="range"
                min={0}
                max={Math.max(forwardFrames.length - 1, 0)}
                value={selectedFrameIndex}
                onChange={(e) => setSelectedFrameIndex(Number(e.target.value))}
                aria-valuetext={`Step ${selectedFrameIndex}`}
              />
            </label>
            <label>
              <span>Token Position</span>
              <input
                type="number"
                min={0}
                max={63}
                value={selectedTokenPosition}
                onChange={(e) => setSelectedTokenPosition(clampNumber(Number(e.target.value), 0, 63))}
              />
            </label>
            <div className="token-list" role="list" aria-label="Token predictions">
              {(selectedForward?.token_summaries ?? []).map((item, index) => {
                const tok = item as {
                  position: number;
                  input_token: string;
                  target_token: string;
                  top_k: Array<{ token: string; prob: number }>;
                };
                return (
                  <article key={`${tok.position}-${index}`} className="token-card" role="listitem">
                    <p>
                      <strong>pos {tok.position}:</strong> <code>{tok.input_token}</code> → <code>{tok.target_token}</code>
                    </p>
                    <p>
                      top-k: {tok.top_k.map((e) => `${e.token}:${e.prob.toFixed(3)}`).join(" | ")}
                    </p>
                  </article>
                );
              })}
              {!selectedForward && <p className="muted" style={{ padding: "8px" }}>Start training to see tokens…</p>}
            </div>
          </div>
        </section>

        {/* ================================================
            ATTENTION HEATMAP
            ================================================ */}
        <section className="panel attention-panel" aria-labelledby="attention-title">
          <PanelTitleBar title="Attention Weights" />
          <div className="panel-content">
            <p className="muted">Token position: {selectedTokenPosition}</p>
            <div className="heatmap-grid" role="table" aria-label="Attention heatmap">
              {(selectedTokenAttention?.heads ?? []).map((head, headIndex) => (
                <div key={headIndex} className="head-row" role="row">
                  <span className="head-label" role="rowheader">Head {headIndex}</span>
                  <div className="head-values" role="rowgroup">
                    {head.map((value, idx) => (
                      <span
                        key={`${headIndex}-${idx}`}
                        className="cell"
                        data-intensity={getIntensityLevel(value)}
                        role="cell"
                        title={`Position ${idx}: ${value.toFixed(4)}`}
                        aria-label={`Attention ${value.toFixed(2)} at position ${idx}`}
                      >
                        {value.toFixed(2)}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
              {!selectedTokenAttention && <p className="muted">Attention data appears during training…</p>}
            </div>
          </div>
        </section>

        {/* ================================================
            GRADIENT & UPDATE NORMS
            ================================================ */}
        <section className="panel gradient-panel" aria-labelledby="gradient-title">
          <PanelTitleBar title="Gradients & Updates" />
          <div className="panel-content">
            <div className="norm-columns">
              <div>
                <h3>Gradient Norms (step {latestGradients?.step ?? 0})</h3>
                <ul aria-label="Gradient norms by layer">
                  {Object.entries(latestGradients?.values ?? {}).map(([key, value]) => (
                    <li key={key}>
                      {key}: {value.toFixed(6)}
                    </li>
                  ))}
                  {!latestGradients && <li className="muted">Waiting for backward pass…</li>}
                </ul>
              </div>
              <div>
                <h3>Update Norms (step {latestUpdates?.step ?? 0})</h3>
                <ul aria-label="Update norms by layer">
                  {Object.entries(latestUpdates?.values ?? {}).map(([key, value]) => (
                    <li key={key}>
                      {key}: {value.toFixed(6)}
                    </li>
                  ))}
                  {!latestUpdates && <li className="muted">Waiting for updates…</li>}
                </ul>
              </div>
            </div>
            <h3>Generated Samples</h3>
            <ul className="samples" aria-label="Generated text samples">
              {samples.map((sample, index) => (
                <li key={`${sample}-${index}`}>{sample || "(empty)"}</li>
              ))}
              {samples.length === 0 && <li className="muted">Samples appear periodically during training…</li>}
            </ul>
          </div>
        </section>

        {/* ================================================
            OPERATION GRAPH
            ================================================ */}
        <section className="panel op-graph-panel" aria-labelledby="opgraph-title">
          <PanelTitleBar title="Operation Graph" />
          <div className="panel-content">
            <label>
              <span>Graph Snapshot ({selectedOpGraphIndex})</span>
              <input
                type="range"
                min={0}
                max={Math.max(opGraphs.length - 1, 0)}
                value={selectedOpGraphIndex}
                onChange={(e) => {
                  setSelectedOpGraphIndex(Number(e.target.value));
                  setHighlightNodeIndex(0);
                }}
                aria-valuetext={`Snapshot ${selectedOpGraphIndex}`}
              />
            </label>
            <div className="action-row">
              <button
                className="secondary"
                onClick={() => setIsPlayingGraph((prev) => !prev)}
                aria-pressed={isPlayingGraph}
                aria-label={isPlayingGraph ? "Pause animation" : "Play animation"}
              >
                {isPlayingGraph ? "Pause" : "Play"}
              </button>
              <button
                className="secondary"
                onClick={() => {
                  if (!selectedGraph?.nodes.length) return;
                  setHighlightNodeIndex((prev) => (prev + 1) % selectedGraph.nodes.length);
                }}
                disabled={!selectedGraph?.nodes.length}
                aria-label="Step to next node"
              >
                Step
              </button>
            </div>
            <p className="muted">
              Step: {selectedGraph?.step ?? 0} • Nodes: {selectedGraph?.nodes.length ?? 0} • Edges: {selectedGraph?.edges.length ?? 0}
            </p>
            <div className="graph-list" role="list" aria-label="Computation graph nodes">
              {(selectedGraph?.nodes ?? []).map((node, index) => (
                <div
                  key={node.id}
                  className={index === highlightNodeIndex ? "graph-node highlighted" : "graph-node"}
                  role="listitem"
                  aria-current={index === highlightNodeIndex ? "true" : undefined}
                >
                  <span>id:{node.id}</span>
                  <span>val:{node.value.toFixed(5)}</span>
                  <span>grad:{node.grad.toFixed(5)}</span>
                </div>
              ))}
              {!selectedGraph && <p className="muted" style={{ padding: "8px" }}>Graph data appears during training…</p>}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
