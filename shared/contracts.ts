export type RunStatus = "queued" | "running" | "completed" | "failed" | "canceled";

export type RunConfig = {
  n_embd: number;
  n_head: number;
  n_layer: number;
  block_size: number;
  num_steps: number;
  learning_rate: number;
  temperature: number;
  seed: number;
  sample_count: number;
  sample_interval: number;
  top_k: number;
  op_graph_token_index: number;
  op_graph_step_interval: number;
};

export type RunSummary = {
  run_id: string;
  status: RunStatus;
  pack_id: string;
  config: RunConfig;
  created_at: string;
  updated_at: string;
  error?: string | null;
};

export type PackDescriptor = {
  pack_id: string;
  title: string;
  description: string;
  document_count: number;
  character_count: number;
};

export type RunEvent = {
  seq: number;
  type: string;
  timestamp: string;
  payload: Record<string, unknown>;
};

export type UploadResponse = {
  upload_id: string;
  document_count: number;
  character_count: number;
  expires_at: string;
};
