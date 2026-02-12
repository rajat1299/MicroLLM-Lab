import type { PackDescriptor, RunConfig, RunSummary, UploadResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function listPacks(): Promise<PackDescriptor[]> {
  const response = await fetch(`${API_BASE}/api/v1/packs`);
  return parseJson<PackDescriptor[]>(response);
}

export async function createRun(packId: string, config: RunConfig): Promise<RunSummary> {
  const response = await fetch(`${API_BASE}/api/v1/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pack_id: packId, config }),
  });
  return parseJson<RunSummary>(response);
}

export async function fetchRun(runId: string): Promise<RunSummary> {
  const response = await fetch(`${API_BASE}/api/v1/runs/${runId}`);
  return parseJson<RunSummary>(response);
}

export async function cancelRun(runId: string): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/api/v1/runs/${runId}/cancel`, { method: "POST" });
  return parseJson<{ status: string }>(response);
}

export async function uploadCorpus(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/v1/uploads`, {
    method: "POST",
    body: form,
  });
  return parseJson<UploadResponse>(response);
}

export function eventStreamUrl(runId: string): string {
  return `${API_BASE}/api/v1/runs/${runId}/events`;
}
