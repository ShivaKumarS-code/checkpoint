export interface RunSummary {
  run_id: string;
  name: string;
  mode: "RECORD" | "REPLAY" | "FORK";
  status: "RUNNING" | "COMPLETED" | "FAILED" | "DIVERGED";
  parent_run_id: string | null;
  experiment_id: string | null;
  variant_name: string | null;
  forked_at_step: number | null;
  run_seed: number;
  created_at: number;
  total_tokens: number;
  total_latency_ms: number;
  divergence_step: number | null;
  step_count?: number;
}

export interface RunDetail {
  run: RunSummary;
  lineage: RunSummary[];
  children: RunSummary[];
  divergence_diff: string | null;
}

export interface Step {
  step_id: string;
  run_id: string;
  step_index: number;
  step_type: "LLM_CALL" | "TOOL_CALL" | "CHECKPOINT";
  name: string;
  occurrence: number;
  source: "RECORDED" | "REPLAYED" | "LIVE" | "SEMANTIC";
  tool_mode: string | null;
  error: string | null;
  latency_ms: number;
  frozen_timestamp: string | null;
  rng_seed: number | null;
  input_hash?: string | null;
  output_hash?: string | null;
  post_state_hash?: string | null;
  match_key?: string | null;
}

export interface StepDetail extends Step {
  input: unknown;
  output: unknown;
  post_state: unknown;
}

export interface ExperimentSummary {
  experiment_id: string;
  name: string;
  parent_run_id: string;
  fork_step_index: number;
  created_at: number;
  run_count?: number;
}

export interface Verdict {
  verdict_id: string;
  metric_name: string;
  tier: string;
  passed: boolean;
  reasoning: unknown;
}

export interface ExperimentRun {
  run: RunSummary;
  verdicts: Verdict[];
}

export interface ExperimentCheck {
  type?: string;
  name?: string;
  [key: string]: unknown;
}

export interface ExperimentDetail {
  experiment: ExperimentSummary;
  variants: Array<{ name: string; [key: string]: unknown }>;
  checks: ExperimentCheck[];
  runs: ExperimentRun[];
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { error?: { message?: string } };
      if (payload.error?.message) {
        message = payload.error.message;
      }
    } catch {
      // Keep the HTTP status message.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  runs: (limit = 200) => request<{ runs: RunSummary[] }>(`/api/runs?limit=${limit}`),
  run: (runId: string) =>
    request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`),
  steps: (runId: string) =>
    request<{ run_id: string; steps: Step[] }>(
      `/api/runs/${encodeURIComponent(runId)}/steps`,
    ),
  step: (stepId: string) =>
    request<StepDetail>(`/api/steps/${encodeURIComponent(stepId)}`),
  experiments: (limit = 200) =>
    request<{ experiments: ExperimentSummary[] }>(
      `/api/experiments?limit=${limit}`,
    ),
  experiment: (experimentId: string) =>
    request<ExperimentDetail>(
      `/api/experiments/${encodeURIComponent(experimentId)}`,
    ),
};
