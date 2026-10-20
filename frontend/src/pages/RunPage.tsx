import { useCallback, useEffect, useMemo, useState } from "react";

import {
  api,
  type RunDetail,
  type Step,
  type StepDetail,
} from "../api";
import {
  LiveDot,
  ModeBadge,
  SourceBadge,
  StatusBadge,
  StepTypeBadge,
} from "../components/Badges";
import { EmptyState, ErrorBox, Loading, PageHeader, Panel } from "../components/Panel";
import { Payload } from "../components/Payload";
import { useEvents } from "../events";
import { formatDuration, formatTimestamp, pretty, shortId } from "../format";
import {
  diffPath,
  experimentPath,
  hashHref,
  runPath,
} from "../router";

export function RunPage({ runId }: { runId: string }) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [payloads, setPayloads] = useState<Record<string, StepDetail>>({});
  const [payloadErrors, setPayloadErrors] = useState<Record<string, string>>({});
  const [loadingSteps, setLoadingSteps] = useState<Set<string>>(() => new Set());

  const load = useCallback(async () => {
    try {
      const [runDetail, runSteps] = await Promise.all([
        api.run(runId),
        api.steps(runId),
      ]);
      setDetail(runDetail);
      setSteps(runSteps.steps);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load run");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  const connected = useEvents((event) => {
    if (event.type === "step_added" && event.step?.run_id === runId) {
      const incoming = event.step;
      setSteps((current) => {
        if (current.some((step) => step.step_id === incoming.step_id)) {
          return current;
        }
        return [...current, incoming].sort((a, b) => a.step_index - b.step_index);
      });
    }
    if (
      (event.type === "run_finished" || event.type === "run_started") &&
      event.run?.run_id === runId
    ) {
      void load();
    }
  });

  const toggle = async (step: Step) => {
    const isOpen = expanded.has(step.step_id);
    const next = new Set(expanded);
    if (isOpen) {
      next.delete(step.step_id);
      setExpanded(next);
      return;
    }
    next.add(step.step_id);
    setExpanded(next);
    if (payloads[step.step_id] || loadingSteps.has(step.step_id)) {
      return;
    }
    setLoadingSteps((current) => new Set(current).add(step.step_id));
    try {
      const payload = await api.step(step.step_id);
      setPayloads((current) => ({ ...current, [step.step_id]: payload }));
      setPayloadErrors((current) => {
        const nextErrors = { ...current };
        delete nextErrors[step.step_id];
        return nextErrors;
      });
    } catch (caught) {
      setPayloadErrors((current) => ({
        ...current,
        [step.step_id]: caught instanceof Error ? caught.message : "failed to load step",
      }));
    } finally {
      setLoadingSteps((current) => {
        const nextLoading = new Set(current);
        nextLoading.delete(step.step_id);
        return nextLoading;
      });
    }
  };

  const children = useMemo(() => detail?.children ?? [], [detail]);
  const experimentId =
    detail?.run.experiment_id ??
    children.find((child) => child.experiment_id)?.experiment_id ??
    null;

  if (loading) {
    return <Loading label="loading run" />;
  }
  if (error) {
    return (
      <div>
        <PageHeader title="Run unavailable" />
        <ErrorBox message={error} />
        <a href={hashHref(runPath(runId))} className="mt-4 inline-block text-xs text-accent">
          Back to runs
        </a>
      </div>
    );
  }
  if (!detail) {
    return null;
  }

  const run = detail.run;

  return (
    <div data-testid="run-page">
      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-ink-500">
        <a href={hashHref("/")} className="text-ink-400 transition hover:text-accent">
          Runs
        </a>
        {detail.lineage.map((parent) => (
          <span key={parent.run_id} className="flex items-center gap-2">
            <span className="text-ink-600">/</span>
            <a
              href={hashHref(runPath(parent.run_id))}
              className="text-ink-400 transition hover:text-accent"
            >
              {parent.name}
            </a>
          </span>
        ))}
        <span className="text-ink-600">/</span>
        <span className="text-ink-300">{run.name}</span>
      </div>

      <PageHeader
        eyebrow={`${run.mode} RUN`}
        title={<span data-testid="run-name">{run.name}</span>}
        subtitle={
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">{run.run_id}</span>
            <ModeBadge mode={run.mode} />
            <StatusBadge status={run.status} />
            {run.variant_name ? (
              <span className="rounded border border-warn/25 bg-warn/10 px-1.5 py-0.5 font-mono text-[10px] text-warn">
                {run.variant_name}
              </span>
            ) : null}
            {run.forked_at_step !== null ? (
              <span className="text-[11px] text-ink-400">
                forked at step {run.forked_at_step}
              </span>
            ) : null}
            <span className="text-[11px] text-ink-500">
              {formatTimestamp(run.created_at)}
            </span>
          </div>
        }
        actions={
          <>
            <LiveDot connected={connected} />
            {run.parent_run_id ? (
              <a
                href={hashHref(diffPath(run.parent_run_id, run.run_id))}
                className="rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-xs font-medium text-ink-200 transition hover:border-accent/40 hover:text-accent"
              >
                Compare with parent
              </a>
            ) : null}
            {experimentId ? (
              <a
                href={hashHref(experimentPath(experimentId))}
                className="rounded-md border border-accent/30 bg-accent-soft/60 px-3 py-1.5 text-xs font-medium text-accent-dim transition hover:border-accent/60 hover:text-accent"
              >
                Open experiment
              </a>
            ) : null}
          </>
        }
      />

      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="steps" value={String(steps.length)} />
        <Metric label="llm latency" value={formatDuration(run.total_latency_ms)} />
        <Metric label="tokens" value={String(run.total_tokens)} />
        <Metric
          label="divergence"
          value={run.divergence_step === null ? "none" : `step ${run.divergence_step}`}
        />
      </div>

      {detail.divergence_diff ? (
        <Panel className="mb-6 border-warn/30 bg-warn/5 p-4" testId="divergence-diff">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-warn">
            divergence diff
          </div>
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[11px] leading-5 text-ink-200">
            {detail.divergence_diff}
          </pre>
        </Panel>
      ) : null}

      <Panel className="overflow-hidden" testId="step-timeline">
        <div className="flex items-center justify-between border-b border-ink-700/80 px-5 py-3">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-400">
            Step timeline
          </h2>
          <span className="font-mono text-[10px] text-ink-500">{steps.length} steps</span>
        </div>
        {steps.length === 0 ? (
          <div className="px-5 py-10 text-center text-sm text-ink-400">
            No steps recorded yet. Live updates will appear as the agent runs.
          </div>
        ) : (
          <ol className="divide-y divide-ink-800/80">
            {steps.map((step) => {
              const isOpen = expanded.has(step.step_id);
              const payload = payloads[step.step_id];
              const payloadError = payloadErrors[step.step_id];
              const isLoading = loadingSteps.has(step.step_id);
              return (
                <li key={step.step_id} data-testid={`step-row-${step.step_index}`}>
                  <button
                    type="button"
                    data-testid={`step-toggle-${step.step_index}`}
                    aria-expanded={isOpen}
                    onClick={() => void toggle(step)}
                    className="grid w-full grid-cols-[42px_72px_minmax(0,1fr)_auto] items-center gap-3 px-5 py-4 text-left transition hover:bg-ink-800/40"
                  >
                    <span className="font-mono text-xs text-ink-500">
                      {String(step.step_index).padStart(2, "0")}
                    </span>
                    <StepTypeBadge type={step.step_type} />
                    <span className="flex min-w-0 flex-wrap items-center gap-2">
                      <span className="truncate font-mono text-xs text-ink-100">
                        {step.name}
                      </span>
                      <span className="text-[10px] text-ink-500">×{step.occurrence + 1}</span>
                      <SourceBadge source={step.source} />
                      {step.error ? (
                        <span className="rounded border border-bad/30 bg-bad/10 px-1.5 py-0.5 text-[10px] text-bad">
                          error
                        </span>
                      ) : null}
                    </span>
                    <span className="flex items-center gap-3 font-mono text-[10px] text-ink-500">
                      {formatDuration(step.latency_ms)}
                      <span className={`transition ${isOpen ? "rotate-90" : ""}`}>›</span>
                    </span>
                  </button>
                  {isOpen ? (
                    <div
                      className="space-y-4 border-t border-ink-800 bg-ink-900/40 px-5 py-4"
                      data-testid={`step-payload-${step.step_index}`}
                    >
                      {step.error ? <ErrorBox message={step.error} /> : null}
                      <div className="grid gap-2 font-mono text-[10px] text-ink-500 sm:grid-cols-3">
                        <span>source {step.source}</span>
                        <span>frozen {step.frozen_timestamp ?? "—"}</span>
                        <span>rng seed {step.rng_seed ?? "—"}</span>
                      </div>
                      {isLoading ? <Loading label="loading payload" /> : null}
                      {payloadError ? <ErrorBox message={payloadError} /> : null}
                      {payload ? (
                        <div className="grid gap-4 lg:grid-cols-2">
                          <Payload title="input" value={payload.input} />
                          <Payload title="output" value={payload.output} />
                          {payload.post_state !== null &&
                          payload.post_state !== undefined ? (
                            <div className="lg:col-span-2">
                              <Payload title="post state" value={payload.post_state} />
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        )}
      </Panel>

      {children.length > 0 ? (
        <div className="mt-8">
          <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-ink-400">
            Forks
          </h2>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {children.map((child) => (
              <a
                key={child.run_id}
                href={hashHref(runPath(child.run_id))}
                className="panel block p-4 transition hover:border-accent/40 hover:bg-ink-800/70"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate text-sm text-ink-100">{child.name}</span>
                  <StatusBadge status={child.status} />
                </div>
                <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-400">
                  {child.variant_name ? (
                    <span className="font-mono text-warn">{child.variant_name}</span>
                  ) : null}
                  <span>step {child.forked_at_step ?? "?"}</span>
                  <span className="font-mono text-ink-500">{shortId(child.run_id)}</span>
                </div>
              </a>
            ))}
          </div>
        </div>
      ) : null}

      {detail.lineage.length === 0 && children.length === 0 && !run.parent_run_id ? (
        <EmptyState title="Root run">
          This run has no parent or forks. Fork it with{" "}
          <code className="text-accent">checkpoint fork {shortId(run.run_id)}</code> to
          explore variants.
        </EmptyState>
      ) : null}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel px-4 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-500">
        {label}
      </div>
      <div className="mt-1 truncate font-mono text-sm text-ink-100">{pretty(value)}</div>
    </div>
  );
}
