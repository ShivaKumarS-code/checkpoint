import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type ExperimentSummary, type RunSummary } from "../api";
import { LiveDot, ModeBadge, StatusBadge } from "../components/Badges";
import { EmptyState, ErrorBox, Loading, PageHeader, Panel } from "../components/Panel";
import { useEvents } from "../events";
import { formatTimestamp, shortId } from "../format";
import { experimentPath, hashHref, runPath } from "../router";

function newestFirst(a: RunSummary, b: RunSummary): number {
  return b.created_at - a.created_at;
}

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [runResult, experimentResult] = await Promise.all([
        api.runs(),
        api.experiments(),
      ]);
      setRuns(runResult.runs);
      setExperiments(experimentResult.experiments);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load runs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const connected = useEvents((event) => {
    if (
      event.type === "run_started" ||
      event.type === "run_finished" ||
      event.type === "experiment_started" ||
      event.type === "experiment_finished"
    ) {
      void load();
    }
  });

  const ordered = useMemo(() => {
    const byId = new Map(runs.map((run) => [run.run_id, run]));
    const children = new Map<string, RunSummary[]>();
    for (const run of runs) {
      if (run.parent_run_id && byId.has(run.parent_run_id)) {
        const siblings = children.get(run.parent_run_id) ?? [];
        siblings.push(run);
        children.set(run.parent_run_id, siblings);
      }
    }
    for (const siblings of children.values()) {
      siblings.sort((a, b) => a.created_at - b.created_at);
    }
    const roots = runs
      .filter((run) => !run.parent_run_id || !byId.has(run.parent_run_id))
      .sort(newestFirst);
    return roots.flatMap((root) => [root, ...(children.get(root.run_id) ?? [])]);
  }, [runs]);

  if (loading) {
    return <Loading label="loading runs" />;
  }

  return (
    <div data-testid="runs-page">
      <PageHeader
        eyebrow="local cassette"
        title="Runs"
        subtitle={
          <>
            {runs.length} recorded {runs.length === 1 ? "run" : "runs"} ·{" "}
            {experiments.length} {experiments.length === 1 ? "experiment" : "experiments"}
          </>
        }
        actions={
          <>
            <LiveDot connected={connected} />
            <button
              type="button"
              onClick={() => void load()}
              className="rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-xs font-medium text-ink-200 transition hover:border-ink-500 hover:text-ink-100"
            >
              Refresh
            </button>
          </>
        }
      />

      {error ? <ErrorBox message={error} /> : null}

      {ordered.length === 0 ? (
        <EmptyState title="No runs recorded yet">
          Run an agent with <code className="text-accent">checkpoint run</code> and it
          will appear here. Forks and replay runs keep their lineage in this list.
        </EmptyState>
      ) : (
        <Panel className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-left text-sm">
              <thead className="border-b border-ink-700/80 bg-ink-900/60 text-[10px] uppercase tracking-[0.16em] text-ink-400">
                <tr>
                  <th className="px-5 py-3 font-medium">Run</th>
                  <th className="px-3 py-3 font-medium">Mode</th>
                  <th className="px-3 py-3 font-medium">Status</th>
                  <th className="px-3 py-3 text-right font-medium">Steps</th>
                  <th className="px-3 py-3 font-medium">Variant</th>
                  <th className="px-3 py-3 font-medium">Lineage</th>
                  <th className="px-5 py-3 font-medium">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-800/80">
                {ordered.map((run) => {
                  const isChild = Boolean(run.parent_run_id);
                  const childCount = runs.filter(
                    (candidate) => candidate.parent_run_id === run.run_id,
                  ).length;
                  return (
                    <tr
                      key={run.run_id}
                      data-testid="run-row"
                      data-run-id={run.run_id}
                      data-run-name={run.name}
                      className="group transition hover:bg-ink-800/40"
                    >
                      <td className={`py-3.5 pr-3 ${isChild ? "pl-10" : "pl-5"}`}>
                        <div className="flex items-start gap-2">
                          {isChild ? (
                            <span className="mt-0.5 font-mono text-ink-500">↳</span>
                          ) : null}
                          <div className="min-w-0">
                            <a
                              href={hashHref(runPath(run.run_id))}
                              className="block truncate font-medium text-ink-100 transition group-hover:text-accent"
                            >
                              {run.name}
                            </a>
                            <div className="mt-0.5 font-mono text-[10px] text-ink-500">
                              {shortId(run.run_id)}
                              {run.experiment_id ? ` · exp ${shortId(run.experiment_id)}` : ""}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-3.5">
                        <ModeBadge mode={run.mode} />
                      </td>
                      <td className="px-3 py-3.5">
                        <StatusBadge status={run.status} />
                      </td>
                      <td className="px-3 py-3.5 text-right font-mono text-xs text-ink-300">
                        {run.step_count ?? "—"}
                      </td>
                      <td className="px-3 py-3.5 text-xs text-ink-300">
                        {run.variant_name ? (
                          <span className="font-mono text-[11px] text-warn">
                            {run.variant_name}
                          </span>
                        ) : (
                          <span className="text-ink-500">—</span>
                        )}
                      </td>
                      <td className="px-3 py-3.5 text-xs text-ink-400">
                        {run.parent_run_id ? (
                          <a
                            href={hashHref(runPath(run.parent_run_id))}
                            className="font-mono text-[10px] text-ink-400 hover:text-accent"
                            title={run.parent_run_id}
                          >
                            {shortId(run.parent_run_id)}
                            {run.forked_at_step !== null ? ` @ ${run.forked_at_step}` : ""}
                          </a>
                        ) : childCount > 0 ? (
                          <span className="text-ink-500">{childCount} forks</span>
                        ) : (
                          <span className="text-ink-500">root</span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 text-xs text-ink-400">
                        {formatTimestamp(run.created_at)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <div className="mt-10">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-ink-400">
            Experiments
          </h2>
        </div>
        {experiments.length === 0 ? (
          <EmptyState title="No experiments yet">
            Fork a run with <code className="text-accent">checkpoint fork</code> to
            compare variants side by side.
          </EmptyState>
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {experiments.map((experiment) => (
              <a
                key={experiment.experiment_id}
                href={hashHref(experimentPath(experiment.experiment_id))}
                data-testid="experiment-card"
                className="panel group block p-5 transition hover:border-accent/40 hover:bg-ink-800/70"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="truncate text-sm font-medium text-ink-100 group-hover:text-accent">
                    {experiment.name}
                  </div>
                  <span className="shrink-0 rounded border border-warn/25 bg-warn/10 px-1.5 py-0.5 font-mono text-[10px] text-warn">
                    fork @ {experiment.fork_step_index}
                  </span>
                </div>
                <div className="mt-4 flex items-center justify-between font-mono text-[10px] text-ink-500">
                  <span>{shortId(experiment.experiment_id)}</span>
                  <span>
                    {experiment.run_count ?? 0} variants · {formatTimestamp(experiment.created_at)}
                  </span>
                </div>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
