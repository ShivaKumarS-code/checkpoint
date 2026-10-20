import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";

import {
  api,
  type ExperimentDetail,
  type ExperimentRun,
  type RunSummary,
  type Verdict,
} from "../api";
import { Badge, LiveDot, StatusBadge } from "../components/Badges";
import { EmptyState, ErrorBox, Loading, PageHeader, Panel } from "../components/Panel";
import { useEvents } from "../events";
import { checkLabel, formatTimestamp, shortId } from "../format";
import { diffPath, hashHref, runPath } from "../router";

type RunNodeData = {
  title: string;
  subtitle: string;
  kind: "root" | "fork" | "variant";
  status: string;
  passed?: boolean;
  runId?: string;
  testId: string;
};

type RunNodeType = Node<RunNodeData, "run">;

function RunNode({ data }: NodeProps<RunNodeType>) {
  const border =
    data.kind === "fork"
      ? "border-warn/40 bg-warn/10"
      : data.passed === false
        ? "border-bad/50 bg-bad/10"
        : data.passed === true
          ? "border-good/50 bg-good/10"
          : "border-accent/35 bg-accent-soft/40";
  return (
    <div
      data-testid={data.testId}
      data-variant-status={data.status}
      className={`w-[220px] rounded-xl border px-4 py-3 shadow-panel backdrop-blur ${border}`}
    >
      <Handle type="target" position={Position.Top} className="!h-0 !w-0 !border-0 !opacity-0" />
      <div className="flex items-center justify-between gap-2">
        <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-ink-400">
          {data.kind === "fork" ? "fork" : data.kind}
        </span>
        {data.kind === "fork" ? (
          <span className="h-1.5 w-1.5 rounded-full bg-warn" />
        ) : (
          <StatusBadge status={data.status} />
        )}
      </div>
      <div className="mt-2 truncate text-sm font-medium text-ink-100">{data.title}</div>
      <div className="mt-1 truncate font-mono text-[10px] text-ink-400">{data.subtitle}</div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-0 !w-0 !border-0 !opacity-0"
      />
    </div>
  );
}

const nodeTypes: NodeTypes = { run: RunNode };

function verdictFor(run: ExperimentRun): boolean {
  if (run.verdicts.length > 0) {
    return run.verdicts.every((verdict) => verdict.passed);
  }
  return run.run.status === "COMPLETED" && run.run.divergence_step === null;
}

function VerdictRow({ verdict }: { verdict: Verdict }) {
  return (
    <div className="flex items-start gap-2 border-t border-ink-800/70 py-1.5 first:border-0">
      <span className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${verdict.passed ? "bg-good" : "bg-bad"}`} />
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-[11px] text-ink-200">{verdict.metric_name}</span>
          {verdict.tier === "TIER2" ? (
            <span className="rounded border border-violet/30 bg-violet/10 px-1 py-0.5 font-mono text-[8px] text-violet">
              TIER 2
            </span>
          ) : null}
          {verdict.tier === "TIER3" ? (
            <span className="rounded border border-accent/35 bg-accent-soft/40 px-1 py-0.5 font-mono text-[8px] text-accent-dim">
              TIER 3
            </span>
          ) : null}
          {verdict.position_consistent === false ? (
            <span
              className="rounded border border-bad/50 bg-bad/10 px-1 py-0.5 font-mono text-[8px] text-bad"
              title="The judge picked a different winner when the runs were swapped."
            >
              ORDER BIAS
            </span>
          ) : null}
          {verdict.position_consistent === true ? (
            <span
              className="rounded border border-good/30 bg-good/10 px-1 py-0.5 font-mono text-[8px] text-good"
              title="The judge selected the same run in both presentation orders."
            >
              SWAP OK
            </span>
          ) : null}
          {typeof verdict.score === "number" ? (
            <span className="font-mono text-[9px] text-ink-500">
              {verdict.score.toFixed(2)}
            </span>
          ) : null}
        </div>
        {verdict.compared_run_id ? (
          <a
            href={hashHref(runPath(verdict.compared_run_id))}
            className="mt-0.5 inline-block font-mono text-[9px] text-ink-500 transition hover:text-accent"
          >
            vs {shortId(verdict.compared_run_id)}
          </a>
        ) : null}
        <div className="mt-0.5 line-clamp-2 text-[10px] leading-4 text-ink-500">
          {typeof verdict.reasoning === "string"
            ? verdict.reasoning
            : JSON.stringify(verdict.reasoning)}
        </div>
      </div>
    </div>
  );
}

export function ExperimentPage({ experimentId }: { experimentId: string }) {
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [root, setRoot] = useState<RunSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const experiment = await api.experiment(experimentId);
      setDetail(experiment);
      try {
        const rootDetail = await api.run(experiment.experiment.parent_run_id);
        setRoot(rootDetail.run);
      } catch {
        setRoot(null);
      }
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load experiment");
    } finally {
      setLoading(false);
    }
  }, [experimentId]);

  useEffect(() => {
    void load();
  }, [load]);

  const connected = useEvents((event) => {
    if (
      event.experiment_id === experimentId ||
      event.experiment?.experiment_id === experimentId
    ) {
      void load();
    }
  });

  const variants = useMemo(() => {
    if (!detail) {
      return [];
    }
    const byVariant = new Map<string, ExperimentRun>();
    for (const run of detail.runs) {
      if (run.run.variant_name) {
        byVariant.set(run.run.variant_name, run);
      }
    }
    return detail.variants.map((variant) => ({
      name: variant.name,
      run: byVariant.get(variant.name) ?? null,
    }));
  }, [detail]);

  const graph = useMemo(() => {
    if (!detail) {
      return { nodes: [] as RunNodeType[], edges: [] as Edge[] };
    }
    const spacing = 260;
    const offset = ((variants.length - 1) * spacing) / 2;
    const nodes: RunNodeType[] = [
      {
        id: "root",
        type: "run",
        position: { x: -110, y: 0 },
        data: {
          title: root?.name ?? "parent run",
          subtitle: root ? `${root.mode} · ${shortId(root.run_id)}` : shortId(detail.experiment.parent_run_id),
          kind: "root",
          status: root?.status ?? "COMPLETED",
          runId: detail.experiment.parent_run_id,
          testId: "dag-node-root",
        },
      },
      {
        id: "fork",
        type: "run",
        position: { x: -110, y: 128 },
        data: {
          title: `fork at step ${detail.experiment.fork_step_index}`,
          subtitle: "pre-fork replay, then live",
          kind: "fork",
          status: "FORK",
          testId: "dag-node-fork",
        },
      },
    ];
    const edges: Edge[] = [
      {
        id: "root-fork",
        source: "root",
        target: "fork",
        type: "smoothstep",
        style: { stroke: "#4b586d", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#4b586d" },
      },
    ];
    variants.forEach((variant, index) => {
      const id = `variant-${variant.name}`;
      const passed = variant.run ? verdictFor(variant.run) : undefined;
      const color = passed === false ? "#ff7b93" : passed === true ? "#5ee2a0" : "#4b586d";
      nodes.push({
        id,
        type: "run",
        position: { x: index * spacing - offset - 110, y: 286 },
        data: {
          title: variant.name,
          subtitle: variant.run
            ? `${variant.run.run.status} · ${shortId(variant.run.run.run_id)}`
            : "not started",
          kind: "variant",
          status: variant.run?.run.status ?? "PENDING",
          passed,
          runId: variant.run?.run.run_id,
          testId: `dag-node-${variant.name}`,
        },
      });
      edges.push({
        id: `fork-${id}`,
        source: "fork",
        target: id,
        type: "smoothstep",
        animated: variant.run?.run.status === "RUNNING",
        style: { stroke: color, strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color },
      });
    });
    return { nodes, edges };
  }, [detail, root, variants]);

  if (loading) {
    return <Loading label="loading experiment" />;
  }
  if (error) {
    return (
      <div>
        <PageHeader title="Experiment unavailable" />
        <ErrorBox message={error} />
      </div>
    );
  }
  if (!detail) {
    return null;
  }

  return (
    <div data-testid="experiment-page">
      <div className="mb-4 flex items-center gap-2 text-xs text-ink-500">
        <a href={hashHref("/")} className="text-ink-400 transition hover:text-accent">
          Runs
        </a>
        <span className="text-ink-600">/</span>
        <span className="text-ink-300">experiment</span>
      </div>

      <PageHeader
        eyebrow="FORK EXPERIMENT"
        title={detail.experiment.name}
        subtitle={
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[11px] text-ink-500">
              {detail.experiment.experiment_id}
            </span>
            <span className="rounded border border-warn/25 bg-warn/10 px-1.5 py-0.5 font-mono text-[10px] text-warn">
              fork @ step {detail.experiment.fork_step_index}
            </span>
            <span className="text-[11px] text-ink-500">
              {detail.runs.length} variants · {formatTimestamp(detail.experiment.created_at)}
            </span>
            {detail.semantic_matches ? (
              <span
                data-testid="semantic-summary"
                title={detail.semantic_steps
                  ?.map((step) => `${step.variant_name}: ${step.name}`)
                  .join(", ")}
              >
                <Badge tone="green">
                  {detail.semantic_summary ??
                    `${detail.semantic_matches} steps served by semantic match`}
                </Badge>
              </span>
            ) : null}
          </div>
        }
        actions={
          <>
            <LiveDot connected={connected} />
            <a
              href={hashHref(runPath(detail.experiment.parent_run_id))}
              className="rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-xs font-medium text-ink-200 transition hover:border-accent/40 hover:text-accent"
            >
              Open parent run
            </a>
          </>
        }
      />

      <Panel className="mb-6 overflow-hidden" testId="experiment-dag">
        <div className="flex items-center justify-between border-b border-ink-700/80 px-5 py-3">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-400">
            Fork graph
          </h2>
          <span className="font-mono text-[10px] text-ink-500">
            root → fork @ {detail.experiment.fork_step_index} → variants
          </span>
        </div>
        <div className="h-[460px] w-full" data-testid="dag">
          <ReactFlow
            nodes={graph.nodes}
            edges={graph.edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.28 }}
            nodesConnectable={false}
            nodesDraggable={false}
            elementsSelectable
            proOptions={{ hideAttribution: true }}
            onNodeClick={(_, node) => {
              const data = node.data as RunNodeData;
              if (data.runId) {
                window.location.hash = runPath(data.runId);
              }
            }}
          >
            <Background color="#222937" gap={26} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </Panel>

      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.2em] text-ink-400">
          Variants
        </h2>
        {detail.checks.length > 0 ? (
          <span className="font-mono text-[10px] text-ink-500">
            {detail.checks.map(checkLabel).join(" · ")}
          </span>
        ) : null}
      </div>

      {variants.length === 0 ? (
        <EmptyState title="No variants in this experiment" />
      ) : (
        <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {variants.map((variant) => {
            const run = variant.run;
            const passed = run ? verdictFor(run) : null;
            return (
              <Panel
                key={variant.name}
                testId={`variant-row-${variant.name}`}
                className={`p-4 ${
                  passed === false
                    ? "border-bad/30"
                    : passed === true
                      ? "border-good/30"
                      : ""
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-ink-100">
                      {variant.name}
                    </div>
                    <div className="mt-1 font-mono text-[10px] text-ink-500">
                      {run ? `${run.run.status} · ${shortId(run.run.run_id)}` : "not started"}
                    </div>
                  </div>
                  <StatusBadge status={run?.run.status ?? "PENDING"} />
                </div>

                <div className="mt-3 min-h-[52px]">
                  {run && run.verdicts.length > 0 ? (
                    run.verdicts.map((verdict) => (
                      <VerdictRow key={verdict.verdict_id} verdict={verdict} />
                    ))
                  ) : (
                    <div className="text-[11px] text-ink-500">No checks recorded.</div>
                  )}
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-ink-800 pt-3">
                  {run?.semantic_matches ? (
                    <Badge
                      tone="green"
                      className="normal-case tracking-normal"
                    >
                      {run.semantic_matches} semantic
                    </Badge>
                  ) : null}
                  {run ? (
                    <>
                      <a
                        href={hashHref(runPath(run.run.run_id))}
                        className="rounded-md border border-ink-600 bg-ink-800 px-2.5 py-1 text-[11px] text-ink-200 transition hover:border-accent/40 hover:text-accent"
                      >
                        View run
                      </a>
                      <a
                        href={hashHref(diffPath(detail.experiment.parent_run_id, run.run.run_id))}
                        data-testid={`compare-${variant.name}`}
                        className="rounded-md border border-accent/30 bg-accent-soft/50 px-2.5 py-1 text-[11px] text-accent-dim transition hover:border-accent/60 hover:text-accent"
                      >
                        Compare with parent
                      </a>
                    </>
                  ) : (
                    <span className="text-[10px] text-ink-500">waiting for this variant</span>
                  )}
                  <span className="ml-auto font-mono text-[10px] text-ink-500">
                    {run ? `${run.run.total_latency_ms.toFixed(0)} ms` : ""}
                  </span>
                </div>
              </Panel>
            );
          })}
        </div>
      )}
    </div>
  );
}
