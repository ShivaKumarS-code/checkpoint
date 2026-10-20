import { useEffect, useState } from "react";

import { api, type RunDetail, type Step, type StepDetail } from "../api";
import { SourceBadge, StatusBadge, StepTypeBadge } from "../components/Badges";
import { ErrorBox, Loading, PageHeader, Panel } from "../components/Panel";
import { Payload } from "../components/Payload";
import { firstDivergence } from "../diff";
import { formatDuration, shortId } from "../format";
import { hashHref, runPath } from "../router";

interface DiffData {
  leftRun: RunDetail;
  rightRun: RunDetail;
  leftSteps: Step[];
  rightSteps: Step[];
}

export function DiffPage({ left, right }: { left: string; right: string }) {
  const [data, setData] = useState<DiffData | null>(null);
  const [leftDetail, setLeftDetail] = useState<StepDetail | null>(null);
  const [rightDetail, setRightDetail] = useState<StepDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [payloadLoading, setPayloadLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void (async () => {
      try {
        const [leftRun, rightRun, leftSteps, rightSteps] = await Promise.all([
          api.run(left),
          api.run(right),
          api.steps(left),
          api.steps(right),
        ]);
        if (!cancelled) {
          setData({
            leftRun,
            rightRun,
            leftSteps: leftSteps.steps,
            rightSteps: rightSteps.steps,
          });
          setError(null);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "failed to load diff");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [left, right]);

  const divergentIndex = data
    ? firstDivergence(data.leftSteps, data.rightSteps)
    : null;

  useEffect(() => {
    if (!data || divergentIndex === null) {
      setLeftDetail(null);
      setRightDetail(null);
      return;
    }
    const leftStep = data.leftSteps[divergentIndex];
    const rightStep = data.rightSteps[divergentIndex];
    if (!leftStep && !rightStep) {
      return;
    }
    let cancelled = false;
    setPayloadLoading(true);
    void (async () => {
      try {
        const [leftPayload, rightPayload] = await Promise.all([
          leftStep ? api.step(leftStep.step_id) : Promise.resolve(null),
          rightStep ? api.step(rightStep.step_id) : Promise.resolve(null),
        ]);
        if (!cancelled) {
          setLeftDetail(leftPayload);
          setRightDetail(rightPayload);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "failed to load step payloads");
        }
      } finally {
        if (!cancelled) {
          setPayloadLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [data, divergentIndex]);

  if (loading) {
    return <Loading label="comparing runs" />;
  }
  if (error) {
    return (
      <div>
        <PageHeader title="Diff unavailable" />
        <ErrorBox message={error} />
      </div>
    );
  }
  if (!data) {
    return null;
  }

  const leftStep = divergentIndex === null ? null : data.leftSteps[divergentIndex];
  const rightStep = divergentIndex === null ? null : data.rightSteps[divergentIndex];

  return (
    <div data-testid="diff-page">
      <div className="mb-4 flex items-center gap-2 text-xs text-ink-500">
        <a href={hashHref("/")} className="text-ink-400 transition hover:text-accent">
          Runs
        </a>
        <span className="text-ink-600">/</span>
        <span className="text-ink-300">branch diff</span>
      </div>

      <PageHeader
        eyebrow="BRANCH DIFF"
        title={
          <span data-testid="diff-title">
            {divergentIndex === null
              ? "No divergent steps"
              : `First divergence at step ${divergentIndex}`}
          </span>
        }
        subtitle={
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="text-ink-400">{data.leftRun.run.name}</span>
            <span className="font-mono text-ink-600">{shortId(left)}</span>
            <span className="text-ink-600">vs</span>
            <span className="text-ink-400">{data.rightRun.run.name}</span>
            <span className="font-mono text-ink-600">{shortId(right)}</span>
          </div>
        }
        actions={
          <>
            <a
              href={hashHref(runPath(left))}
              className="rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-xs text-ink-200 transition hover:border-accent/40 hover:text-accent"
            >
              Open left run
            </a>
            <a
              href={hashHref(runPath(right))}
              className="rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-xs text-ink-200 transition hover:border-accent/40 hover:text-accent"
            >
              Open right run
            </a>
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <RunColumn
          side="left"
          run={data.leftRun}
          steps={data.leftSteps}
          step={leftStep}
          payload={leftDetail}
          payloadLoading={payloadLoading && Boolean(leftStep)}
          highlight={divergentIndex}
        />
        <RunColumn
          side="right"
          run={data.rightRun}
          steps={data.rightSteps}
          step={rightStep}
          payload={rightDetail}
          payloadLoading={payloadLoading && Boolean(rightStep)}
          highlight={divergentIndex}
        />
      </div>

      {divergentIndex === null ? (
        <div className="mt-6 rounded-lg border border-good/30 bg-good/10 px-4 py-3 text-sm text-good">
          Both runs have {data.leftSteps.length} matching step inputs. Their outputs may
          still differ.
        </div>
      ) : null}
    </div>
  );
}

function RunColumn({
  side,
  run,
  steps,
  step,
  payload,
  payloadLoading,
  highlight,
}: {
  side: "left" | "right";
  run: RunDetail;
  steps: Step[];
  step: Step | null;
  payload: StepDetail | null;
  payloadLoading: boolean;
  highlight: number | null;
}) {
  return (
    <Panel
      testId={`diff-${side}`}
      className={`overflow-hidden ${side === "right" ? "border-violet/25" : ""}`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-ink-700/80 px-4 py-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-ink-100">{run.run.name}</div>
          <div className="mt-1 font-mono text-[10px] text-ink-500">
            {shortId(run.run.run_id)} · {run.run.mode}
          </div>
        </div>
        <StatusBadge status={run.run.status} />
      </div>

      <div className="border-b border-ink-800 px-4 py-3">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink-500">
          steps through divergence
        </div>
        <ol className="space-y-1">
          {steps.slice(0, highlight === null ? steps.length : highlight + 1).map((item) => (
            <li
              key={item.step_id}
              className={`flex items-center gap-2 rounded px-2 py-1 font-mono text-[10px] ${
                item.step_index === highlight
                  ? "bg-warn/10 text-warn"
                  : "text-ink-400"
              }`}
            >
              <span className="w-5 text-right text-ink-500">{item.step_index}</span>
              <StepTypeBadge type={item.step_type} />
              <span className="truncate">{item.name}</span>
              <span className="ml-auto">
                <SourceBadge source={item.source} />
              </span>
            </li>
          ))}
          {steps.length === 0 ? (
            <li className="px-2 py-1 text-[10px] text-ink-500">no steps</li>
          ) : null}
        </ol>
      </div>

      <div className="space-y-4 p-4">
        {step ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs text-ink-100">
                {String(step.step_index).padStart(2, "0")} · {step.name}
              </span>
              <StepTypeBadge type={step.step_type} />
              <SourceBadge source={step.source} />
              <span className="ml-auto font-mono text-[10px] text-ink-500">
                {formatDuration(step.latency_ms)}
              </span>
            </div>
            {step.error ? <ErrorBox message={step.error} /> : null}
            {payloadLoading ? <Loading label="loading payload" /> : null}
            {payload ? (
              <div className="space-y-4">
                <Payload
                  title="input"
                  value={payload.input}
                  testId={`diff-${side}-input`}
                />
                <Payload
                  title="output"
                  value={payload.output}
                  testId={`diff-${side}-output`}
                  tone="muted"
                />
                {payload.post_state !== null && payload.post_state !== undefined ? (
                  <Payload title="post state" value={payload.post_state} />
                ) : null}
              </div>
            ) : null}
          </>
        ) : (
          <div className="rounded-lg border border-dashed border-ink-700 px-4 py-8 text-center text-xs text-ink-500">
            No recorded step at the divergence point.
          </div>
        )}
      </div>
    </Panel>
  );
}
