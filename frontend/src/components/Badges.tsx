import type { ReactNode } from "react";

type Tone = "neutral" | "cyan" | "violet" | "green" | "red" | "amber";

const tones: Record<Tone, string> = {
  neutral: "border-ink-600/70 bg-ink-700/60 text-ink-200",
  cyan: "border-accent/30 bg-accent-soft/70 text-accent-dim",
  violet: "border-violet/30 bg-violet/10 text-violet",
  green: "border-good/30 bg-good/10 text-good",
  red: "border-bad/30 bg-bad/10 text-bad",
  amber: "border-warn/30 bg-warn/10 text-warn",
};

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const tone: Tone =
    status === "COMPLETED"
      ? "green"
      : status === "RUNNING"
        ? "cyan"
        : status === "DIVERGED"
          ? "amber"
          : status === "FAILED"
            ? "red"
            : "neutral";
  return <Badge tone={tone}>{status}</Badge>;
}

export function ModeBadge({ mode }: { mode: string }) {
  const tone: Tone = mode === "FORK" ? "amber" : mode === "REPLAY" ? "violet" : "cyan";
  return <Badge tone={tone}>{mode}</Badge>;
}

export function SourceBadge({ source }: { source: string }) {
  const tone: Tone =
    source === "REPLAYED"
      ? "violet"
      : source === "LIVE"
        ? "amber"
        : source === "SEMANTIC"
          ? "green"
          : "cyan";
  return <Badge tone={tone}>{source}</Badge>;
}

export function StepTypeBadge({ type }: { type: string }) {
  const tone: Tone =
    type === "LLM_CALL" ? "violet" : type === "CHECKPOINT" ? "green" : "cyan";
  const label = type === "LLM_CALL" ? "LLM" : type === "CHECKPOINT" ? "STATE" : "TOOL";
  return <Badge tone={tone}>{label}</Badge>;
}

export function LiveDot({ connected }: { connected: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-ink-400">
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          connected ? "bg-good shadow-[0_0_8px_rgba(94,226,160,0.8)]" : "bg-ink-500"
        }`}
      />
      {connected ? "live" : "disconnected"}
    </span>
  );
}
