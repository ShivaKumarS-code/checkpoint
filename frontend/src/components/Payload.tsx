import { pretty } from "../format";

export function Payload({
  title,
  value,
  testId,
  tone = "default",
}: {
  title: string;
  value: unknown;
  testId?: string;
  tone?: "default" | "muted";
}) {
  const text = pretty(value);
  return (
    <div className="min-w-0" data-testid={testId}>
      <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-400">
        {title}
      </div>
      <pre
        className={`max-h-80 overflow-auto rounded-lg border border-ink-700/80 p-3 font-mono text-[11px] leading-5 ${
          tone === "muted" ? "bg-ink-900/60 text-ink-300" : "bg-ink-950/80 text-ink-200"
        }`}
      >
        {text}
      </pre>
    </div>
  );
}
