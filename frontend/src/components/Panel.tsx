import type { ReactNode } from "react";

export function Panel({
  children,
  className = "",
  testId,
}: {
  children: ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <section
      data-testid={testId}
      className={`panel ${className}`}
    >
      {children}
    </section>
  );
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
  testId,
}: {
  eyebrow?: string;
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  testId?: string;
}) {
  return (
    <div className="mb-7 flex flex-wrap items-end justify-between gap-4" data-testid={testId}>
      <div className="min-w-0">
        {eyebrow ? (
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.24em] text-accent/80">
            {eyebrow}
          </div>
        ) : null}
        <h1 className="text-3xl font-semibold tracking-tight text-ink-100">{title}</h1>
        {subtitle ? <div className="mt-2 text-sm text-ink-400">{subtitle}</div> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function EmptyState({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="panel flex flex-col items-center gap-2 px-6 py-14 text-center">
      <div className="text-sm font-medium text-ink-200">{title}</div>
      {children ? <div className="max-w-md text-xs leading-6 text-ink-400">{children}</div> : null}
    </div>
  );
}

export function Loading({ label = "loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 px-1 py-8 text-sm text-ink-400">
      <span className="h-3 w-3 animate-spin rounded-full border border-ink-600 border-t-accent" />
      {label}…
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-bad/30 bg-bad/10 px-4 py-3 text-sm text-bad">
      {message}
    </div>
  );
}
