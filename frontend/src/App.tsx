import { useRoute } from "./router";
import { RunsPage } from "./pages/RunsPage";
import { RunPage } from "./pages/RunPage";
import { ExperimentPage } from "./pages/ExperimentPage";
import { DiffPage } from "./pages/DiffPage";

export default function App() {
  const route = useRoute();

  let page;
  switch (route.page) {
    case "run":
      page = <RunPage key={route.id} runId={route.id} />;
      break;
    case "experiment":
      page = <ExperimentPage key={route.id} experimentId={route.id} />;
      break;
    case "diff":
      page = <DiffPage key={`${route.left}:${route.right}`} left={route.left} right={route.right} />;
      break;
    case "unknown":
      page = (
        <div className="panel px-6 py-14 text-center">
          <div className="text-sm text-ink-200">Page not found</div>
          <div className="mt-2 font-mono text-xs text-ink-500">{route.path}</div>
          <a href="#/" className="mt-4 inline-block text-xs text-accent">
            Back to runs
          </a>
        </div>
      );
      break;
    default:
      page = <RunsPage />;
  }

  return (
    <div
      data-testid="app-shell"
      data-theme="dark"
      className="relative min-h-screen overflow-x-hidden bg-ink-950 text-ink-100"
    >
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_15%_-10%,rgba(98,217,232,0.12),transparent_34%),radial-gradient(circle_at_85%_0%,rgba(180,156,255,0.08),transparent_30%)]" />
      <div className="pointer-events-none fixed inset-0 bg-[linear-gradient(rgba(148,163,184,0.025)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.025)_1px,transparent_1px)] bg-[size:48px_48px]" />

      <header className="sticky top-0 z-20 border-b border-ink-800/80 bg-ink-950/80 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-5 px-5">
          <a href="#/" className="group flex items-center gap-2.5">
            <span className="relative flex h-7 w-7 items-center justify-center rounded-md border border-accent/40 bg-accent-soft/60">
              <span className="absolute h-3 w-3 rounded-full border border-accent/70" />
              <span className="absolute h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_10px_rgba(98,217,232,0.9)]" />
            </span>
            <span className="font-mono text-sm font-semibold tracking-tight text-ink-100 group-hover:text-accent">
              checkpoint
            </span>
            <span className="hidden rounded border border-ink-700 bg-ink-850 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.16em] text-ink-400 sm:inline">
              replay fork
            </span>
          </a>

          <nav className="ml-2 flex items-center gap-1 text-xs">
            <a
              href="#/"
              className="rounded-md px-2.5 py-1.5 text-ink-300 transition hover:bg-ink-800 hover:text-ink-100"
            >
              Runs
            </a>
            <a
              href="/docs"
              className="rounded-md px-2.5 py-1.5 text-ink-400 transition hover:bg-ink-800 hover:text-ink-100"
            >
              API docs
            </a>
          </nav>

          <div className="ml-auto flex items-center gap-2 text-[10px] text-ink-500">
            <span className="h-1.5 w-1.5 rounded-full bg-good/80" />
            local · 127.0.0.1
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-5 py-8 sm:py-10">{page}</main>

      <footer className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-5 pb-8 pt-4 text-[10px] text-ink-600">
        <span>deterministic replay + fork debugger</span>
        <span className="font-mono">cassette · local only</span>
      </footer>
    </div>
  );
}
