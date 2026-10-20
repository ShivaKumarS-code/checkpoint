import { useEffect, useState } from "react";

export type Route =
  | { page: "runs" }
  | { page: "run"; id: string }
  | { page: "experiment"; id: string }
  | { page: "diff"; left: string; right: string }
  | { page: "unknown"; path: string };

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#/, "") || "/";
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts.length === 0) {
    return { page: "runs" };
  }
  if (parts[0] === "runs" && parts[1]) {
    return { page: "run", id: parts[1] };
  }
  if (parts[0] === "experiments" && parts[1]) {
    return { page: "experiment", id: parts[1] };
  }
  if (parts[0] === "diff" && parts[1] && parts[2]) {
    return { page: "diff", left: parts[1], right: parts[2] };
  }
  return { page: "unknown", path };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function hashHref(path: string): string {
  return `#${path}`;
}

export function navigate(path: string): void {
  window.location.hash = path;
}

export function runPath(runId: string): string {
  return `/runs/${encodeURIComponent(runId)}`;
}

export function experimentPath(experimentId: string): string {
  return `/experiments/${encodeURIComponent(experimentId)}`;
}

export function diffPath(left: string, right: string): string {
  return `/diff/${encodeURIComponent(left)}/${encodeURIComponent(right)}`;
}
