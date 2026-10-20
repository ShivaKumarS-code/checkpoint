export function shortId(id: string | null | undefined): string {
  if (!id) {
    return "—";
  }
  return id.length > 10 ? id.slice(0, 8) : id;
}

export function formatTimestamp(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) {
    return "—";
  }
  const date = new Date(seconds * 1000);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatDuration(milliseconds: number | null | undefined): string {
  if (milliseconds === null || milliseconds === undefined) {
    return "—";
  }
  if (milliseconds < 1000) {
    return `${milliseconds.toFixed(1)} ms`;
  }
  return `${(milliseconds / 1000).toFixed(2)} s`;
}

export function pretty(value: unknown): string {
  if (value === undefined) {
    return "—";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

export function checkLabel(check: { type?: string; name?: string }): string {
  return [check.type, check.name].filter(Boolean).join(":") || "check";
}
