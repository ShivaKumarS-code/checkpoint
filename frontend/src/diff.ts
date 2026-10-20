import type { Step } from "./api";

export function firstDivergence(left: Step[], right: Step[]): number | null {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const a = left[index];
    const b = right[index];
    if (
      a.step_type !== b.step_type ||
      a.name !== b.name ||
      (a.match_key ?? "") !== (b.match_key ?? "")
    ) {
      return index;
    }
  }
  if (left.length !== right.length) {
    return length;
  }
  return null;
}
