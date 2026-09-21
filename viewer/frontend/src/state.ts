export type ViewState = {
  model: string;
  compare: string;
  event: string;
  sample: string;
  layer: string;
};

const KEYS: (keyof ViewState)[] = ["model", "compare", "event", "sample", "layer"];

export function readState(): ViewState {
  const q = new URLSearchParams(location.search);
  return {
    model: q.get("model") ?? "",
    compare: q.get("compare") ?? "",
    event: q.get("event") ?? "",
    sample: q.get("sample") ?? "",
    layer: q.get("layer") ?? "confusion",
  };
}

export function writeState(s: ViewState): void {
  const q = new URLSearchParams();
  for (const k of KEYS) {
    if (s[k]) q.set(k, s[k]);
  }
  const next = `${location.pathname}?${q}`;
  history.replaceState(s, "", next);
}
