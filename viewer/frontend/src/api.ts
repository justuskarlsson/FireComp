export type Metrics = {
  iou: number;
  precision: number;
  recall: number;
  f1: number;
  brier: number;
  mean_prob: number;
  tp: number;
  fp: number;
  fn: number;
};

export type Sample = {
  id: string;
  event_id: number;
  xi: number;
  yi: number;
  dt: string;
  day_of_fire: number;
  lon: number;
  lat: number;
  bbox: [number, number, number, number];
  img_size: number;
  region_id: number;
  region: string;
  country: string;
  num_fire: number;
  n_gt_px: number;
  area_km2: number;
  metrics: Record<string, Metrics>;
};

export type Event = {
  id: number;
  region: string;
  country: string;
  lon: number;
  lat: number;
  bbox: [number, number, number, number];
  start: string;
  end: string;
  duration_days: number;
  n_samples: number;
  area_km2: number;
  sample_ids: string[];
  metrics: Record<string, { iou: number }>;
};

/** Event with GT pixel count and mean per-model metrics from its samples. */
export type EventRow = Event & {
  n_gt_px: number;
  mean_metrics: Record<string, Pick<Metrics, "iou" | "precision" | "recall" | "f1">>;
};

export type ModelInfo = {
  id: string;
  label: string;
  kind: string;
  threshold: number;
  n_samples: number;
};

export type LayerInfo = {
  id: string;
  label: string;
  kind: string;
};

export type Index = {
  generated_at: string;
  split: string;
  fire_type: string;
  target: string;
  models: ModelInfo[];
  n_samples: number;
  n_events: number;
  events: Event[];
  samples: Sample[];
  layers: LayerInfo[];
  padding: number;
};

export async function fetchIndex(): Promise<Index> {
  const res = await fetch("/api/index");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export function layerUrl(sampleId: string, name: string, model: string): string {
  const q = new URLSearchParams();
  if (model) q.set("model", model);
  return `/api/samples/${encodeURIComponent(sampleId)}/layers/${name}.png?${q}`;
}

export function legendUrl(name: string): string {
  return `/api/legend/${name}.png`;
}

export function enrichEvents(events: Event[], samples: Sample[]): EventRow[] {
  const byId = new Map(samples.map((s) => [s.id, s]));
  return events.map((e) => {
    const days = e.sample_ids
      .map((id) => byId.get(id))
      .filter((s): s is Sample => s !== undefined);
    const n_gt_px = days.reduce((n, s) => n + s.n_gt_px, 0);
    const mean_metrics: EventRow["mean_metrics"] = {};
    const models = new Set(days.flatMap((s) => Object.keys(s.metrics)));
    for (const mid of models) {
      const rows = days.map((s) => s.metrics[mid]).filter(Boolean);
      if (rows.length === 0) continue;
      mean_metrics[mid] = {
        iou: mean(rows.map((r) => r.iou)),
        precision: mean(rows.map((r) => r.precision)),
        recall: mean(rows.map((r) => r.recall)),
        f1: mean(rows.map((r) => r.f1)),
      };
    }
    return { ...e, n_gt_px, mean_metrics };
  });
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
