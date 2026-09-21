import "./cesium-base";
import "./style.css";
import {
  enrichEvents, fetchIndex, legendUrl,
  type EventRow, type Index, type ModelInfo, type Sample,
} from "./api";
import { Globe } from "./globe";
import { readState, writeState } from "./state";
import { EventTable } from "./table";

async function main(): Promise<void> {
  const status = el("#status");
  status.textContent = "loading index…";

  let index: Index;
  try {
    index = await fetchIndex();
  } catch (err) {
    status.textContent = err instanceof Error ? err.message : String(err);
    return;
  }

  const events = enrichEvents(index.events, index.samples);
  const state = readState();
  if (!state.model) {
    const preferred = index.models.find((m) => m.id === "unetpp");
    state.model = preferred?.id ?? index.models[0]?.id ?? "";
  }
  if (!state.layer) state.layer = "confusion";

  fillSelect(el("#model-select") as HTMLSelectElement, index.models.map((m) => [m.id, m.label]));
  fillSelect(el("#compare-select") as HTMLSelectElement, [
    ["", "—"],
    ...index.models.map((m) => [m.id, m.label] as [string, string]),
  ]);
  const regions = [...new Set(events.map((e) => e.region))].sort();
  fillSelect(el("#region-select") as HTMLSelectElement, [
    ["", "all"],
    ...regions.map((r): [string, string] => [r, r]),
  ]);
  (el("#model-select") as HTMLSelectElement).value = state.model;
  (el("#compare-select") as HTMLSelectElement).value = state.compare;

  const layersBox = el("#layers");
  for (const layer of index.layers) {
    const lab = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "layer";
    input.value = layer.id;
    input.checked = layer.id === state.layer;
    lab.append(input, document.createTextNode(layer.label));
    layersBox.append(lab);
  }

  const globe = new Globe(el("#cesium-container"));
  globe.padding = index.padding ?? 16;
  const table = new EventTable(el("#table-body"), el("#table-wrap thead"));
  table.setEvents(events, state.model);
  globe.setPoints(events, state.model);
  status.textContent = `${index.n_events} events · ${index.n_samples} samples`;

  const byId = new Map(index.samples.map((s) => [s.id, s]));
  const byEvent = new Map(events.map((e) => [String(e.id), e]));
  const eventSamples = new Map<number, Sample[]>();
  for (const s of index.samples) {
    const list = eventSamples.get(s.event_id) ?? [];
    list.push(s);
    eventSamples.set(s.event_id, list);
  }
  for (const list of eventSamples.values()) list.sort((a, b) => a.dt.localeCompare(b.dt));

  async function selectEvent(eventId: string, fly: boolean): Promise<void> {
    const event = byEvent.get(eventId);
    const all = eventSamples.get(Number(eventId));
    if (!event || !all?.length) return;

    const keep = all.find((s) => s.id === state.sample);
    const sample = keep ?? richestSample(all);
    state.event = eventId;
    state.sample = sample.id;
    writeState(state);
    table.setSelected(eventId);
    globe.highlight(eventId);
    updateMetrics(sample, event, state.model, state.compare);
    updateTimeline(sample, all);
    await globe.showOverlay(sample, state.layer, state.model, opacity(), state.compare);
    setLegend(state.layer);
    setSplitUi(globe, state.model, state.compare, index.models);
    if (fly) globe.flyToSample(sample);
  }

  table.setOnSelect((id) => void selectEvent(id, true));
  globe.setPickHandler((id) => void selectEvent(id, true));

  el("#model-select").addEventListener("change", (e) => {
    state.model = (e.target as HTMLSelectElement).value;
    writeState(state);
    table.setModel(state.model);
    globe.setPoints(events, state.model);
    if (state.event) {
      globe.highlight(state.event);
      void selectEvent(state.event, false);
    }
  });
  el("#compare-select").addEventListener("change", (e) => {
    state.compare = (e.target as HTMLSelectElement).value;
    writeState(state);
    if (state.event) void selectEvent(state.event, false);
  });
  el("#region-select").addEventListener("change", (e) => {
    table.setRegion((e.target as HTMLSelectElement).value);
    status.textContent = `${table.filteredCount} shown · ${index.n_events} events`;
  });
  el("#search").addEventListener("input", (e) => {
    table.setQuery((e.target as HTMLInputElement).value);
    status.textContent = `${table.filteredCount} shown · ${index.n_events} events`;
  });
  layersBox.addEventListener("change", () => {
    const picked = layersBox.querySelector<HTMLInputElement>("input:checked");
    if (!picked) return;
    state.layer = picked.value;
    writeState(state);
    if (state.event) void selectEvent(state.event, false);
  });
  el("#opacity").addEventListener("input", (e) => {
    globe.setOpacity(Number((e.target as HTMLInputElement).value));
  });

  const splitBar = el("#split-bar");
  const splitHandle = el("#split-handle");
  splitHandle.addEventListener("pointerdown", (e: PointerEvent) => {
    splitHandle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  splitHandle.addEventListener("pointermove", (e: PointerEvent) => {
    if (!splitHandle.hasPointerCapture(e.pointerId)) return;
    const box = el("#cesium-container").getBoundingClientRect();
    globe.setSplit((e.clientX - box.left) / box.width);
    splitBar.style.left = `${globe.split * 100}%`;
    splitHandle.setAttribute("aria-valuenow", String(Math.round(globe.split * 100)));
  });

  el("#prev-day").addEventListener("click", () => stepDay(-1));
  el("#next-day").addEventListener("click", () => stepDay(1));
  el("#day-slider").addEventListener("input", (e) => {
    const days = currentTileDays();
    const i = Number((e.target as HTMLInputElement).value);
    if (days[i]) {
      state.sample = days[i].id;
      if (state.event) void selectEvent(state.event, false);
    }
  });
  el("#tile-select").addEventListener("change", (e) => {
    const cur = byId.get(state.sample);
    const all = state.event ? eventSamples.get(Number(state.event)) : undefined;
    if (!cur || !all) return;
    const [xi, yi] = (e.target as HTMLSelectElement).value.split(",").map(Number);
    const next = sampleOnTile(all, xi, yi, cur.dt);
    if (!next) return;
    state.sample = next.id;
    if (state.event) void selectEvent(state.event, true);
  });

  function currentTileDays(): Sample[] {
    if (!state.event) return [];
    const all = eventSamples.get(Number(state.event)) ?? [];
    const cur = byId.get(state.sample);
    if (!cur) return all;
    return all.filter((s) => s.xi === cur.xi && s.yi === cur.yi);
  }
  function stepDay(delta: number): void {
    const days = currentTileDays();
    const i = days.findIndex((s) => s.id === state.sample);
    const next = days[i + delta];
    if (next) {
      state.sample = next.id;
      if (state.event) void selectEvent(state.event, false);
    }
  }

  if (state.event && byEvent.has(state.event)) {
    await selectEvent(state.event, true);
  } else if (state.sample && byId.has(state.sample)) {
    await selectEvent(String(byId.get(state.sample)!.event_id), true);
  }

  status.textContent = `${index.n_events} events · ${index.n_samples} samples`;
}

function updateTimeline(sample: Sample, all: Sample[]): void {
  const tileDays = all.filter((s) => s.xi === sample.xi && s.yi === sample.yi);
  const tiles = uniqueTiles(all);
  const bar = el("#event-bar");
  if (tileDays.length < 2 && tiles.length < 2) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;

  const dayControls = el("#day-controls");
  dayControls.hidden = tileDays.length < 2;
  if (tileDays.length >= 2) {
    const slider = el("#day-slider") as HTMLInputElement;
    slider.max = String(tileDays.length - 1);
    slider.value = String(Math.max(0, tileDays.findIndex((d) => d.id === sample.id)));
  }

  const tileWrap = el("#tile-wrap");
  tileWrap.hidden = tiles.length < 2;
  if (tiles.length >= 2) {
    fillSelect(
      el("#tile-select") as HTMLSelectElement,
      tiles.map((t) => [`${t.xi},${t.yi}`, `${t.xi}, ${t.yi}`]),
    );
    (el("#tile-select") as HTMLSelectElement).value = `${sample.xi},${sample.yi}`;
  }

  const dof = sample.day_of_fire >= 0 ? sample.day_of_fire : (el("#day-slider") as HTMLInputElement).value;
  const tileBit = tiles.length >= 2 ? ` · tile ${sample.xi},${sample.yi}` : "";
  el("#day-label").textContent =
    `day ${dof} · ${sample.dt.slice(0, 10)} · event ${sample.event_id}${tileBit} · ${tileDays.length} days`;
}

function updateMetrics(
  sample: Sample, event: EventRow, model: string, compare: string,
): void {
  const box = el("#metrics");
  const m = sample.metrics[model];
  const c = compare ? sample.metrics[compare] : undefined;
  box.hidden = false;
  const loc = [sample.country, sample.region].filter(Boolean).join(" · ");
  box.innerHTML = `<h2>${loc || "event"}</h2>
    <div class="k">${sample.dt.slice(0, 10)} · event ${event.id} · tile ${sample.xi},${sample.yi} · GT+ ${sample.n_gt_px} px · ${sample.area_km2.toFixed(2)} km²</div>
    <div class="grid" style="margin-top:8px">
      ${metricPair("IoU", m?.iou, c?.iou)}
      ${metricPair("F1", m?.f1, c?.f1)}
      ${metricPair("P", m?.precision, c?.precision)}
      ${metricPair("R", m?.recall, c?.recall)}
    </div>`;
}

function metricPair(label: string, a?: number, b?: number): string {
  const left = a === undefined ? "—" : a.toFixed(3);
  const right = b === undefined ? "" : ` / ${b.toFixed(3)}`;
  return `<span class="k">${label}</span><span>${left}${right}</span>`;
}

function setLegend(layer: string): void {
  const img = el("#legend") as HTMLImageElement;
  img.src = legendUrl(layer);
}

function setSplitUi(
  globe: Globe, model: string, compare: string, models: ModelInfo[],
): void {
  const bar = el("#split-bar");
  const on = Boolean(compare && compare !== model);
  bar.hidden = !on;
  if (!on) return;
  el("#split-left-label").textContent = modelLabel(models, model);
  el("#split-right-label").textContent = modelLabel(models, compare);
  bar.style.left = `${globe.split * 100}%`;
}

function modelLabel(models: ModelInfo[], id: string): string {
  return models.find((m) => m.id === id)?.label ?? id;
}

function richestSample(samples: Sample[]): Sample {
  return samples.reduce((best, s) => (s.n_gt_px > best.n_gt_px ? s : best));
}

function uniqueTiles(samples: Sample[]): { xi: number; yi: number }[] {
  const seen = new Map<string, { xi: number; yi: number }>();
  for (const s of samples) seen.set(`${s.xi},${s.yi}`, { xi: s.xi, yi: s.yi });
  return [...seen.values()].sort((a, b) => a.yi - b.yi || a.xi - b.xi);
}

function sampleOnTile(
  samples: Sample[], xi: number, yi: number, dt: string,
): Sample | undefined {
  const onTile = samples.filter((s) => s.xi === xi && s.yi === yi);
  if (onTile.length === 0) return undefined;
  const exact = onTile.find((s) => s.dt === dt);
  if (exact) return exact;
  const t = Date.parse(dt);
  return onTile.reduce((best, s) =>
    Math.abs(Date.parse(s.dt) - t) < Math.abs(Date.parse(best.dt) - t) ? s : best,
  );
}

function opacity(): number {
  return Number((el("#opacity") as HTMLInputElement).value);
}

function fillSelect(sel: HTMLSelectElement, items: [string, string][]): void {
  sel.innerHTML = items.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
}

function el(sel: string): HTMLElement {
  const node = document.querySelector<HTMLElement>(sel);
  if (!node) throw new Error(`missing ${sel}`);
  return node;
}

void main();
