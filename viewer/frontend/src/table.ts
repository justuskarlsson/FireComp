import type { EventRow } from "./api";

const COLS = ["start", "region", "id", "n_samples", "n_gt_px", "iou", "precision", "recall"] as const;
type Col = (typeof COLS)[number];

export class EventTable {
  private body: HTMLElement;
  private all: EventRow[] = [];
  private filtered: EventRow[] = [];
  private sorted: EventRow[] = [];
  private model = "";
  private sortCol: Col = "n_gt_px";
  private sortDir: 1 | -1 = -1;
  private query = "";
  private region = "";
  private selected = "";
  private onSelect: ((eventId: string) => void) | null = null;
  private rowH = 28;

  constructor(body: HTMLElement, thead: HTMLElement) {
    this.body = body;
    thead.querySelectorAll<HTMLElement>("th[data-sort]").forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.dataset.sort as Col;
        if (this.sortCol === col) this.sortDir = this.sortDir === 1 ? -1 : 1;
        else {
          this.sortCol = col;
          this.sortDir = col === "start" ? 1 : -1;
        }
        thead.querySelectorAll("th").forEach((el) => {
          el.classList.remove("sort-asc", "sort-desc");
        });
        th.classList.add(this.sortDir === 1 ? "sort-asc" : "sort-desc");
        this.sort();
        this.render();
      });
    });
    thead.querySelector<HTMLElement>(`th[data-sort="${this.sortCol}"]`)
      ?.classList.add("sort-desc");
    this.body.addEventListener("scroll", () => this.render());
  }

  setOnSelect(fn: (eventId: string) => void): void {
    this.onSelect = fn;
  }

  setEvents(events: EventRow[], model: string): void {
    this.all = events;
    this.model = model;
    this.apply();
  }

  setModel(model: string): void {
    this.model = model;
    this.apply();
  }

  setQuery(q: string): void {
    this.query = q.trim().toLowerCase();
    this.apply();
  }

  setRegion(region: string): void {
    this.region = region;
    this.apply();
  }

  setSelected(id: string): void {
    this.selected = id;
    this.body.querySelectorAll(".row").forEach((el) => {
      el.classList.toggle("selected", el.getAttribute("data-id") === id);
    });
  }

  get filteredCount(): number {
    return this.filtered.length;
  }

  private apply(): void {
    const q = this.query;
    this.filtered = this.all.filter((e) => {
      if (this.region && e.region !== this.region) return false;
      if (!q) return true;
      return (
        e.start.includes(q) ||
        e.end.includes(q) ||
        String(e.id).includes(q) ||
        e.region.toLowerCase().includes(q) ||
        e.country.toLowerCase().includes(q)
      );
    });
    this.sort();
    this.render();
  }

  private sort(): void {
    const dir = this.sortDir;
    const col = this.sortCol;
    this.sorted = [...this.filtered].sort((a, b) => {
      const va = this.value(a, col);
      const vb = this.value(b, col);
      if (va < vb) return -1 * dir;
      if (va > vb) return 1 * dir;
      return 0;
    });
  }

  private value(e: EventRow, col: Col): string | number {
    if (col === "iou" || col === "precision" || col === "recall") {
      return e.mean_metrics[this.model]?.[col] ?? -1;
    }
    if (col === "n_gt_px") return e.n_gt_px;
    if (col === "n_samples") return e.n_samples;
    if (col === "id") return e.id;
    if (col === "region") return e.region;
    return e.start;
  }

  private render(): void {
    const n = this.sorted.length;
    const h = this.rowH;
    const view = this.body.clientHeight || 400;
    const start = Math.max(0, Math.floor(this.body.scrollTop / h) - 10);
    const count = Math.ceil(view / h) + 20;
    const end = Math.min(n, start + count);
    const slice = this.sorted.slice(start, end);
    const rows = slice.map((e) => {
      const m = e.mean_metrics[this.model];
      const id = String(e.id);
      const sel = id === this.selected ? " selected" : "";
      return `<div class="row${sel}" data-id="${id}" style="height:${h}px">
        <span>${e.start.slice(0, 10)}</span>
        <span class="muted" title="${e.region}">${shortRegion(e.region)}</span>
        <span>${e.id}</span>
        <span>${e.n_samples}</span>
        <span title="positive GT pixels">${e.n_gt_px}</span>
        <span>${fmt(m?.iou)}</span>
        <span>${fmt(m?.precision)}</span>
        <span>${fmt(m?.recall)}</span>
      </div>`;
    }).join("");
    this.body.innerHTML = n === 0
      ? `<div class="row muted">no events</div>`
      : `<div style="height:${start * h}px"></div>${rows}<div style="height:${(n - end) * h}px"></div>`;
    this.body.querySelectorAll<HTMLElement>(".row[data-id]").forEach((el) => {
      el.addEventListener("click", () => this.onSelect?.(el.dataset.id!));
    });
  }
}

function fmt(v: number | undefined): string {
  return v === undefined || v < 0 ? "—" : v.toFixed(2);
}

function shortRegion(name: string): string {
  const map: Record<string, string> = {
    "Western Europe": "W.Eur",
    "Eastern Europe": "E.Eur",
    "North Asia": "N.Asia",
    "South Asia": "S.Asia",
    "North NA": "N.NA",
    "Central NA": "C.NA",
    "South America": "S.Am",
  };
  return map[name] ?? name;
}
