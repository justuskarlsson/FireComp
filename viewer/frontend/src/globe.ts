import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { layerUrl, type EventRow, type Sample } from "./api";

const ESRI =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export class Globe {
  viewer: Cesium.Viewer;
  private points: Cesium.PointPrimitiveCollection;
  private overlay: Cesium.ImageryLayer | null = null;
  private compareOverlay: Cesium.ImageryLayer | null = null;
  private overlayGen = 0;
  private outline: Cesium.Entity | null = null;
  split = 0.5;
  private onPick: ((eventId: string) => void) | null = null;
  private idToPrimitive = new Map<string, Cesium.PointPrimitive>();
  private baseSize = new Map<string, number>();
  padding = 16;

  constructor(container: HTMLElement) {
    this.viewer = new Cesium.Viewer(container, {
      animation: false,
      timeline: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      baseLayerPicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      baseLayer: new Cesium.ImageryLayer(
        new Cesium.UrlTemplateImageryProvider({ url: ESRI, maximumLevel: 17 }),
      ),
    });
    this.viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#0b1d2a");
    this.points = this.viewer.scene.primitives.add(
      new Cesium.PointPrimitiveCollection(),
    );

    const handler = new Cesium.ScreenSpaceEventHandler(this.viewer.scene.canvas);
    handler.setInputAction((click: { position: Cesium.Cartesian2 }) => {
      const picked = this.viewer.scene.pick(click.position);
      const id = pickId(picked);
      if (id && this.onPick) this.onPick(id);
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  }

  setPickHandler(fn: (eventId: string) => void): void {
    this.onPick = fn;
  }

  setPoints(events: EventRow[], model: string): void {
    this.points.removeAll();
    this.idToPrimitive.clear();
    this.baseSize.clear();
    for (const e of events) {
      const id = String(e.id);
      const iou = e.mean_metrics[model]?.iou ?? e.metrics[model]?.iou ?? 0;
      const size = eventSize(e.n_samples);
      const p = this.points.add({
        id,
        position: Cesium.Cartesian3.fromDegrees(e.lon, e.lat),
        color: iouColor(iou),
        pixelSize: size,
        outlineColor: Cesium.Color.BLACK.withAlpha(0.5),
        outlineWidth: 1,
      });
      this.idToPrimitive.set(id, p);
      this.baseSize.set(id, size);
    }
  }

  highlight(id: string | null): void {
    this.idToPrimitive.forEach((p, pid) => {
      const base = this.baseSize.get(pid) ?? 7;
      p.pixelSize = pid === id ? base + 8 : base;
    });
  }

  async showOverlay(
    sample: Sample,
    layer: string,
    model: string,
    opacity: number,
    compare?: string,
  ): Promise<void> {
    const gen = ++this.overlayGen;
    this.clearOverlay();
    const rect = overlayRect(sample, this.padding);
    const useSplit = Boolean(compare && compare !== model);
    const urls = [layerUrl(sample.id, layer, model)];
    if (useSplit) urls.push(layerUrl(sample.id, layer, compare!));
    const providers = await Promise.all(
      urls.map((u) => Cesium.SingleTileImageryProvider.fromUrl(u, { rectangle: rect })),
    );
    if (gen !== this.overlayGen) return;
    this.overlay = this.viewer.imageryLayers.addImageryProvider(providers[0]);
    this.overlay.alpha = opacity;
    if (useSplit && providers[1]) {
      this.compareOverlay = this.viewer.imageryLayers.addImageryProvider(providers[1]);
      this.compareOverlay.alpha = opacity;
      this.overlay.splitDirection = Cesium.SplitDirection.LEFT;
      this.compareOverlay.splitDirection = Cesium.SplitDirection.RIGHT;
      this.viewer.scene.splitPosition = this.split;
    }
    this.outline = this.viewer.entities.add({
      rectangle: {
        coordinates: rect,
        fill: false,
        outline: true,
        outlineColor: Cesium.Color.WHITE.withAlpha(0.85),
        outlineWidth: 2,
      },
    });
  }

  setSplit(t: number): void {
    this.split = Math.min(0.92, Math.max(0.08, t));
    this.viewer.scene.splitPosition = this.split;
  }

  setOpacity(alpha: number): void {
    if (this.overlay) this.overlay.alpha = alpha;
    if (this.compareOverlay) this.compareOverlay.alpha = alpha;
  }

  clearOverlay(): void {
    if (this.overlay) {
      this.overlay.splitDirection = Cesium.SplitDirection.NONE;
      this.viewer.imageryLayers.remove(this.overlay, true);
      this.overlay = null;
    }
    if (this.compareOverlay) {
      this.compareOverlay.splitDirection = Cesium.SplitDirection.NONE;
      this.viewer.imageryLayers.remove(this.compareOverlay, true);
      this.compareOverlay = null;
    }
    if (this.outline) {
      this.viewer.entities.remove(this.outline);
      this.outline = null;
    }
  }

  flyToSample(sample: Sample): void {
    const [west, south, east, north] = sample.bbox;
    const pad = 0.4 * Math.max(east - west, north - south, 0.05);
    const rect = Cesium.Rectangle.fromDegrees(
      west - pad, south - pad, east + pad, north + pad,
    );
    this.viewer.camera.flyTo({ destination: rect, duration: 1.1 });
  }
}

/** Crop the 16px training border out of the georeferenced overlay. */
function overlayRect(sample: Sample, padding: number): Cesium.Rectangle {
  const [west, south, east, north] = sample.bbox;
  const img = sample.img_size || 256;
  const lonPad = (east - west) * padding / img;
  const latPad = (north - south) * padding / img;
  return Cesium.Rectangle.fromDegrees(
    west + lonPad, south + latPad, east - lonPad, north - latPad,
  );
}

function eventSize(nSamples: number): number {
  return 6 + Math.min(14, 3 * Math.sqrt(Math.max(nSamples, 1)));
}

function pickId(picked: unknown): string | null {
  if (!Cesium.defined(picked) || picked === null || typeof picked !== "object") {
    return null;
  }
  const obj = picked as { id?: unknown };
  if (typeof obj.id === "string") return obj.id;
  if (typeof obj.id === "number") return String(obj.id);
  if (obj.id && typeof obj.id === "object" && "id" in obj.id) {
    const inner = (obj.id as { id?: unknown }).id;
    if (typeof inner === "string") return inner;
    if (typeof inner === "number") return String(inner);
  }
  return null;
}

function iouColor(iou: number): Cesium.Color {
  const t = Math.min(Math.max(iou / 0.4, 0), 1);
  return Cesium.Color.fromHsl(t * 0.33, 0.85, 0.48, 0.95);
}
