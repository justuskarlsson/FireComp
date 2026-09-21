"""
firecomp/viewer/export.py — Offline test-set export for the research viewer.

Writes visualization arrays + per-sample nf_equiv metrics.  The web app
never runs PyTorch.

Usage:
    python -m firecomp.viewer.export inputs
    python -m firecomp.viewer.export preds --model unet
    python -m firecomp.viewer.export collect
    python -m firecomp.viewer.export smoke --max-samples 8 --batch-size 8
"""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from firecomp.dsrc.vnp14 import DEG_CELL_SIZE
from firecomp.next_day.config import NextDayConfig
from firecomp.next_day.dataset import (
    NextDayDataset,
    Sample,
    _rel_t,
    _union_accum_cur_mask,
    compute_loss_mask,
)
from firecomp.core.regions import Regions


KM2_PER_PIXEL = (0.375) ** 2  # 375 m VIIRS cell
DEFAULT_MODELS = Path(__file__).parent / "models.json"
PADDING = 16


def main():
    args = _parse_args()
    spec = _load_spec(args.models)
    out = Path(args.out_dir or spec.get("out_dir", "data/viewer"))
    dataset_dir = args.dataset_dir or spec.get("dataset_dir", "data/next_day_v3")
    fire_type = spec.get("fire_type", "vegetation")
    max_samples = args.max_samples
    batch_size = args.batch_size

    if args.command == "smoke" and max_samples == 0:
        max_samples = 8

    if args.command == "inputs":
        ExportInputs.run(out, dataset_dir, fire_type, max_samples)
    elif args.command == "preds":
        model = _find_model(spec, args.model)
        ExportPreds.run(out, dataset_dir, fire_type, model, max_samples,
                        batch_size)
    elif args.command == "collect":
        Collect.run(out, spec)
    elif args.command == "smoke":
        ExportSmoke.run(out, dataset_dir, fire_type, spec, max_samples,
                        batch_size)
    else:
        raise SystemExit(f"unknown command {args.command!r}")


def sample_id(s: Sample) -> str:
    return f"{s.fire_id}_{s.xi}_{s.yi}_{s.dt}"


def sample_bbox(lon: float, lat: float, img_size: int = 256) -> list[float]:
    """[west, south, east, north] in degrees for a 256×256 patch."""
    half = (img_size / 2) * DEG_CELL_SIZE
    return [lon - half, lat - half, lon + half, lat + half]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


class ExportInputs:
    @staticmethod
    def run(out: Path, dataset_dir: str, fire_type: str,
            max_samples: int) -> None:
        cfg = NextDayConfig(
            dataset_dir=dataset_dir, fire_type=fire_type, device="cpu",
        )
        ds = NextDayDataset(cfg)
        samples = list(ds.test_samples)
        if max_samples > 0:
            samples = samples[:max_samples]
        print(f"export inputs: {len(samples)} test samples → {out}")

        extra = _load_extra_meta(Path(dataset_dir))
        last_dt = _last_dt_by_fire(samples)
        regions = Regions()
        countries = CountryLookup.try_load()

        sample_dir = out / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = []

        for i, s in enumerate(samples):
            sid = sample_id(s)
            raw = ds._store.load(s)
            _union_accum_cur_mask(raw)
            accum = _rel_t(raw["accum_t_min"] if "accum_t_min" in raw
                           else raw["accum_t"])
            cur = (raw["cur_mask"][0] > 0.5).astype(np.uint8)
            nxt = (raw["next_mask"][0] > 0.5).astype(np.uint8)
            is_last = s.dt == last_dt[s.fire_id]
            loss = compute_loss_mask(raw, padding=PADDING, is_last_day=is_last)
            weather = raw["weather"].astype(np.float16)
            gfs = raw["gfs"].astype(np.float16)

            np.savez_compressed(
                sample_dir / f"{sid}.npz",
                accum_t=accum[0].astype(np.float16),
                cur_mask=cur,
                next_mask=nxt,
                loss_mask=loss.astype(np.uint8),
                weather=weather,
                gfs=gfs,
            )

            burned = accum[0] >= -0.5
            n_gt = int((nxt & ~burned & (loss > 0.5)).sum())
            ext = extra.get((s.fire_id, s.xi, s.yi, s.dt), {})
            bbox = sample_bbox(s.lon, s.lat, s.img_size)
            country = countries.lookup(s.lon, s.lat) if countries else ""
            meta_rows.append({
                "id": sid,
                "event_id": s.fire_id,
                "xi": s.xi,
                "yi": s.yi,
                "dt": s.dt,
                "day_of_fire": ext.get("day_of_fire", -1),
                "lon": s.lon,
                "lat": s.lat,
                "bbox": bbox,
                "img_size": s.img_size,
                "region_id": s.region_id,
                "region": regions.id_to_name(s.region_id),
                "country": country,
                "num_fire": s.num_fire,
                "n_gt_px": n_gt,
                "area_km2": round(n_gt * KM2_PER_PIXEL, 4),
                "metrics": {},
            })
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(samples)}")

        meta_path = out / "samples_meta.json"
        meta_path.write_text(json.dumps(meta_rows))
        print(f"wrote {meta_path} ({len(meta_rows)} samples)")


# ---------------------------------------------------------------------------
# preds
# ---------------------------------------------------------------------------


class ExportPreds:
    @staticmethod
    def run(out: Path, dataset_dir: str, fire_type: str,
            model: dict, max_samples: int, batch_size: int) -> None:
        kind = model["kind"]
        if kind == "checkpoint":
            ExportPreds._run_checkpoint(
                out, dataset_dir, model, max_samples, batch_size)
        elif kind == "morphological":
            ExportPreds._run_morph(
                out, dataset_dir, fire_type, model, max_samples, batch_size)
        else:
            raise ValueError(f"unknown model kind {kind!r}")

    @staticmethod
    def _run_checkpoint(out: Path, dataset_dir: str, model: dict,
                        max_samples: int, batch_size: int) -> None:
        import torch
        from firecomp.core.checkpoint import BestCheckpoint
        from firecomp.core.metrics import Metrics
        from firecomp.core.torch_utils import get_device, setup_precision
        from firecomp.models.segmentation_models import model_factory

        device = get_device("cuda")
        setup_precision(device)
        ckpt = BestCheckpoint(Path(model["checkpoint"]).parent).load(
            map_location=device)
        cfg = replace(ckpt["cfg"], dataset_dir=dataset_dir, device="cuda",
                      batch_size=batch_size)
        threshold = float(model["threshold"])
        ds = NextDayDataset(cfg)
        samples = ds.test_samples
        n_take = len(samples) if max_samples <= 0 else min(max_samples, len(samples))

        net = model_factory[cfg.model_type](
            in_channels=ds.num_channels,
            out_channels=1,
            encoder_name=cfg.encoder_name,
        ).to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()

        pred_dir = out / "preds" / model["id"]
        pred_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        seen = 0
        accum_ch = None

        print(f"export preds [{model['id']}]: {n_take} samples  "
              f"threshold={threshold}  device={device}  batch_size={batch_size}")
        with torch.no_grad():
            for batch in ds.test(batch_size=batch_size):
                if accum_ch is None:
                    accum_ch = batch.channel_names.index("accum_t_min")
                pred = torch.sigmoid(net(batch.x))
                accum = batch.x[:, accum_ch:accum_ch + 1]
                nf_mask = batch.loss_mask * (accum < -0.1).float()
                m = Metrics(pred, batch.y, nf_mask, threshold=threshold)
                per = m.per_sample
                pred_np = pred.cpu().numpy()
                nf_np = nf_mask.cpu().numpy()
                for i, sm in enumerate(per):
                    if seen >= n_take:
                        break
                    s = batch.samples[i]
                    sid = sample_id(s)
                    np.save(pred_dir / f"{sid}.npy",
                            pred_np[i, 0].astype(np.float16))
                    rows.append(_metric_row(
                        sid, sm, threshold, pred_np[i, 0], nf_np[i, 0]))
                    seen += 1
                if seen >= n_take:
                    break
                print(f"  {seen}/{n_take}")

        del net
        torch.cuda.empty_cache()
        _write_metrics(out, model["id"], rows, threshold)
        print(f"wrote {len(rows)} predictions for {model['id']}")

    @staticmethod
    def _run_morph(out: Path, dataset_dir: str, fire_type: str,
                   model: dict, max_samples: int, batch_size: int) -> None:
        from firecomp.core.metrics import Metrics
        from firecomp.next_day.implementations.morphological import (
            _dilate_batch, _disk_structuring_element,
        )

        cfg = NextDayConfig(
            dataset_dir=dataset_dir, fire_type=fire_type, device="cpu",
            batch_size=batch_size,
        )
        ds = NextDayDataset(cfg)
        threshold = float(model["threshold"])
        radius = int(model["radius"])
        struct = _disk_structuring_element(radius)
        n_take = len(ds.test_samples) if max_samples <= 0 else min(
            max_samples, len(ds.test_samples))

        pred_dir = out / "preds" / model["id"]
        pred_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        seen = 0
        cur_ch = None
        accum_ch = None

        print(f"export preds [{model['id']}]: {n_take} samples  "
              f"radius={radius}  batch_size={batch_size}")
        for batch in ds.test(batch_size=batch_size):
            if cur_ch is None:
                cur_ch = batch.channel_names.index("cur_mask")
                accum_ch = batch.channel_names.index("accum_t_min")
            cur = batch.x[:, cur_ch:cur_ch + 1]
            pred = _dilate_batch(cur, struct, is_new_fires=False)
            accum = batch.x[:, accum_ch:accum_ch + 1]
            nf_mask = batch.loss_mask * (accum < -0.1).float()
            m = Metrics(pred, batch.y, nf_mask, threshold=threshold)
            per = m.per_sample
            pred_np = pred.cpu().numpy()
            nf_np = nf_mask.cpu().numpy()
            for i, sm in enumerate(per):
                if seen >= n_take:
                    break
                s = batch.samples[i]
                sid = sample_id(s)
                np.save(pred_dir / f"{sid}.npy",
                        pred_np[i, 0].astype(np.float16))
                rows.append(_metric_row(
                    sid, sm, threshold, pred_np[i, 0], nf_np[i, 0]))
                seen += 1
            if seen >= n_take:
                break
            print(f"  {seen}/{n_take}")

        _write_metrics(out, model["id"], rows, threshold)
        print(f"wrote {len(rows)} predictions for {model['id']}")


# ---------------------------------------------------------------------------
# smoke — inputs + every model + collect
# ---------------------------------------------------------------------------


class ExportSmoke:
    @staticmethod
    def run(out: Path, dataset_dir: str, fire_type: str, spec: dict,
            max_samples: int, batch_size: int) -> None:
        print(f"smoke: max_samples={max_samples}  batch_size={batch_size}")
        ExportInputs.run(out, dataset_dir, fire_type, max_samples)
        for model in spec["models"]:
            ExportPreds.run(out, dataset_dir, fire_type, model,
                            max_samples, batch_size)
        Collect.run(out, spec)
        print("smoke done")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


class Collect:
    @staticmethod
    def run(out: Path, spec: dict) -> None:
        meta_path = out / "samples_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing — run `export inputs` first")
        samples = json.loads(meta_path.read_text())
        by_id = {s["id"]: s for s in samples}

        model_summaries = []
        for model in spec["models"]:
            mid = model["id"]
            mpath = out / "metrics" / f"{mid}.json"
            if not mpath.exists():
                print(f"  skip {mid}: no metrics file")
                continue
            payload = json.loads(mpath.read_text())
            n = 0
            for row in payload["samples"]:
                s = by_id.get(row["id"])
                if s is None:
                    continue
                s["metrics"][mid] = {
                    k: row[k] for k in
                    ("iou", "precision", "recall", "f1", "brier",
                     "mean_prob", "tp", "fp", "fn")
                    if k in row
                }
                n += 1
            model_summaries.append({
                "id": mid,
                "label": model["label"],
                "kind": model["kind"],
                "threshold": payload.get("threshold", model["threshold"]),
                "n_samples": n,
            })
            print(f"  merged {mid}: {n} samples")

        events = Collect._build_events(samples, spec["models"])
        index = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "split": spec.get("split", "test"),
            "fire_type": spec.get("fire_type", "vegetation"),
            "target": spec.get("target", "nf_equiv"),
            "models": model_summaries,
            "n_samples": len(samples),
            "n_events": len(events),
            "events": events,
            "samples": samples,
        }
        index_path = out / "index.json"
        index_path.write_text(json.dumps(index))
        print(f"wrote {index_path}  "
              f"({len(samples)} samples, {len(events)} events)")

    @staticmethod
    def _build_events(samples: list[dict], models: list[dict]) -> list[dict]:
        by_fire: dict[int, list[dict]] = defaultdict(list)
        for s in samples:
            by_fire[s["event_id"]].append(s)

        events = []
        for fid, rows in by_fire.items():
            rows = sorted(rows, key=lambda r: r["dt"])
            dts = [r["dt"] for r in rows]
            start, end = dts[0], dts[-1]
            duration = (
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).days + 1
            west = min(r["bbox"][0] for r in rows)
            south = min(r["bbox"][1] for r in rows)
            east = max(r["bbox"][2] for r in rows)
            north = max(r["bbox"][3] for r in rows)
            metrics = {}
            for m in models:
                mid = m["id"]
                ious = [r["metrics"][mid]["iou"]
                        for r in rows if mid in r["metrics"]]
                if ious:
                    metrics[mid] = {"iou": round(sum(ious) / len(ious), 4)}
            events.append({
                "id": fid,
                "region": rows[0]["region"],
                "country": rows[0]["country"],
                "lon": sum(r["lon"] for r in rows) / len(rows),
                "lat": sum(r["lat"] for r in rows) / len(rows),
                "bbox": [west, south, east, north],
                "start": start,
                "end": end,
                "duration_days": duration,
                "n_samples": len(rows),
                "area_km2": round(max(r["area_km2"] for r in rows), 4),
                "sample_ids": [r["id"] for r in rows],
                "metrics": metrics,
            })
        events.sort(key=lambda e: e["start"])
        return events


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _metric_row(sid: str, sm, threshold: float,
                pred_hw: np.ndarray, nf_hw: np.ndarray) -> dict:
    valid = nf_hw > 0.5
    mean_prob = float(pred_hw[valid].mean()) if valid.any() else 0.0
    return {
        "id": sid,
        "iou": round(sm.iou, 4),
        "precision": round(sm.precision, 4),
        "recall": round(sm.recall, 4),
        "f1": round(sm.f1, 4),
        "brier": round(sm.brier, 4),
        "tp": int(sm.tp),
        "fp": int(sm.fp),
        "fn": int(sm.fn),
        "mean_prob": round(mean_prob, 4),
        "threshold": threshold,
    }


def _write_metrics(out: Path, model_id: str, rows: list[dict],
                   threshold: float) -> None:
    d = out / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "threshold": threshold,
        "n_samples": len(rows),
        "samples": rows,
    }
    (d / f"{model_id}.json").write_text(json.dumps(payload))


def _load_spec(path: str | None) -> dict:
    p = Path(path) if path else DEFAULT_MODELS
    with open(p) as f:
        return json.load(f)


def _find_model(spec: dict, model_id: str) -> dict:
    for m in spec["models"]:
        if m["id"] == model_id:
            return m
    known = [m["id"] for m in spec["models"]]
    raise SystemExit(f"unknown model {model_id!r}. known: {known}")


def _load_extra_meta(dataset_dir: Path) -> dict:
    extra = {}
    for p in sorted(dataset_dir.glob("dataset_*.json")):
        for d in json.loads(p.read_text()):
            extra[(d["fire_id"], d["xi"], d["yi"], d["dt"])] = d
    return extra


def _last_dt_by_fire(samples: list[Sample]) -> dict[int, str]:
    last: dict[int, str] = {}
    for s in samples:
        prev = last.get(s.fire_id)
        if prev is None or s.dt > prev:
            last[s.fire_id] = s.dt
    return last


class CountryLookup:
    """Optional offline country name from data/countries/."""

    def __init__(self, raster, xmin, ymax, pix, code_to_name):
        self._raster = raster
        self._xmin = xmin
        self._ymax = ymax
        self._pix = pix
        self._code_to_name = code_to_name

    @staticmethod
    def try_load():
        raster_path = Path("data/countries/countries.tif")
        json_path = Path("data/countries/countries.json")
        if not raster_path.exists() or not json_path.exists():
            print("  country lookup skipped (no countries.tif)")
            return None
        from osgeo import gdal
        ds = gdal.Open(str(raster_path))
        raster = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        ds = None
        mapping = json.loads(json_path.read_text(encoding="utf-8"))
        names = {int(k): v for k, v in mapping["name"].items()}
        return CountryLookup(raster, gt[0], gt[3], gt[1], names)

    def lookup(self, lon: float, lat: float) -> str:
        col = int((lon - self._xmin) / self._pix)
        row = int((self._ymax - lat) / self._pix)
        col = min(max(col, 0), self._raster.shape[1] - 1)
        row = min(max(row, 0), self._raster.shape[0] - 1)
        code = int(self._raster[row, col])
        return self._code_to_name.get(code, "")


def _parse_args():
    p = argparse.ArgumentParser(
        prog="firecomp.viewer.export",
        description="Export next-day test predictions for the research viewer.",
    )
    p.add_argument("command", choices=["inputs", "preds", "collect", "smoke"])
    p.add_argument("--model", default="",
                   help="Model id from models.json (preds only).")
    p.add_argument("--models", default="",
                   help="Path to models.json (default: firecomp/viewer/models.json).")
    p.add_argument("--out-dir", default="",
                   help="Output directory (default: data/viewer).")
    p.add_argument("--dataset-dir", default="",
                   help="Override dataset directory.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap samples for a smoke test (0 = all; smoke defaults to 8).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Inference batch size (default 16; 80GB runs used 128).")
    args = p.parse_args()
    if args.command == "preds" and not args.model:
        p.error("preds requires --model")
    return args


if __name__ == "__main__":
    main()
