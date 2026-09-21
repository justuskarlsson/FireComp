"""Matplotlib overlay PNGs for Cesium. No PyTorch."""

from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

mpl.use("Agg")

# Same border as NextDayConfig.padding / compute_loss_mask — never trained.
PADDING = 16
# nf_equiv: unburned iff accum_t_min < -0.1 (matches dl_2d / Metrics).
BURNED_THR = -0.1

TP_C = (0.15, 0.65, 0.15, 1.0)
FP_C = (0.85, 0.15, 0.15, 1.0)
FN_C = (0.15, 0.35, 0.85, 1.0)
FIRE_C = (0.85, 0.15, 0.05, 0.85)
CUR_C = (1.0, 0.55, 0.0, 0.85)

LAYER_INFO = {
    "accum_t":       {"label": "Fire history (accum_t)", "kind": "continuous"},
    "cur_mask":      {"label": "Current fire", "kind": "mask"},
    "gt_newfire":    {"label": "New-fire truth", "kind": "mask"},
    "prob":          {"label": "P(new fire)", "kind": "continuous"},
    "confusion":     {"label": "TP / FP / FN", "kind": "categorical"},
    "vpd":           {"label": "VPD", "kind": "continuous"},
    "soil_moisture": {"label": "Soil moisture", "kind": "continuous"},
    "wind":          {"label": "Wind magnitude", "kind": "continuous"},
    "gfs_temp":      {"label": "GFS temp max", "kind": "continuous"},
    "gfs_rh":        {"label": "GFS RH min", "kind": "continuous"},
    "gfs_precip":    {"label": "GFS precip", "kind": "continuous"},
}

WEATHER_CH = {"vpd": 0, "soil_moisture": 1, "wind": 3}
GFS_CH = {"gfs_temp": 0, "gfs_rh": 1, "gfs_precip": 4}


class SampleArrays:
    def __init__(self, path: Path):
        z = np.load(path)
        self.accum_t = z["accum_t"].astype(np.float32)
        self.cur_mask = z["cur_mask"]
        self.next_mask = z["next_mask"]
        self.loss_mask = z["loss_mask"].astype(np.float32)
        self.weather = z["weather"].astype(np.float32)
        self.gfs = z["gfs"].astype(np.float32)

    @property
    def burned(self) -> np.ndarray:
        return self.accum_t >= BURNED_THR

    @property
    def nf_valid(self) -> np.ndarray:
        return (~self.burned) & (self.loss_mask > 0.5)


def load_pred(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def render_layer(name: str, arrays: SampleArrays,
                 pred: np.ndarray | None, threshold: float,
                 size: int = 512) -> bytes:
    if name not in LAYER_INFO:
        raise KeyError(name)
    rgba = _layer_rgba(name, arrays, pred, threshold)
    if PADDING > 0:
        rgba = rgba[PADDING:-PADDING, PADDING:-PADDING]
    return _rgba_to_png(rgba, size)


def render_legend(name: str) -> bytes:
    fig = Figure(figsize=(2.4, 0.28), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.35, 0.84, 0.4])
    if name == "confusion":
        ax.set_axis_off()
        fig.clear()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.text(0.15, 0.5, "TP", color=TP_C[:3], ha="center", va="center",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.5, "FP", color=FP_C[:3], ha="center", va="center",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(0.85, 0.5, "FN", color=FN_C[:3], ha="center", va="center",
                fontsize=9, fontweight="bold", transform=ax.transAxes)
    elif name in ("cur_mask", "gt_newfire"):
        ax.imshow(np.linspace(0, 1, 256).reshape(1, -1),
                  cmap=mpl.colors.ListedColormap(["#00000000", FIRE_C]),
                  aspect="auto")
        ax.set_axis_off()
    elif name == "prob":
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([0, 127, 255])
        ax.set_xticklabels(["0", "thresh", "1"], fontsize=7)
    else:
        cmap = "YlOrRd" if name == "accum_t" else "viridis"
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(grad, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([0, 255])
        ax.set_xticklabels(["low", "high"], fontsize=7)
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    return buf.getvalue()


def _layer_rgba(name: str, a: SampleArrays,
                pred: np.ndarray | None, threshold: float) -> np.ndarray:
    h, w = a.accum_t.shape
    out = np.zeros((h, w, 4), dtype=np.float32)

    if name == "accum_t":
        cmap = mpl.colormaps["YlOrRd"]
        vals = a.accum_t.copy()
        show = a.burned
        finite = vals[show]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                hi = lo + 1e-6
            norm = np.clip((vals - lo) / (hi - lo), 0, 1)
            out[show] = cmap(norm[show])
            out[show, 3] = 0.85

    elif name == "cur_mask":
        out[a.cur_mask > 0] = CUR_C

    elif name == "gt_newfire":
        show = (a.next_mask > 0) & (~a.burned)
        out[show] = FIRE_C

    elif name == "prob":
        if pred is None:
            raise ValueError("prob layer needs a prediction")
        cmap = mpl.colormaps["viridis"]
        show = ~a.burned
        p = _threshold_norm(np.clip(pred, 0, 1), threshold)
        out[show] = cmap(p[show])
        out[show, 3] = np.clip(0.15 + 0.85 * p[show], 0, 1)

    elif name == "confusion":
        if pred is None:
            raise ValueError("confusion layer needs a prediction")
        # Same rule as Metrics.pixel_confusion(pred, next_mask, nf_valid, t)
        pred_bin = pred > threshold
        gt_bin = a.next_mask > 0.5
        valid = a.nf_valid
        out[a.burned | (a.loss_mask < 0.5)] = (0.81, 0.81, 0.81, 0.55)
        out[pred_bin & gt_bin & valid] = TP_C
        out[pred_bin & ~gt_bin & valid] = FP_C
        out[~pred_bin & gt_bin & valid] = FN_C

    elif name in WEATHER_CH:
        ch = a.weather[WEATHER_CH[name]]
        _fill_continuous(out, ch, "YlOrBr")

    elif name in GFS_CH:
        ch = a.gfs[GFS_CH[name]]
        cmap = "Blues" if name == "gfs_precip" else "coolwarm"
        _fill_continuous(out, ch, cmap)

    return out


def _threshold_norm(p: np.ndarray, t: float) -> np.ndarray:
    """Put the model's decision threshold at colormap midpoint 0.5.

    [0, t] → [0, 0.5], [t, 1] → [0.5, 1].  Lets models with different
    thresholds (0.08 vs 0.38) be compared on the same colour scale.
    """
    t = float(t)
    out = np.empty_like(p)
    below = p < t
    if t <= 0:
        out[:] = np.clip(p, 0, 1)
        return out
    out[below] = 0.5 * p[below] / t
    if t >= 1:
        out[~below] = 0.5
    else:
        out[~below] = 0.5 + 0.5 * (p[~below] - t) / (1.0 - t)
    return np.clip(out, 0, 1)


def _fill_continuous(out: np.ndarray, ch: np.ndarray, cmap_name: str) -> None:
    """Upsample a native-res channel to the overlay grid and color it."""
    h, w = out.shape[:2]
    up = _upsample_nearest(ch, h, w)
    finite = up[np.isfinite(up)]
    if finite.size == 0:
        return
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((up - lo) / (hi - lo), 0, 1)
    cmap = mpl.colormaps[cmap_name]
    out[:] = cmap(norm)
    out[..., 3] = 0.75


def _upsample_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    yh = arr.shape[0] / h
    xh = arr.shape[1] / w
    rows = np.minimum((np.arange(h) * yh).astype(int), arr.shape[0] - 1)
    cols = np.minimum((np.arange(w) * xh).astype(int), arr.shape[1] - 1)
    return arr[rows[:, None], cols[None, :]]


def _rgba_to_png(rgba: np.ndarray, size: int) -> bytes:
    fig = Figure(figsize=(size / 100, size / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(rgba, origin="upper", interpolation="nearest")
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=100)
    return buf.getvalue()
