"""
core/plotting.py — Mid-level matplotlib helpers.

The matplotlib twin of `firecomp/agent/core.py::print_image`: same flexible
tensor handling (CHW/HWC inference, percentile stretch, NaN+nodata),
but rendered into a matplotlib axes instead of base64.

    - Free functions, not a framework.
    - `Style` is a class with class-attrs you can mutate to override defaults.
    - `Cmaps` consolidates the LinearSegmentedColormaps that were duplicated
      across viz/ files.
    - Cartopy-specific helpers (basemaps, mask_to_contour, setup_map_axes)
      stay in `firecomp/viz/common.py` — they're paper-figure specific.

Public API:
    imshow_tensor(ax, x, ...)            tensor -> ax.imshow with smart norm
    make_figure(nrows, ncols, ...)       plt.subplots, axes always 2D
    add_colorbar(im, ax, label, ...)     house-style colorbar
    save_figure(fig, path, ...)          house-style savefig
    tensor_grid(tensors, ...)            quick N-tensor inspection grid

    classification_overlay(pred, gt)     (H,W,4) RGBA: TP/FP/FN colours
    scalar_field_overlay(field, ...)     (H,W,4) RGBA from scalar field

    Style                                house style (mutate to override)
    Cmaps                                named LinearSegmentedColormaps
"""

__all__ = [
    "Style", "Cmaps",
    "GT_COLOR", "PRED_COLOR", "TP_COLOR", "FP_COLOR", "FN_COLOR",
    "imshow_tensor", "make_figure", "add_colorbar", "save_figure",
    "tensor_grid",
    "classification_overlay", "scalar_field_overlay",
    "add_satellite_basemap",
    "cartopy_heatmap",
]

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


# =============================================================================
# House style — mutate Style.* to override globally
# =============================================================================


class Style:
    """House style for paper figures. Mutate fields to override globally."""
    title_fontsize: int = 14
    title_pad: int = 4
    label_fontsize: int = 14
    suptitle_fontsize: int = 20
    monospace: str = "monospace"
    cbar_fraction: float = 0.046
    cbar_pad: float = 0.04
    cbar_shrink: float = 0.8
    save_dpi: int = 300
    save_bbox: str = "tight"
    save_facecolor: str = "white"
    grid_alpha: float = 0.3
    legend_framealpha: float = 0.85
    hspace: float = 0.35
    wspace: float = 0.15


# Standard fire-prediction colours (used in contours / legends / overlays)
GT_COLOR = "#2ca02c"      # green
PRED_COLOR = "#d62728"    # red
TP_COLOR = (50, 205, 50, 220)        # lime green
FP_COLOR = (255, 127, 80, 220)       # coral
FN_COLOR = (135, 206, 235, 220)      # sky blue


class Cmaps:
    """Named LinearSegmentedColormaps used across the project."""

    # transparent -> orange -> red — for fire prediction overlays
    fire = LinearSegmentedColormap.from_list(
        "fire", [(1, 1, 1, 0), (1, 0.6, 0, 0.7), (0.8, 0, 0, 0.9)]
    )

    # viridis-like, for fire spread time index (early -> late)
    fire_temporal = LinearSegmentedColormap.from_list(
        "fire_temporal", ["#440154", "#31688e", "#35b779", "#fde725"]
    )

    # red (early/dark) -> orange -> yellow (late) — for accum_t underlays
    fire_spread = LinearSegmentedColormap.from_list(
        "fire_spread",
        ["#8b0000", "#cc2200", "#e65c00", "#ff9900", "#ffcc00", "#ffee55"],
        N=256,
    )

    # light yellow -> green -> dark green — for canopy height
    canopy = LinearSegmentedColormap.from_list(
        "canopy", ["#f7fcb1", "#31a354", "#003300"]
    )


# =============================================================================
# Display: imshow_tensor — the matplotlib twin of print_image()
# =============================================================================


def imshow_tensor(
    ax,
    x,
    *,
    pct: float = 100.0,
    nodata: float | None = None,
    cmap: str | mpl.colors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    off: bool = True,
    extent: list[float] | None = None,
    origin: str = "upper",
    interpolation: str = "nearest",
    **imshow_kwargs,
):
    """Display a tensor / ndarray / PIL image on a matplotlib axes.

    Mirrors `firecomp/agent/core.py::print_image`:
        - moves torch tensors to numpy on cpu
        - infers layout: (H,W) | (H,W,C) | (C,H,W), C in {1,3,4}
        - C==1 -> squeezed to grayscale, uses `cmap`
        - C==3 -> RGB, no cmap
        - C==4 -> RGBA, no cmap
        - excludes NaN and `nodata` from min/max stats
        - `pct < 100` clips outliers via percentile stretch

    Args:
        ax: matplotlib Axes (or GeoAxes) to draw into.
        x: tensor, ndarray, or PIL Image.
        pct: percentile stretch. 100 = full min/max (default).
            97.5 = clip top/bottom 2.5% (good for noisy satellite data).
        nodata: pixel value treated as missing (excluded from stats,
            rendered as cmap "bad" colour or transparent in RGB).
            NaN is always nodata for floats.
        cmap: only used for grayscale / single-channel.
        vmin / vmax: explicit overrides; skip percentile if both given.
        title: optional title (uses Style.title_fontsize).
        off: ax.axis("off") after drawing (default True).
        extent / origin / interpolation: passed to imshow.
        **imshow_kwargs: forwarded to ax.imshow (e.g. transform=...).

    Returns:
        matplotlib.image.AxesImage (so caller can pass to add_colorbar).
    """
    arr = _to_numpy(x)
    arr = _infer_layout(arr)
    nd_mask = _build_nodata_mask(arr, nodata)

    if arr.ndim == 2:
        # Grayscale: let mpl scale, paint nodata as NaN -> cmap "bad" colour
        v_lo, v_hi = _resolve_range(arr, nd_mask, pct, vmin, vmax)
        if nd_mask is not None and nd_mask.any():
            arr = arr.astype(np.float32)
            arr[nd_mask] = np.nan
        im = ax.imshow(
            arr, cmap=cmap, vmin=v_lo, vmax=v_hi,
            extent=extent, origin=origin, interpolation=interpolation,
            **imshow_kwargs,
        )
    else:
        # RGB / RGBA: pre-normalise to uint8 (mpl wants uint8 or float [0,1])
        arr_u8 = _to_uint8_color(arr, nd_mask, pct, vmin, vmax)
        im = ax.imshow(
            arr_u8, extent=extent, origin=origin, interpolation=interpolation,
            **imshow_kwargs,
        )

    if title is not None:
        ax.set_title(title, fontsize=Style.title_fontsize, pad=Style.title_pad)
    if off:
        ax.axis("off")
    return im


# =============================================================================
# Display: figure construction, colorbars, saving
# =============================================================================


def make_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | None = None,
    cell: tuple[float, float] = (3.0, 3.0),
    height_ratios: list[float] | None = None,
    width_ratios: list[float] | None = None,
    hspace: float | None = None,
    wspace: float | None = None,
    projection=None,
    constrained: bool = False,
):
    """Like plt.subplots, but axes is always a 2D ndarray.

    Args:
        nrows / ncols: grid shape.
        figsize: explicit (w, h). If None, computed as cell * (ncols, nrows).
        cell: per-cell (w, h) inches. Default (3, 3) gives sane sizes.
        height_ratios / width_ratios: GridSpec ratios.
        hspace / wspace: GridSpec spacing (default from Style).
        projection: cartopy projection (or any subplot_kw["projection"]).
        constrained: use constrained_layout instead of tight_layout.

    Returns:
        (fig, axes_2d) — axes_2d.shape == (nrows, ncols) always.
    """
    if figsize is None:
        figsize = (cell[0] * ncols, cell[1] * nrows)

    subplot_kw = {"projection": projection} if projection is not None else None
    gridspec_kw = {
        "hspace": Style.hspace if hspace is None else hspace,
        "wspace": Style.wspace if wspace is None else wspace,
    }
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = width_ratios

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
        constrained_layout=constrained,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    return fig, axes


def add_colorbar(
    im,
    ax,
    label: str = "",
    *,
    fontsize: int | None = None,
    fraction: float | None = None,
    pad: float | None = None,
    shrink: float | None = None,
):
    """Attach a colorbar with house-style defaults. Pass kwargs to override."""
    cb = plt.colorbar(
        im,
        ax=ax,
        fraction=Style.cbar_fraction if fraction is None else fraction,
        pad=Style.cbar_pad if pad is None else pad,
        shrink=Style.cbar_shrink if shrink is None else shrink,
    )
    if label:
        cb.set_label(label, fontsize=Style.label_fontsize if fontsize is None else fontsize)
    return cb


def save_figure(
    fig,
    path: str,
    *,
    dpi: int | None = None,
    formats: tuple[str, ...] = ("png",),
    bbox: str | None = None,
    facecolor: str | None = None,
    close: bool = True,
    log: bool = True,
) -> list[str]:
    """Save figure(s) with house-style defaults.

    `path` is treated as a stem; `formats` controls extensions. If `path`
    already ends in one of the requested format extensions, it's stripped
    so we don't end up with `foo.png.png`.

    Returns the list of paths written.
    """
    path = str(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

    base, ext = os.path.splitext(path)
    if ext.lower().lstrip(".") in formats:
        path = base

    saved: list[str] = []
    for fmt in formats:
        out = f"{path}.{fmt}"
        fig.savefig(
            out,
            dpi=Style.save_dpi if dpi is None else dpi,
            bbox_inches=Style.save_bbox if bbox is None else bbox,
            facecolor=Style.save_facecolor if facecolor is None else facecolor,
        )
        saved.append(out)

    if log:
        print(f"saved: {', '.join(saved)}")
    if close:
        plt.close(fig)
    return saved


def tensor_grid(
    tensors,
    *,
    titles: list[str | None] | None = None,
    ncols: int = 4,
    cell: tuple[float, float] = (3.0, 3.0),
    cmap: str | mpl.colors.Colormap | None = None,
    pct: float = 100.0,
    nodata: float | None = None,
    suptitle: str | None = None,
):
    """Render a grid of tensors via `imshow_tensor`. Quick inspection helper.

    Args:
        tensors: list of tensors / ndarrays, OR dict {title: tensor}.
        titles: parallel labels (ignored if tensors is a dict).
        ncols: grid width; nrows is computed.
        cell: per-cell (w, h) inches.
        cmap / pct / nodata: passed to imshow_tensor.
        suptitle: figure-level title.

    Returns:
        (fig, axes_2d).
    """
    if isinstance(tensors, dict):
        items = list(tensors.values())
        titles = list(tensors.keys())
    else:
        items = list(tensors)
        if titles is None:
            titles = [None] * len(items)

    n = len(items)
    nrows = max(1, (n + ncols - 1) // ncols)
    # constrained_layout handles suptitle headroom automatically and avoids
    # the tight_layout warning when GridSpec params (h/wspace) are set.
    fig, axes = make_figure(nrows, ncols, cell=cell, constrained=True)

    for i, (x, t) in enumerate(zip(items, titles)):
        r, c = divmod(i, ncols)
        imshow_tensor(axes[r, c], x, title=t, cmap=cmap, pct=pct, nodata=nodata)

    # blank any leftover cells
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=Style.suptitle_fontsize, fontweight="bold")
    return fig, axes


# =============================================================================
# RGBA overlay primitives
# =============================================================================


def classification_overlay(
    pred_bin,
    gt_bin,
    *,
    mask=None,
    pad: int = 0,
    tp: tuple[int, int, int, int] = TP_COLOR,
    fp: tuple[int, int, int, int] = FP_COLOR,
    fn: tuple[int, int, int, int] = FN_COLOR,
) -> np.ndarray:
    """Encode TP / FP / FN as an (H, W, 4) uint8 RGBA overlay.

    Pixels outside `mask` (or inside the `pad` border) are transparent —
    they don't contribute to either class. TN pixels are also transparent
    (we only colour where there's signal).

    Args:
        pred_bin: (H, W) bool / 0-1 — thresholded predictions.
        gt_bin:   (H, W) bool / 0-1 — ground truth.
        mask: optional (H, W) bool — 1 = valid, 0 = exclude.
        pad: number of border pixels to exclude (e.g. 16 for the next_day
            convolution padding).
        tp / fp / fn: RGBA tuples (uint8 0-255).

    Returns:
        (H, W, 4) uint8 ndarray.
    """
    pred = np.asarray(pred_bin).astype(bool)
    gt = np.asarray(gt_bin).astype(bool)
    h, w = pred.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = np.ones((h, w), dtype=bool)
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    if pad > 0:
        valid[:pad] = False
        valid[-pad:] = False
        valid[:, :pad] = False
        valid[:, -pad:] = False

    pred = pred & valid
    gt = gt & valid

    rgba[pred & gt] = tp
    rgba[pred & ~gt] = fp
    rgba[~pred & gt] = fn
    return rgba


def scalar_field_overlay(
    field,
    *,
    cmap: str | mpl.colors.Colormap = "hot",
    alpha: float = 0.35,
    vmin: float | None = None,
    vmax: float | None = None,
    background: float | None = None,
    nan_transparent: bool = True,
) -> np.ndarray:
    """Render a scalar (H, W) field as an (H, W, 4) uint8 RGBA overlay.

    Background / NaN pixels render fully transparent. Other pixels are
    coloured by `cmap` at uniform `alpha`.

    Args:
        field: (H, W) numeric array.
        cmap: matplotlib colormap name or instance.
        alpha: opacity for non-background pixels (0-1).
        vmin / vmax: explicit range; auto from valid pixels if None.
        background: value to render as transparent (e.g. -1 for accum_t).
        nan_transparent: NaNs render transparent (default True).

    Returns:
        (H, W, 4) uint8 ndarray.
    """
    arr = np.asarray(field).astype(float)
    h, w = arr.shape

    transparent = np.zeros((h, w), dtype=bool)
    if background is not None:
        transparent |= (arr == background)
    if nan_transparent:
        transparent |= np.isnan(arr)

    valid = arr[~transparent]
    if valid.size == 0:
        return np.zeros((h, w, 4), dtype=np.uint8)

    lo = float(valid.min()) if vmin is None else vmin
    hi = float(valid.max()) if vmax is None else vmax
    if hi > lo:
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        norm = np.zeros_like(arr)

    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = cm(norm)  # (H, W, 4) float
    rgba = (colors * 255).astype(np.uint8)
    rgba[..., 3] = int(alpha * 255)
    rgba[transparent] = 0
    return rgba


# =============================================================================
# Internal helpers
# =============================================================================


def _to_numpy(x) -> np.ndarray:
    """Convert tensor / PIL / ndarray to ndarray (cpu, copy when from torch)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return np.array(x)
    return np.array(x)  # copies; we mutate downstream


def _infer_layout(arr: np.ndarray) -> np.ndarray:
    """Return arr in (H, W) or (H, W, C) form, inferring CHW vs HWC.

    C is 1, 3, or 4. (H, W, 1) is squeezed to (H, W).
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        pass  # already HWC
    elif arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    else:
        raise ValueError(
            f"Cannot infer layout for shape {arr.shape}: "
            "first or last dim must be 1, 3, or 4"
        )

    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def _build_nodata_mask(arr: np.ndarray, nodata: float | None):
    """Build (H, W) bool mask of pixels to exclude from stats / show as missing.

    Combines explicit `nodata` value + NaN (for floats). For multi-channel
    arrays a pixel is masked if ANY channel matches.
    """
    nd_mask = None
    if nodata is not None:
        m = np.isclose(arr, nodata) if np.issubdtype(arr.dtype, np.floating) else (arr == nodata)
        nd_mask = m.any(axis=-1) if m.ndim == 3 else m
    if np.issubdtype(arr.dtype, np.floating):
        nan_m = np.isnan(arr)
        nan_m = nan_m.any(axis=-1) if nan_m.ndim == 3 else nan_m
        nd_mask = nan_m if nd_mask is None else (nd_mask | nan_m)
    return nd_mask


def _resolve_range(
    arr: np.ndarray,
    nd_mask,
    pct: float,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    """Compute (vmin, vmax) for grayscale imshow.

    Explicit `vmin`/`vmax` override percentile. If either is None and pct
    determines the missing one. Returns (None, None) for empty/uniform input
    so matplotlib falls back to its own auto-scale.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax

    valid = arr[~nd_mask] if nd_mask is not None and nd_mask.any() else arr
    if valid.size == 0:
        return vmin, vmax

    if pct < 100.0:
        lo = float(np.percentile(valid, 100.0 - pct))
        hi = float(np.percentile(valid, pct))
    else:
        lo = float(valid.min())
        hi = float(valid.max())

    if lo == hi:
        return vmin, vmax  # let mpl handle it

    return (lo if vmin is None else vmin, hi if vmax is None else vmax)


def _to_uint8_color(
    arr: np.ndarray,
    nd_mask,
    pct: float,
    vmin: float | None,
    vmax: float | None,
) -> np.ndarray:
    """Normalise an (H, W, 3) or (H, W, 4) array to uint8 for imshow.

    Uses GLOBAL min/max across channels (matches print_image semantics).
    Already-uint8 arrays pass through. Bool -> 0/255.
    """
    if arr.dtype == np.uint8:
        out = arr.copy()
    elif arr.dtype == bool:
        out = (arr.astype(np.uint8) * 255)
    else:
        valid = arr[~nd_mask] if nd_mask is not None and nd_mask.any() else arr
        if valid.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)

        if vmin is not None and vmax is not None:
            lo, hi = float(vmin), float(vmax)
        elif pct < 100.0:
            lo = float(vmin) if vmin is not None else float(np.percentile(valid, 100.0 - pct))
            hi = float(vmax) if vmax is not None else float(np.percentile(valid, pct))
        else:
            lo = float(vmin) if vmin is not None else float(valid.min())
            hi = float(vmax) if vmax is not None else float(valid.max())

        # [0, 1] float shortcut
        if (np.issubdtype(arr.dtype, np.floating)
                and lo >= 0.0 and hi <= 1.0 and pct >= 100.0
                and vmin is None and vmax is None):
            out = (arr * 255).clip(0, 255).astype(np.uint8)
        elif lo == hi:
            out = np.zeros(arr.shape, dtype=np.uint8)
        else:
            out = ((arr.astype(np.float64) - lo) / (hi - lo) * 255
                   ).clip(0, 255).astype(np.uint8)

    if nd_mask is not None and nd_mask.any():
        out[nd_mask] = 0
    return out


# =============================================================================
# Cartopy helpers — basemap tiles, minimap inset
# =============================================================================

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


class EsriWorldImagery(cimgt.GoogleWTS if HAS_CARTOPY else object):
    """Esri satellite basemap (free for academic use, no API key)."""

    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )


def add_minimap(
    fig,
    ax,
    lon: float,
    lat: float,
    size: float = 0.2,
):
    """Add a small orthographic globe inset showing a location.

    Args:
        fig:  Figure (needed for add_axes).
        ax:   Parent axes — minimap is placed in its upper-right corner.
        lon:  Longitude (degrees).
        lat:  Latitude (degrees).
        size: Fraction of parent axes width.
    """
    if not HAS_CARTOPY:
        return None

    proj = ccrs.Orthographic(central_longitude=lon, central_latitude=lat)

    # Position: upper-right of parent axes
    bbox = ax.get_position()
    inset_w = bbox.width * size
    inset_h = bbox.height * size
    inset = fig.add_axes(
        [bbox.x1 - inset_w - 0.005, bbox.y1 - inset_h - 0.005,
         inset_w, inset_h],
        projection=proj,
    )

    inset.set_global()
    inset.add_feature(cfeature.LAND, facecolor="#d4d4d4", edgecolor="none")
    inset.add_feature(cfeature.OCEAN, facecolor="#b8d4e8", edgecolor="none")
    inset.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#888888")

    # Red dot
    inset.plot(
        lon, lat,
        marker="o", color="#e63946", markersize=4,
        markeredgecolor="white", markeredgewidth=0.5,
        transform=ccrs.PlateCarree(), zorder=10,
    )
    inset.spines["geo"].set_edgecolor("#666666")
    inset.spines["geo"].set_linewidth(0.5)

    return inset


def cartopy_heatmap(
    ax,
    data: np.ndarray,
    *,
    extent: list[float] = (-180, 180, -90, 90),
    cmap="YlOrRd",
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    zorder: int = 2,
) -> "mpl.collections.QuadMesh":
    """Draw a lat/lon grid on a cartopy GeoAxes using ``pcolormesh``.

    Unlike ``ax.imshow(..., transform=PlateCarree())``, pcolormesh draws
    each cell as a proper projected quadrilateral — no interpolation seams,
    no border artifacts between NaN and data cells.

    Args:
        ax:     Cartopy GeoAxes.
        data:   (nlat, nlon) array.  NaN cells are transparent.
        extent: (lon_min, lon_max, lat_min, lat_max) in degrees.
        cmap:   Colormap (name or instance).
        norm:   Optional matplotlib Normalize (e.g. LogNorm).
        vmin / vmax: Explicit range (ignored when *norm* is given).
        zorder: Drawing order.

    Returns:
        The QuadMesh artist (pass to ``fig.colorbar(...)``).
    """
    import cartopy.crs as ccrs

    nlat, nlon = data.shape
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, nlon + 1)
    lats = np.linspace(lat_min, lat_max, nlat + 1)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    if isinstance(cmap, str):
        cmap = plt.colormaps[cmap].copy()
    cmap.set_bad(alpha=0)  # NaN → transparent

    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    return ax.pcolormesh(
        lon_mesh, lat_mesh, data,
        cmap=cmap, shading="flat", rasterized=True,
        transform=ccrs.PlateCarree(), zorder=zorder,
        **kw,
    )


def add_satellite_basemap(
    ax,
    extent: list[float],
    *,
    zoom: int | None = None,
    min_zoom: int = 4,
    max_zoom: int = 14,
) -> None:
    """Add Esri World Imagery satellite tiles to a cartopy GeoAxes.

    Consolidates the recurring pattern: set_extent → compute zoom →
    add Esri tiles.  Used by ``figures.py`` (fire type panels, close-ups)
    and ``solidity.py`` (patch analysis grids).

    Args:
        ax:       Cartopy GeoAxes.
        extent:   ``[lon_min, lon_max, lat_min, lat_max]`` in degrees.
        zoom:     Explicit tile zoom level.  If *None*, auto-computed from
                  the extent width so that each panel covers ~1 440 px.
        min_zoom: Lower bound for auto zoom (default 4).
        max_zoom: Upper bound for auto zoom (default 14).
    """
    if not HAS_CARTOPY:
        return
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if zoom is None:
        extent_deg = max(extent[1] - extent[0], 1e-6)
        zoom = int(np.clip(np.log2(1440.0 / extent_deg), min_zoom, max_zoom))
    tiles = EsriWorldImagery()
    ax.add_image(tiles, zoom)
