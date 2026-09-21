"""
next_day/figures.py — Pure visualization functions for Paper 1 figures.

Each function takes numpy arrays / dicts and returns a matplotlib Figure.
No data loading, no Dataset objects, no file I/O beyond the optional
save_figure call.  Data assembly is the caller's job (run_figures.py or a test).

Uses firecomp/core/plotting.py helpers for house style.
Output: PDF via save_figure(..., formats=("pdf",)).

Figures
=======
1  fig:inputs        fig_input_group          One PDF per channel group (simple grid)
2  fig:regions       fig_region_extents       World map, regions colour-coded (qualitative palette + legend)
3  fig:region_stats  fig_region_stats         Bar chart: positive pixel ratio + no-data frac per region
4  fig:fire_types    fig_fire_types           3×3 grid: 3 fire types × 3 examples, detection rasters
5  fig:clusters      fig_cluster_burnability  World map, 0.1° cells coloured by tame-to-wild ratio (log)
6  fig:cluster_lc    fig_cluster_lc           World map, 0.1° cells coloured by dominant land-cover class
7  fig:lc_ratio      fig_lc_ratio             Bar chart: tame-to-wild ratio per LC class (global)
8  fig:lc_ratio_uw   fig_lc_ratio             Bar chart: same but unweighted (each fire = 1)
10 fig:density       fig_sample_density       Grid: sample density heatmap per year-group (pcolormesh)
10b fig:num_fire_hist fig_num_fire_histograms 2-col grid of per-region log-binned num_fire histograms (dataset splits)
11 fig:predictions   fig_prediction_samples   Grid: accum_t | cur_mask | ground truth | prob map | TP/FP/FN overlay
12 fig:size_region   fig_size_and_region_f1   Scatter: per-sample F1 by region, sized by fire size bucket
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from firecomp.core.plotting import (
    Cmaps,
    Style,
    add_colorbar,
    add_minimap,
    add_satellite_basemap,
    cartopy_heatmap,
    imshow_tensor,
    make_figure,
    save_figure,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1  fig:inputs — one PDF per channel group
# ═══════════════════════════════════════════════════════════════════════════


def fig_input_group(
    channels: dict[str, np.ndarray],
    group_name: str,
    *,
    cmaps: dict[str, str | mpl.colors.Colormap] | None = None,
    nodata: dict[str, float] | None = None,
    ncols: int = 3,
    pct: float = 98.0,
    out_path: str | None = None,
) -> plt.Figure:
    """Plot channels belonging to one input group in a simple grid.

    The caller produces one PDF per group (Terrain, Fire State, Weather, etc.)
    and assembles the final composite figure externally.

    Args:
        channels:   {label: (H, W) array} — ordered dict of 2-D arrays for
                    this group only.
        group_name: Display name shown as suptitle (e.g. "Weather (ERA5)").
        cmaps:      Per-channel colourmap overrides {label: cmap}.
        nodata:     Per-channel nodata values {label: value}.
        ncols:      Grid width (default 3).
        pct:        Percentile stretch (default 98).
        out_path:   Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    cmaps = cmaps or {}
    nodata = nodata or {}
    labels = list(channels.keys())
    n = len(labels)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = make_figure(nrows, ncols, cell=(2.8, 2.8), constrained=True)

    for i, label in enumerate(labels):
        r, c = divmod(i, ncols)
        cmap = cmaps.get(label, "viridis")
        nd = nodata.get(label)
        imshow_tensor(axes[r, c], channels[label], title=label, cmap=cmap,
                      pct=pct, nodata=nd)

    # blank remaining cells
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


def fig_input_grid(
    channels: dict[str, np.ndarray],
    *,
    ncols: int = 4,
    pct: float = 98.0,
    cmap: str = "viridis",
    nodata: dict[str, float] | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """4-column N-row grid of all input channels with individual colorbars.

    Args:
        channels: ``{label: (H, W) array}`` — ordered dict.
        ncols:    Grid width.
        pct:      Percentile stretch.
        cmap:     Default colormap for all channels.
        nodata:   Per-channel nodata values.
        out_path: Stem for output files.
    """
    nodata = nodata or {}
    labels = list(channels.keys())
    n = len(labels)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = make_figure(nrows, ncols, cell=(2.4, 2.2), constrained=True,
                            hspace=0.03, wspace=0.03)

    for i, label in enumerate(labels):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        arr = channels[label].copy().astype(np.float32)
        nd = nodata.get(label)
        if nd is not None:
            arr[arr == nd] = np.nan

        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lo = np.percentile(finite, 100 - pct)
            hi = np.percentile(finite, pct)
        else:
            lo, hi = 0, 1

        cmap_obj = mpl.colormaps[cmap].copy()
        cmap_obj.set_bad(color="#333333")
        masked = np.where(np.isfinite(arr), arr, np.nan)
        im = ax.imshow(masked, cmap=cmap_obj, vmin=lo, vmax=hi,
                        interpolation="nearest")
        ax.set_title(label, fontsize=11, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2  fig:regions — world map coloured by sample count per split
# ═══════════════════════════════════════════════════════════════════════════


def fig_region_samples(
    region_samples: dict[str, int],
    title: str = "Sample distribution",
    *,
    raster: np.ndarray | None = None,
    cmap: str = "YlOrRd",
    out_path: str | None = None,
) -> plt.Figure:
    """World map with study regions rasterised and coloured by sample count.

    Each region's full footprint (all pixels in the region raster) is filled
    with a single heatmap colour proportional to its sample count (log scale).
    Produces one map per split — call once for train, val, test.

    Args:
        region_samples: {region_name: sample_count}.  Names must match the
                        Regions registry (e.g. "Western Europe").
        title:          Figure title (e.g. "Train — samples per region").
        raster:         Optional (H, W) uint8 region raster (0 = no region,
                        1-N = region IDs).  If None, loaded from disk via
                        RegionRaster.
        cmap:           Colormap name (default "YlOrRd").
        out_path:       Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from firecomp.core.regions import Regions

    # -- load raster --
    if raster is None:
        from firecomp.core.regions import RegionRaster
        raster = RegionRaster.load().raster  # (H, W) uint8

    regions = Regions()

    # -- build name → region_id mapping --
    name_to_id: dict[str, int] = {}
    for name in region_samples:
        if name in regions:
            name_to_id[name] = regions[name]

    # -- build RGBA image: colour each region's pixels (log scale) --
    counts = [v for k, v in region_samples.items() if k in name_to_id]
    vmin = max(min(counts), 1) if counts else 1
    vmax = max(counts) if counts else 1
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    colormap = plt.colormaps[cmap]

    h, w = raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)  # fully transparent

    for name, count in region_samples.items():
        if name not in name_to_id:
            continue
        rid = name_to_id[name]
        mask = raster == rid
        color = colormap(norm(max(count, 1)))  # (r, g, b, a) floats 0-1
        rgba[mask] = (np.array(color[:3]) * 255).astype(np.uint8).tolist() + [200]

    # -- plot --
    proj = ccrs.Robinson()
    fig, axes = make_figure(1, 1, figsize=(10, 5), projection=proj)
    ax = axes[0, 0]

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    # Overlay rasterised regions
    ax.imshow(
        rgba,
        origin="upper",
        extent=[-180, 180, -90, 90],
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        zorder=2,
    )

    # Colorbar (log scale)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Samples", fontsize=Style.label_fontsize)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


def fig_region_extents(
    region_names: dict[int, str],
    *,
    raster: np.ndarray | None = None,
    title: str = "Study regions",
    out_path: str | None = None,
) -> plt.Figure:
    """Single world map showing the spatial extent of each study region.

    Each region is filled with a distinct colour (qualitative palette)
    and a legend maps colours to region names.

    Args:
        region_names: ``{region_id: display_name}`` for all regions to show.
        raster:       (H, W) uint8 region raster (0 = background, 1-N = IDs).
                      Loaded from disk if None.
        title:        Figure title.
        out_path:     Stem for output files (no extension).  None = don't save.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if raster is None:
        from firecomp.core.regions import RegionRaster
        raster = RegionRaster.load().raster

    # Qualitative palette — one distinct colour per region
    n = len(region_names)
    base_cmap = plt.colormaps["tab10"] if n <= 10 else plt.colormaps["tab20"]
    colors = {rid: base_cmap(i % base_cmap.N)
              for i, rid in enumerate(sorted(region_names))}

    h, w = raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for rid, color in colors.items():
        mask = raster == rid
        rgba[mask] = (np.array(color[:3]) * 255).astype(np.uint8).tolist() + [200]

    proj = ccrs.Robinson()
    fig, axes = make_figure(1, 1, figsize=(12, 6), projection=proj)
    ax = axes[0, 0]

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    ax.imshow(
        rgba, origin="upper", extent=[-180, 180, -90, 90],
        transform=ccrs.PlateCarree(), interpolation="nearest", zorder=2,
    )

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[rid], edgecolor="grey", linewidth=0.5,
                     label=region_names[rid])
               for rid in sorted(region_names)]
    ax.legend(handles=handles, loc="lower left", fontsize=Style.label_fontsize - 2,
              framealpha=Style.legend_framealpha, edgecolor="#cccccc",
              ncol=2, handlelength=1.2, handleheight=1.0)

    ax.set_title(title, fontsize=Style.title_fontsize, pad=Style.title_pad)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3  fig:region_stats — positive pixel ratio + no-data fraction per region
# ═══════════════════════════════════════════════════════════════════════════


def fig_region_stats(
    region_stats: dict[str, dict[str, float]],
    title: str = "Region class balance",
    *,
    out_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart: positive pixel ratio and no-data fraction per region.

    Illuminates how focal loss alpha / pos_weight should vary across regions.
    Regions with high no-data fractions have unreliable labels; regions with
    very low positive ratios need aggressive class weighting.

    Args:
        region_stats: {region_name: {"pos_ratio": float, "nodata_ratio": float,
                       "n_samples": int}}.  Ratios in [0, 1].
        title:        Figure title.
        out_path:     Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    # Sort regions by sample count (largest on top)
    sorted_names = sorted(
        region_stats.keys(),
        key=lambda n: region_stats[n].get("n_samples", 0),
    )

    pos_ratios = [region_stats[n]["pos_ratio"] for n in sorted_names]
    nodata_ratios = [region_stats[n]["nodata_ratio"] for n in sorted_names]
    n_samples = [region_stats[n].get("n_samples", 0) for n in sorted_names]
    labels = [f"{n}  (n={ns:,})" for n, ns in zip(sorted_names, n_samples)]

    n_regions = len(sorted_names)
    fig, (ax_pos, ax_nd) = plt.subplots(
        1, 2, figsize=(10, max(3, 0.4 * n_regions)), sharey=True,
        constrained_layout=True,
    )

    y = np.arange(n_regions)
    bar_h = 0.7

    # -- positive pixel ratio --
    ax_pos.barh(y, pos_ratios, height=bar_h, color="#e74c3c", alpha=0.85)
    ax_pos.set_xlabel("Positive pixel ratio", fontsize=Style.label_fontsize)
    ax_pos.set_title("Fire pixels / total pixels", fontsize=Style.title_fontsize)
    ax_pos.set_yticks(y)
    ax_pos.set_yticklabels(labels, fontsize=Style.label_fontsize - 1)
    ax_pos.set_xlim(0, None)

    # Annotate exact values
    for i, v in enumerate(pos_ratios):
        ax_pos.text(v, i, f" {v:.4f}", va="center", fontsize=Style.label_fontsize - 2)

    # -- no-data fraction --
    ax_nd.barh(y, nodata_ratios, height=bar_h, color="#7f8c8d", alpha=0.85)
    ax_nd.set_xlabel("No-data fraction", fontsize=Style.label_fontsize)
    ax_nd.set_title("Unobservable pixels / total pixels",
                    fontsize=Style.title_fontsize)
    ax_nd.set_xlim(0, None)

    for i, v in enumerate(nodata_ratios):
        ax_nd.text(v, i, f" {v:.2f}", va="center", fontsize=Style.label_fontsize - 2)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4  fig:fire_types — 3×3 grid of fire type examples
# ═══════════════════════════════════════════════════════════════════════════


def fig_fire_types(
    panels: list[list[np.ndarray]],
    type_labels: list[str],
    fire_labels: list[list[str]],
    *,
    fire_geo: list[list[dict]] | None = None,
    fire_stats: list[list[dict]] | None = None,
    row_max_days: list[float | None] | None = None,
    title: str = "",
    out_path: str | None = None,
) -> plt.Figure:
    """3×N grid showing detection rasters for different fire types.

    Each panel is a rasterised fire detection map where pixel values encode
    the detection day (0 or NaN = no fire).  Rows = fire types, columns =
    example fires.

    When ``fire_geo`` is provided, each panel is rendered on a Cartopy
    GeoAxes with an Esri satellite basemap.  A narrow info column to
    the right of each panel shows a minimap globe and optional fire
    classification stats (t_ratio, xy_neighbors, ign_ratio).

    Args:
        panels:      ``panels[row][col]`` = (H, W) array.  Pixel value =
                     detection day (NaN = background).
        type_labels: One label per row (e.g. ``["Static", "Crop", "Wild"]``).
        fire_labels: ``fire_labels[row][col]`` = per-panel subtitle.
        fire_geo:    ``fire_geo[row][col]`` = dict with keys
                     ``lon_min, lon_max, lat_min, lat_max`` (degrees) and
                     optionally ``lon_mid, lat_mid`` for the minimap.
                     None = plain imshow (no basemap / minimap).
        fire_stats:  ``fire_stats[row][col]`` = dict with optional keys
                     ``t_ratio``, ``xy_neighbors``, ``ign_ratio``.
                     Shown as text below the minimap.  None = no stats.
        row_max_days: Per-row cap on the colour-bar maximum (days).
                     ``[None, 60, None]`` clamps only the 2nd row to 60 d.
                     None = no clamping.
        title:       Figure suptitle.
        out_path:    Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    import cartopy.crs as ccrs

    nrows = len(panels)
    ncols = max(len(row) for row in panels)
    use_geo = fire_geo is not None

    # (info_ax, lon, lat, stats_dict) — deferred until after layout
    _info_requests: list[tuple] = []

    # --- build figure ---
    if use_geo:
        proj = ccrs.PlateCarree()
        info_ratio = 0.22
        cell_w, cell_h = 3.6, 3.6
        width_ratios: list[float] = []
        for _ in range(ncols):
            width_ratios.extend([1.0, info_ratio])

        fig = plt.figure(
            figsize=(cell_w * ncols * (1 + info_ratio), cell_h * nrows),
            constrained_layout=True,
        )
        # Tighten horizontal gap so info column sits close to its panel
        fig.get_layout_engine().set(wspace=0.005)
        gs = fig.add_gridspec(nrows, ncols * 2, width_ratios=width_ratios)

        axes = np.empty((nrows, ncols), dtype=object)
        info_axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, c * 2], projection=proj)
                info_axes[r, c] = fig.add_subplot(gs[r, c * 2 + 1])
                info_axes[r, c].axis("off")
    else:
        fig, axes = make_figure(nrows, ncols, cell=(3.2, 3.2),
                                constrained=True)
        info_axes = None

    # Colormap: fire_spread (red→orange→yellow)
    cmap = Cmaps.fire_spread.copy()
    cmap.set_bad(color="#e0e0e0" if not use_geo else (0, 0, 0, 0))

    # --- render panels ---
    for r in range(nrows):
        row_vals = [p[np.isfinite(p)] for p in panels[r]]
        row_vmax = max((v.max() for v in row_vals if v.size > 0), default=1)
        # Per-row clamp (e.g. cap crop-burn row at 60 days)
        if row_max_days and r < len(row_max_days) and row_max_days[r] is not None:
            row_vmax = min(row_vmax, row_max_days[r])

        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(panels[r]):
                ax.axis("off")
                if info_axes is not None:
                    info_axes[r, c].axis("off")
                continue

            panel = panels[r][c]

            if use_geo and fire_geo[r][c]:
                geo = fire_geo[r][c]
                # Pad extent to square so all cells render uniformly
                dlon = geo["lon_max"] - geo["lon_min"]
                dlat = geo["lat_max"] - geo["lat_min"]
                lon_mid = (geo["lon_min"] + geo["lon_max"]) / 2
                lat_mid = (geo["lat_min"] + geo["lat_max"]) / 2
                half = max(dlon, dlat) * 0.55   # square + 10 % buffer
                sq_extent = [lon_mid - half, lon_mid + half,
                             lat_mid - half, lat_mid + half]
                add_satellite_basemap(ax, sq_extent)

                # Overlay fire raster (use original extent, not padded).
                # origin="lower": row 0 of the raster = y.min() = south,
                # so it maps to the bottom of the extent (lat_min).
                fire_extent = [geo["lon_min"], geo["lon_max"],
                               geo["lat_min"], geo["lat_max"]]
                arr = panel.astype(np.float32).copy()
                ax.imshow(
                    arr, cmap=cmap, vmin=0, vmax=row_vmax,
                    extent=fire_extent, origin="lower",
                    transform=ccrs.PlateCarree(),
                    interpolation="nearest", alpha=0.9, zorder=2,
                )
                ax.set_title(fire_labels[r][c],
                             fontsize=Style.title_fontsize + 2,
                             pad=Style.title_pad)

                # Collect info-column requests (deferred until layout)
                fs = fire_stats[r][c] if fire_stats else {}
                _info_requests.append((
                    ax,                    # fire panel (alignment ref)
                    info_axes[r, c],
                    geo.get("lon_mid", lon_mid),
                    geo.get("lat_mid", lat_mid),
                    fs,
                    panel,               # for close-up origin detection
                    geo,                 # for close-up lon/lat conversion
                    r,                   # row index (for stat highlighting)
                ))
            else:
                imshow_tensor(
                    ax, panel,
                    cmap=cmap, vmin=0, vmax=row_vmax,
                    title=fire_labels[r][c],
                    aspect="auto",
                )

        # Row type label
        axes[r, 0].text(
            -0.05, 0.5, type_labels[r],
            transform=axes[r, 0].transAxes,
            fontsize=Style.label_fontsize + 1,
            fontweight="bold", rotation=90,
            ha="right", va="center",
        )

        # Per-row colorbar (steal space from all columns in the row)
        if info_axes is not None:
            cb_axes: list = []
            for c in range(ncols):
                cb_axes.append(axes[r, c])
                cb_axes.append(info_axes[r, c])
        else:
            cb_axes = [axes[r, c] for c in range(ncols)]
        norm = mpl.colors.Normalize(vmin=0, vmax=row_vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=cb_axes, shrink=0.8, pad=0.02)
        cb.set_label("Days", fontsize=Style.label_fontsize - 1)

    # Add minimap globes + stats text after layout is finalized
    if _info_requests:
        fig.canvas.draw()  # force constrained_layout to settle
        for fire_ax, info_ax, lon, lat, fs, panel, geo, ridx in _info_requests:
            _add_fire_info(fig, fire_ax, info_ax, lon, lat, fs,
                           panel=panel, geo=geo, row_idx=ridx)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


def _add_fire_info(
    fig: plt.Figure,
    fire_ax,
    info_ax,
    lon: float,
    lat: float,
    stats: dict,
    panel: np.ndarray | None = None,
    geo: dict | None = None,
    row_idx: int = -1,
) -> None:
    """Render minimap globe + classification stats + satellite close-up in an
    info column.

    The minimap is vertically aligned with the fire panel image area.
    Stats are rendered with bold keys and regular values; the defining
    stat for each fire type is shown in a highlight colour:
      row 0 (Static):    t_ratio
      row 1 (Crop Burn): xy, ign
    A satellite close-up of the fire origin (first detection day) is placed
    below the stats when panel+geo are supplied.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    bbox = info_ax.get_position()
    fire_bbox = fire_ax.get_position()

    # --- minimap globe — aligned with fire panel top edge, left-aligned ---
    globe_size = min(bbox.width * 0.95, bbox.height * 0.35)
    globe_x = bbox.x0
    globe_y = fire_bbox.y1 - globe_size  # flush with fire panel top

    proj = ccrs.Orthographic(central_longitude=lon, central_latitude=lat)
    inset = fig.add_axes(
        [globe_x, globe_y, globe_size, globe_size],
        projection=proj,
    )
    inset.set_global()
    inset.add_feature(cfeature.LAND, facecolor="#d4d4d4", edgecolor="none")
    inset.add_feature(cfeature.OCEAN, facecolor="#b8d4e8", edgecolor="none")
    inset.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#888888")
    inset.plot(
        lon, lat,
        marker="o", color="#e63946", markersize=4,
        markeredgecolor="white", markeredgewidth=0.5,
        transform=ccrs.PlateCarree(), zorder=10,
    )
    inset.spines["geo"].set_edgecolor("#666666")
    inset.spines["geo"].set_linewidth(0.5)

    # --- stats text (below minimap): bold key, regular value ---
    # Defining stats per row are highlighted in colour:
    #   row 0 (Static):    t_ratio       (high t_ratio → persistent source)
    #   row 1 (Crop Burn): xy, ign       (low xy + high ignition ratio)
    _HIGHLIGHT = {0: {"t_r"}, 1: {"xy", "ign"}}
    highlight_keys = _HIGHLIGHT.get(row_idx, set())
    HIGHLIGHT_COLOR = "#c0392b"   # dark red

    # Left-align text with the globe so the info column hugs the fire panel
    text_x = bbox.x0 + globe_size / 2
    cursor_y = globe_y - 0.012
    if stats:
        fs_key = Style.label_fontsize - 1
        fs_val = Style.label_fontsize

        stat_items = []
        if "t_ratio" in stats:
            stat_items.append(("t_r", f"{stats['t_ratio']:.1f}"))
        if "xy_neighbors" in stats:
            stat_items.append(("xy", f"{stats['xy_neighbors']:.1f}"))
        if "ign_ratio" in stats:
            stat_items.append(("ign", f"{stats['ign_ratio']:.1f}%"))

        for key, val in stat_items:
            is_hl = key in highlight_keys
            fig.text(text_x, cursor_y, f"{key}:",
                     ha="center", va="top",
                     fontsize=fs_key, fontweight="bold",
                     color=HIGHLIGHT_COLOR if is_hl else "black")
            cursor_y -= 0.018
            fig.text(text_x, cursor_y, val,
                     ha="center", va="top",
                     fontsize=fs_val, fontfamily=Style.monospace,
                     color=HIGHLIGHT_COLOR if is_hl else "black")
            cursor_y -= 0.025

    # --- satellite close-up of fire origin (below stats) ---
    if panel is not None and geo and geo.get("lon_min") is not None:
        _add_closeup(fig, info_ax, fire_bbox, panel, geo, cursor_y)


def _add_closeup(
    fig: plt.Figure,
    info_ax,
    fire_bbox,
    panel: np.ndarray,
    geo: dict,
    cursor_y: float,
) -> None:
    """Add a high-zoom Esri satellite crop centred on the fire's origin pixel.

    The origin pixel is chosen from the first detection day (t=0 after
    time-normalisation).  When multiple pixels share that day, one is picked
    at random (seed=0 for reproducibility).  The chosen pixel index is
    printed so it can be hard-coded later.

    The close-up is placed below *cursor_y* in the info column, clipped so it
    never extends below the fire panel's bottom edge.
    """
    import cartopy.crs as ccrs

    bbox = info_ax.get_position()

    # --- find first-detection-day pixels (t == 0 after normalisation) ---
    finite = panel[np.isfinite(panel)]
    if finite.size == 0:
        return
    first_day = finite.min()
    ys, xs = np.where(panel == first_day)
    if len(ys) == 0:
        return

    rng = np.random.default_rng(0)
    idx = int(rng.integers(len(ys)))
    py, px = int(ys[idx]), int(xs[idx])

    H, W = panel.shape
    fire_lon = (geo["lon_min"]
                + (px + 0.5) / W * (geo["lon_max"] - geo["lon_min"]))
    fire_lat = (geo["lat_min"]
                + (py + 0.5) / H * (geo["lat_max"] - geo["lat_min"]))

    print(f"    close-up pixel: x={px}, y={py}  →  "
          f"lon={fire_lon:.4f}, lat={fire_lat:.4f}")

    # Extent: ±0.002° ≈ ±220 m (one VIIRS pixel ≈ 375 m across)
    half = 0.002
    closeup_extent = [fire_lon - half, fire_lon + half,
                      fire_lat - half, fire_lat + half]

    # Bold "origin:" label above the close-up (same style as stat keys)
    # Centre text on the globe's horizontal midpoint (left-aligned column)
    globe_size = min(bbox.width * 0.95, bbox.height * 0.35)
    text_x = bbox.x0 + globe_size / 2
    label_y = cursor_y - 0.006
    fig.text(text_x, label_y, "origin:",
             ha="center", va="top",
             fontsize=Style.label_fontsize - 1, fontweight="bold")

    # Placement: below label, square, left-aligned with globe
    gap = 0.020
    closeup_size = bbox.width * 0.88
    closeup_x = bbox.x0
    closeup_y = label_y - gap - closeup_size

    # Skip if it would bleed below the fire panel
    if closeup_y < fire_bbox.y0:
        return

    closeup_ax = fig.add_axes(
        [closeup_x, closeup_y, closeup_size, closeup_size],
        projection=ccrs.PlateCarree(),
    )
    add_satellite_basemap(closeup_ax, closeup_extent, zoom=17)

    for spine in closeup_ax.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# 5  fig:clusters — world map coloured by tame-to-wild ratio per 0.1° cell
# ═══════════════════════════════════════════════════════════════════════════


def fig_cluster_burnability(
    ratio_map: np.ndarray,
    title: str = "Cluster tame-to-wild ratio",
    *,
    vmin: float = 0.1,
    vmax: float = 10.0,
    cmap: str = "RdYlGn_r",
    out_path: str | None = None,
) -> plt.Figure:
    """World map coloured by per-pixel tame-to-wild ratio (log scale).

    Each pixel represents a 0.1° cell whose colour is determined by
    the dominant land-cover cluster's tame_to_wild_ratio.  Low ratio
    (wildfire-dominated, burnable) renders green; high ratio
    (tame-dominated, non-burnable) renders red.

    The default range [0.1, 10] is log-symmetric around 1.0 so the
    colour midpoint sits at the tame/wild boundary.  Values outside
    this range are clamped to the extremes.

    Used by the fire filtering pipeline to visualise
    which areas would be excluded by a cluster-based burnability filter.

    Args:
        ratio_map: (H, W) float32 array at 0.1° global resolution
                   (1800 × 3600 for a full globe).  NaN = no data.
        title:     Figure title.
        vmin:      Lower bound for log colour scale (default 0.1).
        vmax:      Upper bound for log colour scale (default 10.0).
        cmap:      Colormap name.  Default "RdYlGn_r" maps
                   green → low ratio (wild), red → high ratio (tame).
        out_path:  Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Mask non-positive values to NaN so they render transparent.
    data = ratio_map.astype(np.float64).copy()
    data[~(np.isfinite(data) & (data > 0))] = np.nan

    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    # -- plot --
    proj = ccrs.Robinson()
    fig, axes = make_figure(1, 1, figsize=(12, 6), projection=proj)
    ax = axes[0, 0]

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    cartopy_heatmap(ax, data, cmap=cmap, norm=norm)

    # Colorbar (log scale) with explicit readable ticks
    sm = mpl.cm.ScalarMappable(cmap=plt.colormaps[cmap], norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Tame-to-wild ratio", fontsize=Style.label_fontsize)
    tick_vals = [v for v in [0.1, 0.2, 0.5, 1, 2, 5, 10] if vmin <= v <= vmax]
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([f"{v:g}" for v in tick_vals])

    ax.set_title(title, fontsize=Style.title_fontsize, pad=Style.title_pad)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6  fig:cluster_lc — world map coloured by dominant land-cover class
# ═══════════════════════════════════════════════════════════════════════════

# Colours for Copernicus CGLS-LC100 classes (shared with clusters.py plots)
LC_COLORS: dict[str, str] = {
    "EG_Needle": "#006400",
    "EG_Broad": "#228B22",
    "DC_Needle": "#8FBC8F",
    "DC_Broad": "#32CD32",
    "Mixed_Forest": "#2E8B57",
    "Other_Forest": "#3CB371",
    "Shrub": "#D2691E",
    "Grass": "#BDB76B",
    "Cropland": "#FFD700",
    "Urban": "#808080",
    "Bare": "#F4A460",
    "Wetland": "#4682B4",
    "Moss_Lichen": "#9ACD32",
    "Water": "#1E90FF",
    "Ocean": "#000080",
}


def fig_cluster_lc(
    lc_map: np.ndarray,
    lc_names: list[str],
    *,
    lc_colors: dict[str, str] | None = None,
    title: str = "Dominant land-cover class per cluster",
    out_path: str | None = None,
) -> plt.Figure:
    """World map coloured by the dominant land-cover class of each 0.1° cell.

    Each cell's colour is determined by the dominant cluster's most common
    Copernicus LC100 class.  A legend identifies the classes present.

    Args:
        lc_map:    (H, W) int8 array at 0.1° resolution.  Each value is an
                   index into *lc_names*.  -1 = no data (transparent).
        lc_names:  Ordered list of LC class labels (index → name).
        lc_colors: {name: hex_color} overrides.  Falls back to LC_COLORS.
        title:     Figure title.
        out_path:  Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    colors = {**LC_COLORS, **(lc_colors or {})}

    h, w = lc_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    present_classes: list[str] = []

    for idx, name in enumerate(lc_names):
        mask = lc_map == idx
        if not mask.any():
            continue
        present_classes.append(name)
        hex_c = colors.get(name, "#888888")
        r_val = int(hex_c[1:3], 16)
        g_val = int(hex_c[3:5], 16)
        b_val = int(hex_c[5:7], 16)
        rgba[mask] = [r_val, g_val, b_val, 200]

    # -- plot --
    proj = ccrs.Robinson()
    fig, axes = make_figure(1, 1, figsize=(12, 6), projection=proj)
    ax = axes[0, 0]

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#999999")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    ax.imshow(
        rgba,
        origin="upper",
        extent=[-180, 180, -90, 90],
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        zorder=2,
    )

    # Legend (categorical — not a colorbar)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors.get(name, "#888888"), edgecolor="none",
              label=name)
        for name in present_classes
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=Style.label_fontsize - 2,
        framealpha=Style.legend_framealpha,
        ncol=2,
        title="Land cover",
        title_fontsize=Style.label_fontsize - 1,
    )

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7  fig:lc_ratio — tame-to-wild ratio per LC class (bar chart)
# ═══════════════════════════════════════════════════════════════════════════


def fig_lc_ratio(
    lc_names: list[str],
    ratios: dict[str, float],
    n_wild: dict[str, int] | None = None,
    n_tame: dict[str, int] | None = None,
    *,
    title: str = "Tame-to-wild ratio per LC class",
    out_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of tame-to-wild ratio per LC class (log scale).

    Bars are coloured on a green-red gradient (green = wildfire-dominated,
    red = tame-dominated) with the ratio=1 boundary marked.

    Args:
        lc_names:  Ordered list of LC class names.
        ratios:    {lc_name: ratio} — tame-to-wild ratio (weighted or not).
        n_wild:    {lc_name: count} — optional, shown as annotation.
        n_tame:    {lc_name: count} — optional, shown as annotation.
        title:     Figure title.
        out_path:  Stem for output files (no extension).

    Returns:
        matplotlib Figure.
    """
    # Filter to LC classes that have fires
    names = [n for n in lc_names if n in ratios and ratios[n] > 0]
    if not names:
        fig, _ = make_figure(1, 1, figsize=(8, 4))
        return fig

    vals = np.array([ratios[n] for n in names])

    # Sort by ratio (ascending = most wild at top)
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = vals[order]

    # Colour bars by ratio (green = wild, red = tame)
    cmap = plt.colormaps["RdYlGn_r"]
    log_vals = np.log10(np.clip(vals, 0.01, 100))
    log_min, log_max = -2, 2
    normed = (log_vals - log_min) / (log_max - log_min)
    normed = np.clip(normed, 0.0, 1.0)
    bar_colors = cmap(normed)

    fig, axes = make_figure(1, 1, figsize=(8, max(3, len(names) * 0.4)))
    ax = axes[0, 0]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=bar_colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=Style.label_fontsize)
    ax.set_xscale("log")
    ax.set_xlabel("Tame-to-wild ratio", fontsize=Style.label_fontsize)
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)

    # Set sensible x range
    xmin = max(0.005, vals.min() * 0.5)
    xmax = min(200, vals.max() * 2)
    ax.set_xlim(xmin, xmax)

    # Annotate bars with counts
    if n_wild is not None and n_tame is not None:
        for i, name in enumerate(names):
            nw = n_wild.get(name, 0)
            nt = n_tame.get(name, 0)
            ax.text(vals[i] * 1.1, i, f"  w={nw:,} t={nt:,}",
                    va="center", fontsize=Style.label_fontsize - 2,
                    color="#555555")

    ax.set_title(title, fontsize=Style.title_fontsize, pad=Style.title_pad)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf",))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10  fig:density — sample density heatmap per year
# ═══════════════════════════════════════════════════════════════════════════


def fig_sample_density(
    group_counts: dict[str, np.ndarray],
    *,
    resolution: float = 3.0,
    cmap: str = "YlOrRd",
    title: str = "Sample density",
    out_path: str | None = None,
) -> plt.Figure:
    """Grid of world maps showing sample density per year-group.

    Each subplot is a Robinson-projection map with a heatmap overlay
    rendered via ``pcolormesh`` (no interpolation artefacts).
    Cells with zero samples are transparent.  Colour scale is log,
    shared across all panels.

    Args:
        group_counts: ``{label: (nlat, nlon) int array}`` — one entry
                      per panel.  Labels become subplot titles (e.g.
                      ``"2018–2019"``).
        resolution:   Grid cell size in degrees (default 3.0).
        cmap:         Colormap name.
        title:        Figure suptitle.
        out_path:     Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    labels = list(group_counts.keys())
    n_panels = len(labels)
    ncols = min(n_panels, 2)
    nrows = max(1, (n_panels + ncols - 1) // ncols)

    # Pad label list so the grid is full
    while len(labels) < nrows * ncols:
        labels.append(None)

    # Global vmin / vmax (across all groups, ignoring zeros)
    all_nonzero = np.concatenate(
        [c[c > 0].ravel() for c in group_counts.values()]
    )
    if all_nonzero.size == 0:
        vmin_val, vmax_val = 1, 1
    else:
        vmin_val = max(int(all_nonzero.min()), 1)
        vmax_val = int(all_nonzero.max())
    norm = mpl.colors.LogNorm(vmin=vmin_val, vmax=vmax_val)

    proj = ccrs.Robinson()
    fig, axes = make_figure(nrows, ncols, cell=(6.0, 3.2),
                            constrained=True, projection=proj,
                            hspace=0.02, wspace=0.02)

    mesh = None
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        label = labels[idx]

        if label is None or label not in group_counts:
            ax.axis("off")
            continue

        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "ocean", "110m", facecolor="#e6f2ff", edgecolor="none"))
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#999999")

        # Convert to float, zeros → NaN for transparent rendering
        counts = group_counts[label].astype(np.float64)
        counts[counts == 0] = np.nan

        mesh = cartopy_heatmap(ax, counts, cmap=cmap, norm=norm)

        ax.set_title(label, fontsize=Style.title_fontsize,
                      pad=Style.title_pad)

    # Shared colorbar
    sm = mpl.cm.ScalarMappable(cmap=plt.colormaps[cmap], norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02,
                      aspect=30)
    cb.set_label("Samples per cell", fontsize=Style.label_fontsize)
    tick_vals = [v for v in [1, 10, 100, 1000] if vmin_val <= v <= vmax_val]
    if tick_vals:
        cb.set_ticks(tick_vals)
        cb.set_ticklabels([str(v) for v in tick_vals])

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10b  fig:num_fire_hist — per-region log-binned histograms of num_fire
# ═══════════════════════════════════════════════════════════════════════════

# Default log-spaced bin edges spanning small (3 px) to huge (50 k px) fires.
NUM_FIRE_BIN_EDGES: list[float] = [
    1, 3, 5, 10, 20, 50, 100, 200, 500,
    1000, 2000, 5000, 10000, 20000, 50000,
]


def fig_num_fire_histograms(
    region_data: dict[str, np.ndarray],
    *,
    bin_edges: list[float] | None = None,
    ncols: int = 2,
    color: str = "#e67e22",
    title: str = "num_fire per sample, by region",
    out_path: str | None = None,
) -> plt.Figure:
    """Per-region histograms of per-sample ``num_fire`` on log-spaced bins.

    Each panel shows how the samples in one region are distributed across
    fire-size buckets (``num_fire`` is the total detection count of the
    fire that produced the sample).  All panels share the same x-axis bin
    edges so the shapes are directly comparable; y-axes are per-panel so
    small regions stay readable.

    Args:
        region_data: {region_name: 1-D array of num_fire values}.
                     Insertion order = display order (row-major, 2 cols).
        bin_edges:   Histogram bin edges (log-spaced).  Defaults to
                     ``NUM_FIRE_BIN_EDGES``.
        ncols:       Grid width (default 2).
        color:       Bar colour.
        title:       Figure suptitle.
        out_path:    Stem for output files (no extension).  None = don't save.

    Returns:
        matplotlib Figure.
    """
    if bin_edges is None:
        bin_edges = NUM_FIRE_BIN_EDGES
    edges = np.asarray(bin_edges, dtype=float)
    widths = np.diff(edges)

    names = list(region_data.keys())
    n = len(names)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = make_figure(nrows, ncols, cell=(4.6, 2.6), constrained=True)

    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        vals = np.asarray(region_data[name], dtype=float)
        vals_pos = vals[vals > 0]
        counts, _ = np.histogram(vals_pos, bins=edges)

        ax.bar(edges[:-1], counts, width=widths, align="edge",
               color=color, edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xticks(edges)
        ax.set_xticklabels([f"{int(e)}" for e in edges],
                           rotation=45, ha="right",
                           fontsize=Style.label_fontsize - 3)
        ax.tick_params(axis="y", labelsize=Style.label_fontsize - 2)

        # Suppress matplotlib's minor log-decade labels (1, 2, 3, ...)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        median = float(np.median(vals_pos)) if vals_pos.size else 0.0
        ax.set_title(f"{name}  (n={vals.size:,}, median={median:.0f})",
                     fontsize=Style.title_fontsize)
        if r == nrows - 1 or (i + ncols >= n):
            ax.set_xlabel("num_fire (detections per fire)",
                          fontsize=Style.label_fontsize)
        if c == 0:
            ax.set_ylabel("Samples", fontsize=Style.label_fontsize)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 11  fig:predictions — probability maps from a trained checkpoint
# ═══════════════════════════════════════════════════════════════════════════


def fig_prediction_samples(
    samples: list[dict],
    *,
    title: str = "Prediction probability maps",
    out_path: str | None = None,
) -> mpl.figure.Figure:
    """Grid of prediction visualizations: one row per sample.

    Each row has 5 columns:
        0. accum_t — fire history (where fire has been)
        1. cur_mask — active fire on day_T
        2. ground truth — next_mask (day_T+1)
        3. probability map — continuous model output (inferno colormap)
        4. TP / FP / FN overlay — green=TP, red=FP, blue=FN

    When ``padding`` is provided (or inferred from ``loss_mask``), all
    panels are cropped to the valid interior, removing the border
    region where the model lacks full context.

    When ``loss_mask`` is present, unobserved pixels inside the crop
    are excluded from the TP/FP/FN overlay and shown as dark grey.

    Args:
        samples: list of dicts, each with keys:
            accum_t:     (H, W) float — accumulated fire history
            cur_mask:    (H, W) float — current fire mask
            target:      (H, W) float — ground truth next-day mask
            pred:        (H, W) float — predicted probability [0, 1]
            threshold:   float — binarization threshold
            fire_id:     int
            dt:          str — date
            f1:          float — per-sample F1 (optional)
            loss_mask:   (H, W) float — 1=valid, 0=ignore (optional)
            padding:     int — border pixels to crop (optional)
        title: figure suptitle
        out_path: if given, save PDF here

    Returns:
        matplotlib Figure
    """
    from firecomp.core.metrics import Metrics

    n_rows = len(samples)
    n_cols = 4
    col_titles = ["accum_t", "ground truth", "pred probability", ""]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.0 * n_rows + 0.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, s in enumerate(samples):
        accum = s["accum_t"]
        cur = s["cur_mask"]
        gt = s["target"]
        pred = s["pred"]
        threshold = s["threshold"]
        mask = s.get("loss_mask")     # (H, W) or None
        p = s.get("padding", 0)

        # Crop border padding from all arrays
        if p > 0:
            c = slice(p, -p)
            accum = accum[c, c]
            cur = cur[c, c]
            gt = gt[c, c]
            pred = pred[c, c]
            if mask is not None:
                mask = mask[c, c]

        # Col 0: accum_t — masked where no fire history
        ax = axes[ri, 0]
        masked = np.ma.masked_where(accum < -0.5, accum)
        ax.imshow(masked, cmap="YlOrRd", interpolation="nearest")
        ax.set_ylabel(f"fire {s['fire_id']}\n{s['dt']}",
                      fontsize=Style.label_fontsize - 2,
                      rotation=0, labelpad=55, va="center")

        # # Col: cur_mask (disabled — redundant with accum_t)
        # ax = axes[ri, ?]
        # ax.imshow(cur, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")

        # Col 1: ground truth (next_mask)
        ax = axes[ri, 1]
        ax.imshow(gt, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")

        # Col 2: probability map — show raw model output everywhere
        ax = axes[ri, 2]
        im = ax.imshow(pred, cmap="viridis", vmin=0, vmax=1,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("P(fire)", fontsize=Style.label_fontsize - 2)
        cb.ax.tick_params(labelsize=Style.label_fontsize - 4)

        # Col 3: TP/FP/FN overlay — use Metrics.pixel_confusion for
        # consistency with training evaluation masking
        ax = axes[ri, 3]
        cm = Metrics.pixel_confusion(pred, gt, mask=mask, threshold=threshold)
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        if mask is not None:
            overlay[mask < 0.5] = [0.15, 0.15, 0.15]  # dark grey = unobserved
        overlay[cm["tp"]] = [0.2, 0.8, 0.2]   # green = TP
        overlay[cm["fp"]] = [0.9, 0.2, 0.2]   # red = FP
        overlay[cm["fn"]] = [0.2, 0.4, 0.9]   # blue = FN
        ax.imshow(overlay, interpolation="nearest")

        # Per-sample F1 annotation
        if "f1" in s:
            axes[ri, 3].set_title(
                f"F1={s['f1']:.3f}",
                fontsize=Style.label_fontsize - 2, pad=4)

    # Column headers (skip empty titles so per-sample F1 isn't overwritten)
    for ci, label in enumerate(col_titles):
        if label:
            axes[0, ci].set_title(label, fontsize=Style.title_fontsize,
                                  fontweight="bold", pad=8)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0.06, 0, 1, 0.98])

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig


# ── prediction grid with satellite basemap (curated case studies) ────────

_DEG_PER_PIXEL = 375 / 111_320  # ~degrees per 375 m VIIRS pixel


def _colored_title(ax, parts: list[tuple[str, object]], *, fontsize) -> None:
    """Set a multi-colour, centred axis title from (text, colour) parts."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    boxes = [TextArea(t, textprops=dict(color=c, fontsize=fontsize,
                                        fontweight="bold"))
             for t, c in parts]
    pack = HPacker(children=boxes, align="center", pad=0, sep=1)
    ab = AnnotationBbox(pack, (0.5, 1.0), xybox=(0, 6),
                        xycoords="axes fraction", boxcoords="offset points",
                        box_alignment=(0.5, 0.0), frameon=False, pad=0)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def fig_prediction_basemap(
    samples: list[dict],
    *,
    title: str | None = None,
    out_path: str | None = None,
) -> mpl.figure.Figure:
    """Curated prediction grid with an Esri satellite basemap behind accum_t.

    One row per named fire.  Four columns, scored on the *new-fires
    equivalent* basis (``nf_equiv``): every pixel that has ever burned during
    the event (``accum_t >= 0``) is masked out, so only genuinely new spread
    counts.  The accum_t column is exactly the region that gets masked.

        0. accum_t over an Esri World Imagery satellite basemap
           (YlOrRd fire-spread progression; doubles as the burned/masked area)
        1. ground truth — new next-day fire; burned pixels zeroed
        2. probability map — model output with burned pixels zeroed (viridis)
        3. TP / FP / FN overlay — green=TP, red=FP, blue=FN

    Cols 1-2 zero out the burned footprint (``accum_t >= 0``).  Col 3 also
    applies the loss mask (unobserved / padding), shown light grey.  All
    panels are cropped to the valid interior (``padding``); the basemap extent
    is derived from the patch centre (patches sit on a square-degree grid).

    Each sample dict needs:
        accum_t, target (next_mask), pred, loss_mask, padding, fire_id, dt, f1
        lon, lat:  patch-centre coordinates (deg) — sets the basemap extent
        threshold: nf_equiv decision threshold
        name:      row label (e.g. ``"LA 2025\\nPalisades"``)

    Args:
        samples: list of sample dicts (see above).
        title:   optional suptitle for the figure.
        out_path: if given, save PDF+PNG here.
    """
    import cartopy.crs as ccrs

    from firecomp.core.metrics import Metrics
    from firecomp.core.plotting import add_satellite_basemap

    TP_C = (0.15, 0.65, 0.15)   # green
    FP_C = (0.85, 0.15, 0.15)   # red
    FN_C = (0.15, 0.35, 0.85)   # blue
    n_rows = len(samples)
    n_cols = 4
    col_titles = ["accum_t + satellite", "new-fire truth",
                  "new-fire probability", "TP / FP / FN"]
    geo = ccrs.PlateCarree()

    fig = plt.figure(figsize=(3.4 * n_cols, 3.0 * n_rows + 0.7))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.08, wspace=0.18,
                          left=0.05, right=0.97, top=0.93, bottom=0.01)

    for ri, s in enumerate(samples):
        accum = s["accum_t"].astype(np.float32)
        gt, pred = s["target"], s["pred"]
        threshold = s["threshold"]
        mask = s.get("loss_mask")
        p = s.get("padding", 0)
        lon, lat = float(s["lon"]), float(s["lat"])

        if p > 0:
            c = slice(p, -p)
            accum, gt, pred = accum[c, c], gt[c, c], pred[c, c]
            if mask is not None:
                mask = mask[c, c]

        half = (128 - p) * _DEG_PER_PIXEL
        extent = [lon - half, lon + half, lat - half, lat + half]

        # nf_equiv: "new fire" = next-day fire on never-burned pixels.  The
        # burned region is greyed in all result columns; the loss mask
        # (unobserved / padding) is applied only in the TP/FP/FN column.
        burned = accum >= -0.5
        show = ~burned                      # cols 1-2: new-fire region only
        excluded = burned.copy()            # col 3: also drop unobserved
        if mask is not None:
            excluded |= mask < 0.5
        nf_valid = ~excluded

        # ── Col 0: accum_t (fire-spread progression) over satellite basemap ──
        ax0 = fig.add_subplot(gs[ri, 0], projection=geo)
        add_satellite_basemap(ax0, extent)
        accum_disp = np.where(burned, accum, np.nan)
        cmap_a = mpl.colormaps["YlOrRd"].copy()
        cmap_a.set_bad(alpha=0)
        finite = accum_disp[np.isfinite(accum_disp)]
        if finite.size:
            ax0.imshow(accum_disp, cmap=cmap_a,
                       vmin=float(finite.min()), vmax=float(finite.max()),
                       extent=extent, origin="upper", transform=geo,
                       interpolation="nearest", alpha=0.8, zorder=5)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.text(-0.06, 0.5, s.get("name", f"fire {s['fire_id']}\n{s['dt']}"),
                 transform=ax0.transAxes, rotation=90, va="center", ha="right",
                 fontsize=Style.label_fontsize, fontweight="bold")

        # ── Col 1: new-fire ground truth (burned pixels zeroed) ──
        ax1 = fig.add_subplot(gs[ri, 1])
        gt_disp = np.where(show, gt.astype(np.float32), 0.0)
        ax1.imshow(gt_disp, cmap="Reds", vmin=0, vmax=1,
                   interpolation="nearest")

        # ── Col 2: new-fire probability (burned pixels zeroed) ──
        ax2 = fig.add_subplot(gs[ri, 2])
        pred_disp = np.where(show, pred, 0.0)
        im = ax2.imshow(pred_disp, cmap="viridis", vmin=0, vmax=1,
                        interpolation="nearest")
        cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=Style.label_fontsize - 4)

        # ── Col 3: TP/FP/FN overlay (new-fires basis) ──
        ax3 = fig.add_subplot(gs[ri, 3])
        cm = Metrics.pixel_confusion(pred, gt, mask=nf_valid.astype(np.float32),
                                     threshold=threshold)
        overlay = np.ones((*gt.shape, 3), dtype=np.float32)
        overlay[excluded] = [0.81, 0.81, 0.81]
        overlay[cm["tp"]] = TP_C
        overlay[cm["fp"]] = FP_C
        overlay[cm["fn"]] = FN_C
        ax3.imshow(overlay, interpolation="nearest")
        if "f1" in s:
            ax3.text(0.03, 0.97, f"F1={s['f1']:.3f}", transform=ax3.transAxes,
                     va="top", ha="left", fontsize=Style.label_fontsize - 1,
                     color="white",
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

        for ax in (ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])

        if ri == 0:
            for ax, t in zip((ax0, ax1, ax2), col_titles[:3]):
                ax.set_title(t, fontsize=Style.title_fontsize,
                             fontweight="bold", pad=8)
            # 4th column: colour each label to match its overlay cell.
            _colored_title(ax3, [("TP", TP_C), (" / ", "#333333"),
                                 ("FP", FP_C), (" / ", "#333333"),
                                 ("FN", FN_C)], fontsize=Style.title_fontsize)

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 12  fig:size_and_region_f1 — scatter of per-sample F1 by region & fire size
# ═══════════════════════════════════════════════════════════════════════════

# Short region labels for compact x-axis
REGION_SHORT: dict[str, str] = {
    "Western Europe": "W.Eur",
    "Eastern Europe": "E.Eur",
    "MENA": "MENA",
    "Africa": "Africa",
    "North Asia": "N.Asia",
    "South Asia": "S.Asia",
    "Oceania": "Oceania",
    "North NA": "N.NA",
    "Central NA": "C.NA",
    "South America": "S.Amer",
}

# Size bucket definitions: (label, lo_inclusive, hi_exclusive, marker_size)
SIZE_BUCKETS: list[tuple[str, int, int, int]] = [
    ("< 500",   0,     500,   120),
    ("500–5k",  500,   5000,  210),
    ("5k+",     5000,  10**9, 360),
]

# Palette per bucket
_BUCKET_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def assign_size_bucket(num_fire: int) -> str | None:
    """Return the bucket label for *num_fire*, or None if no bucket."""
    for label, lo, hi, _ in SIZE_BUCKETS:
        if lo <= num_fire < hi:
            return label
    return None


def fig_size_and_region_f1(
    data: list[dict],
    *,
    min_count: int = 10,
    title: str = "Per-pixel F1 by region and fire size",
    out_path: str | None = None,
) -> mpl.figure.Figure:
    """Scatter of per-pixel F1 per (region × fire-size bucket).

    Each point represents the pooled pixel-level F1 for all test
    samples in one (region, bucket) group.  Marker size encodes
    the fire-size bucket.  Regions are sorted left-to-right by
    their highest F1 across buckets (best region rightmost).

    Groups with fewer than *min_count* samples are omitted.

    Args:
        data: list of dicts, each with keys:
            region:       str — full region name
            f1:           float — pooled pixel-level F1 for this group
            size_bucket:  str — one of ``SIZE_BUCKETS`` labels
            n_samples:    int — number of samples in the group
        min_count: minimum samples for a group to appear.
        title:     figure title.
        out_path:  stem for output files (no extension).

    Returns:
        matplotlib Figure.
    """
    bucket_labels = [b[0] for b in SIZE_BUCKETS]
    bucket_sizes = {b[0]: b[3] for b in SIZE_BUCKETS}
    bucket_colors = {b[0]: c for b, c in zip(SIZE_BUCKETS, _BUCKET_COLORS)}
    n_buckets = len(bucket_labels)

    # ── index by (region, bucket) ──
    lookup: dict[tuple[str, str], dict] = {}
    for d in data:
        key = (d["region"], d["size_bucket"])
        if d.get("n_samples", min_count) >= min_count:
            lookup[key] = d

    # Regions that have at least one qualifying group
    region_set = {r for r, _ in lookup}
    if not region_set:
        region_set = {d["region"] for d in data}

    # Sort by highest F1 across buckets (ascending → best rightmost)
    def _best_f1(region):
        return max((lookup[(region, b)]["f1"]
                     for b in bucket_labels if (region, b) in lookup),
                    default=0.0)
    region_order = sorted(region_set, key=_best_f1)

    n_regions = len(region_order)

    fig, ax = plt.subplots(figsize=(max(8, n_regions * 1.2), 5))

    for bi, blabel in enumerate(bucket_labels):
        ms = bucket_sizes[blabel]
        color = bucket_colors[blabel]
        plotted = False

        for ri, region in enumerate(region_order):
            entry = lookup.get((region, blabel))
            if entry is None:
                continue

            label = blabel if not plotted else None
            ax.scatter(ri, entry["f1"],
                       s=ms, color=color, alpha=0.8,
                       edgecolors="white", linewidths=0.5,
                       label=label, zorder=3)
            plotted = True

    # ── axes ──
    short_labels = [REGION_SHORT.get(r, r) for r in region_order]
    ax.set_xticks(range(n_regions))
    ax.set_xticklabels(short_labels, rotation=35, ha="right",
                       fontsize=Style.label_fontsize - 1)
    ax.set_ylabel("F1", fontsize=Style.label_fontsize)
    ax.set_xlim(-0.5, n_regions - 0.5)
    ax.grid(axis="y", alpha=Style.grid_alpha)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="num_fire", fontsize=Style.label_fontsize - 2,
                  title_fontsize=Style.label_fontsize - 1,
                  framealpha=Style.legend_framealpha,
                  loc="lower right", markerscale=1.0)

    fig.tight_layout()

    if out_path is not None:
        save_figure(fig, out_path, formats=("pdf", "png"))
    return fig
