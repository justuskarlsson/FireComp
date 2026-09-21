"""
next_day/tables.py — Paper 1 analysis tables (LaTeX).

One function per table.  Each is registered via @table("key") and can be
invoked individually or in bulk from the CLI.

Transfer learning tables  (need --transfer-dir with transfer_matrix_*.json)
========================
  matrix-nm       Train × eval region F1 matrix (next_mask)
  matrix-nf       Train × eval region F1 matrix (new_fires)
  transfer-summary  Compact nm vs nf comparison per training region

LOO tables  (need --transfer-dir with loo_results_*.json)
==========
  loo             Leave-one-out data pollution: per-region Δ vs global model

Grid search tables  (need --grid / eval_results.json)
==================
  grid-models     Best config per model — F1, Prec, Recall, IoU, Brier
  grid-losses     Best config per loss type — F1, Prec, Recall, IoU, Brier
  grid-targets    Target comparison (next_mask vs new_fires via nf_equiv)
  grid-regions    Per-region F1 for the best model per target
  grid-all        Full grid — every run, all metrics (supplementary)

Baseline tables  (need --baselines / baselines directory)
===============
  baseline-summary   All baselines vs best DL: F1, Prec, Recall, IoU, Brier
  baseline-regions   Per-region F1 for each baseline

Fire type tables  (need --fire-type / fire_type_results.json)
================
  fire-type-perf     Per-type difficulty: F1 by fire type (veg/static/crop)
  fire-type-train    Training composition effect on per-type performance
  fire-type-detail   Full breakdown: train × eval type × metrics (supplementary)

Ablation tables  (need --ablation / ablation_eval_results.json)
===============
  ablation           Feature group importance: baseline vs each exclusion

Dataset tables  (need --dataset-dir)
==============
  split-summary      Train / val / test: n, %, date range
  region-split       Sample counts per region × split
  region-year        Sample counts per region × year (all splits)
  fire-type-region   Fire type × region counts (all splits)

Morphology tables  (need --solidity / solidity.json)
=================
  solidity        Fire solidity per region (median, IQR, component count)

Usage:
    python -m firecomp.next_day.tables --list
    python -m firecomp.next_day.tables --tables split-summary region-split region-year
    python -m firecomp.next_day.tables --tables grid-models grid-losses
    python -m firecomp.next_day.tables --tables baseline-summary baseline-regions
    python -m firecomp.next_day.tables --tables fire-type-perf fire-type-train
    python -m firecomp.next_day.tables --tables ablation
    python -m firecomp.next_day.tables                       # all tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from firecomp.next_day.latex import tex_escape_text


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------

# Each entry: (fn, needs: set of data source tags)
# Possible tags: "results" (transfer), "dataset", "grid", "solidity"
_TABLES: dict[str, tuple[Callable, set[str]]] = {}


def table(key: str, *, needs: set[str] | None = None):
    """Register a table generator.

    Functions receive ``(results, ds, grid, solidity=None)`` — any may be
    None depending on declared *needs*.
    """
    def decorator(fn: Callable) -> Callable:
        _TABLES[key] = (fn, needs or set())
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Constants — region ordering and short names
# ---------------------------------------------------------------------------

# Regions that appear as eval columns (have test-set samples).
# Order matches wildfire_regions.json (by registry ID).
EVAL_REGIONS = [
    "Western Europe", "MENA", "Africa", "North Asia", "South Asia",
    "Oceania", "North NA", "Central NA", "South America", "Eastern Europe",
]

# All trainable regions (registry ID order).
TRAIN_REGIONS = [
    "Western Europe", "MENA", "Africa", "North Asia", "South Asia",
    "Oceania", "North NA", "Central NA", "South America", "Eastern Europe",
]

SHORT = {
    "Western Europe": "W.~Eur",
    "MENA":         "MENA",
    "Africa":         "Afr",
    "North Asia":     "N.~Asia",
    "South Asia":     "S.~Asia",
    "Oceania":        "Oce",
    "North NA":       "N.~NA",
    "Central NA":     "C.~NA",
    "South America":  "S.~Amer",
    "Eastern Europe": "E.~Eur",
    "global":         "Global",
}

# Canonical display for model names in LaTeX
MODEL_DISPLAY = {
    "unet":   "UNet",
    "unet++": "UNet++",
    "vit":    "ViT",
}

# Canonical display for loss names in LaTeX
LOSS_DISPLAY = {
    "bce":   "BCE",
    "focal": "Focal",
}

# Canonical display for target names in LaTeX
TARGET_DISPLAY = {
    "next_mask":  tex_escape_text("next_mask"),
    "new_fires":  tex_escape_text("new_fires"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset tables
# ═══════════════════════════════════════════════════════════════════════════


# Static literature values for the dataset comparison table.
# Each entry: (name, resolution, coverage, task, samples_str)
_DATASET_COMPARISON = [
    ("Next Day Wildfire",  "1\\,km",   "US",       "Classification", "18,545"),
    ("WildfireSpreadTS",   "375\\,m",  "US",       "Segmentation",   "607"),
    ("WildfireDB",         "0.25°",    "US",       "Forecasting",    "17,820,835*"),
    ("CFSDS",              "30\\,m",   "Canada",   "Segmentation",   "3,269"),
    ("FireCube",           "0.25°",    "Greece",   "Forecasting",    "--"),
    ("SeasFire Cube",      "0.25°",    "Global",   "Forecasting",    "--"),
    ("\\textbf{FireComp}", "375\\,m",  "Global",   "Segmentation",   "\\textbf{130,649}"),
]


@table("datasets")
def table_datasets(results, ds, grid, **kw) -> str:
    """Related-work dataset comparison table.

    Static table — all values are from the literature.  FireComp (ours) is
    highlighted.  The Samples column disambiguates datacube approaches
    (marked '--') from patch/event-based datasets.
    """
    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:datasets}")
    L.append(r"\begin{tabular}{l l l l r}")
    L.append(r"\toprule")
    L.append(r"Dataset & Resolution & Coverage & Task & Samples \\")
    L.append(r"\midrule")

    for name, res, cov, task, samples in _DATASET_COMPARISON:
        L.append(f"{name} & {res} & {cov} & {task} & {samples} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("split-summary", needs={"dataset"})
def table_split_summary(results, ds, grid, **kw) -> str:
    """Train / val / test: n samples, percentage, date range.

    Small table showing the temporal split strategy and how many samples
    end up in each partition.
    """
    from firecomp.core.regions import Regions

    regions = Regions()
    splits = {
        "Train": ds.train_samples,
        "Val":   ds.val_samples,
        "Test":  ds.test_samples,
    }
    total = sum(len(s) for s in splits.values())

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:split-summary}")
    L.append(r"\begin{tabular}{l r r l l}")
    L.append(r"\toprule")
    L.append(r"Split & Samples & \% & Start date & End date \\")
    L.append(r"\midrule")

    for name, samples in splits.items():
        n = len(samples)
        pct = 100.0 * n / total if total > 0 else 0
        dates = sorted(s.dt for s in samples)
        d0 = dates[0] if dates else "--"
        d1 = dates[-1] if dates else "--"
        L.append(f"{name} & {_fmt_n(n)} & {pct:.0f}\\% & {d0} & {d1} \\\\")

    L.append(r"\midrule")
    L.append(f"Total & {_fmt_n(total)} & 100\\% & & \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("region-split", needs={"dataset"})
def table_region_split(results, ds, grid, **kw) -> str:
    """Sample counts per region for train, val, and test splits.

    One row per region, columns for train / val / test / total.
    """
    from firecomp.core.regions import Regions

    regions = Regions()
    splits = {
        "Train": ds.train_samples,
        "Val":   ds.val_samples,
        "Test":  ds.test_samples,
    }

    # Count samples per region per split
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, samples in splits.items():
        for s in samples:
            rname = regions.id_to_name(s.region_id) if s.region_id > 0 else "Other"
            counts[rname][split_name] += 1

    # Sort regions by total count descending
    region_names = sorted(
        counts.keys(),
        key=lambda r: sum(counts[r].values()),
        reverse=True,
    )

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:region-split}")
    L.append(r"\begin{tabular}{l r r r r}")
    L.append(r"\toprule")
    L.append(r"Region & Train & Val & Test & Total \\")
    L.append(r"\midrule")

    totals = {"Train": 0, "Val": 0, "Test": 0}
    for rname in region_names:
        train_n = counts[rname]["Train"]
        val_n = counts[rname]["Val"]
        test_n = counts[rname]["Test"]
        row_total = train_n + val_n + test_n
        totals["Train"] += train_n
        totals["Val"] += val_n
        totals["Test"] += test_n
        short = SHORT.get(rname, rname)
        L.append(f"{short} & {_fmt_n(train_n)} & {_fmt_n(val_n)} & {_fmt_n(test_n)} "
                 f"& {_fmt_n(row_total)} \\\\")

    grand = sum(totals.values())
    L.append(r"\midrule")
    L.append(f"Total & {_fmt_n(totals['Train'])} & {_fmt_n(totals['Val'])} "
             f"& {_fmt_n(totals['Test'])} & {_fmt_n(grand)} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("region-year", needs={"dataset"})
def table_region_year(results, ds, grid, **kw) -> str:
    """Sample counts per region × year (all splits combined).

    2-D table: rows = regions, columns = years.  Shows geographic and
    temporal coverage of the dataset.
    """
    from firecomp.core.regions import Regions

    regions = Regions()
    all_samples = ds.train_samples + ds.val_samples + ds.test_samples

    # Count per (region, year)
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        rname = regions.id_to_name(s.region_id) if s.region_id > 0 else "Other"
        year = int(s.dt[:4])
        counts[rname][year] += 1

    # Determine year range
    all_years = sorted({int(s.dt[:4]) for s in all_samples})
    if not all_years:
        return "% No samples"

    # Sort regions by total count descending
    region_names = sorted(
        counts.keys(),
        key=lambda r: sum(counts[r].values()),
        reverse=True,
    )

    n_years = len(all_years)

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:region-year}")
    L.append(f"\\begin{{tabular}}{{l{'r' * n_years} r}}")
    L.append(r"\toprule")

    # Header
    year_hdrs = " & ".join(str(y) for y in all_years)
    L.append(f"Region & {year_hdrs} & Total \\\\")
    L.append(r"\midrule")

    # Data rows
    year_totals: dict[int, int] = defaultdict(int)
    for rname in region_names:
        short = SHORT.get(rname, rname)
        cells = []
        row_total = 0
        for y in all_years:
            n = counts[rname][y]
            cells.append(_fmt_n(n) if n > 0 else "--")
            year_totals[y] += n
            row_total += n
        L.append(f"{short} & " + " & ".join(cells) + f" & {_fmt_n(row_total)} \\\\")

    # Totals row
    L.append(r"\midrule")
    total_cells = [_fmt_n(year_totals[y]) for y in all_years]
    grand = sum(year_totals.values())
    L.append(f"Total & " + " & ".join(total_cells) + f" & {_fmt_n(grand)} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


FIRE_TYPE_NAMES = {0: "Veg", 1: "Static", 2: "Crop", -1: "Unknown"}
FIRE_TYPE_ORDER = [0, 1, 2]  # display order


@table("fire-type-region", needs={"dataset"})
def table_fire_type_region(results, ds, grid, **kw) -> str:
    """Fire type × region sample counts (all splits).

    Shows how fire types distribute across regions — e.g. Africa has
    many crop fires, N. NA is almost entirely vegetation.
    """
    from firecomp.core.regions import Regions

    regions = Regions()
    all_samples = ds.train_samples + ds.val_samples + ds.test_samples

    # Count per (region, fire_type)
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for s in all_samples:
        rname = regions.id_to_name(s.region_id) if s.region_id > 0 else "Other"
        counts[rname][s.fire_type] += 1

    # Sort regions by total count descending
    region_names = sorted(
        counts.keys(),
        key=lambda r: sum(counts[r].values()),
        reverse=True,
    )

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:fire-type-region}")
    L.append(r"\begin{tabular}{l r r r r}")
    L.append(r"\toprule")
    type_hdrs = " & ".join(FIRE_TYPE_NAMES[t] for t in FIRE_TYPE_ORDER)
    L.append(f"Region & {type_hdrs} & Total \\\\")
    L.append(r"\midrule")

    type_totals: dict[int, int] = defaultdict(int)
    for rname in region_names:
        short = SHORT.get(rname, rname)
        cells = []
        row_total = 0
        for ft in FIRE_TYPE_ORDER:
            n = counts[rname][ft]
            cells.append(_fmt_n(n) if n > 0 else "--")
            type_totals[ft] += n
            row_total += n
        L.append(f"{short} & " + " & ".join(cells) + f" & {_fmt_n(row_total)} \\\\")

    L.append(r"\midrule")
    tot_cells = [_fmt_n(type_totals[ft]) for ft in FIRE_TYPE_ORDER]
    grand = sum(type_totals.values())
    L.append(f"Total & " + " & ".join(tot_cells) + f" & {_fmt_n(grand)} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# Grid search tables
# ═══════════════════════════════════════════════════════════════════════════


def _best_per_group(
    grid: list[dict],
    group_key: str,
    target: str | None = None,
) -> dict[str, dict]:
    """Return the best run (by test_f1) per unique value of *group_key*.

    If *target* is given, filter to that target first.
    """
    subset = grid if target is None else [r for r in grid if r["target"] == target]
    best: dict[str, dict] = {}
    for r in subset:
        g = r[group_key]
        if g not in best or r["test_f1"] > best[g]["test_f1"]:
            best[g] = r
    return best


def _grid_metrics_row(r: dict) -> list[str]:
    """Return [F1, Prec, Recall, IoU] formatted cells for one run."""
    return [
        _fmt(r.get("test_f1")),
        _fmt(r.get("test_precision")),
        _fmt(r.get("test_recall")),
        _fmt(r.get("test_iou")),
    ]

GRID_METRIC_HEADER = r"F1 & Prec & Recall & IoU"


@table("grid-models", needs={"grid"})
def table_grid_models(results, ds, grid, **kw) -> str:
    """Architecture comparison — best config per model per target.

    For each (model, target) combination, pick the run with the highest
    test F1 and show its metrics.  Bold the best F1 per target.
    """
    targets = sorted(set(r["target"] for r in grid))
    models = sorted(set(r["model"] for r in grid))

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:grid-models}")
    L.append(r"\begin{tabular}{l l l " + "c " * 4 + "}")
    L.append(r"\toprule")
    L.append(rf"Model & Target & Loss & {GRID_METRIC_HEADER} \\")
    L.append(r"\midrule")

    for ti, target in enumerate(targets):
        if ti > 0:
            L.append(r"\midrule")
        best_by_model = _best_per_group(grid, "model", target)

        # Find best F1 across models for this target
        best_f1 = max((r["test_f1"] for r in best_by_model.values()), default=0)

        for model in models:
            r = best_by_model.get(model)
            if r is None:
                continue
            cells = _grid_metrics_row(r)
            # Bold F1 if it's the column-best
            if r["test_f1"] == best_f1:
                cells[0] = rf"\textbf{{{cells[0]}}}"
            label = _grid_run_label(r)
            L.append(
                f"{MODEL_DISPLAY.get(model, model)} & "
                f"{TARGET_DISPLAY.get(target, target)} & "
                f"{label} & " +
                " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("grid-losses", needs={"grid"})
def table_grid_losses(results, ds, grid, **kw) -> str:
    """Loss comparison — best config per loss type per target.

    Groups runs by loss type (BCE / Focal), picks the best run for each,
    then shows metrics.  Bold the best F1 per target.
    """
    targets = sorted(set(r["target"] for r in grid))

    # Group by (loss, target) → best run
    grouped: dict[tuple[str, str], dict] = {}
    for r in grid:
        key = (r["loss"], r["target"])
        if key not in grouped or r["test_f1"] > grouped[key]["test_f1"]:
            grouped[key] = r

    losses = sorted(set(r["loss"] for r in grid))

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:grid-losses}")
    L.append(r"\begin{tabular}{l l l " + "c " * 4 + "}")
    L.append(r"\toprule")
    L.append(rf"Loss & Target & Model & {GRID_METRIC_HEADER} \\")
    L.append(r"\midrule")

    for ti, target in enumerate(targets):
        if ti > 0:
            L.append(r"\midrule")
        target_runs = {k: v for k, v in grouped.items() if k[1] == target}
        best_f1 = max((r["test_f1"] for r in target_runs.values()), default=0)

        for loss in losses:
            r = grouped.get((loss, target))
            if r is None:
                continue
            cells = _grid_metrics_row(r)
            if r["test_f1"] == best_f1:
                cells[0] = rf"\textbf{{{cells[0]}}}"
            L.append(
                f"{LOSS_DISPLAY.get(loss, loss)} & "
                f"{TARGET_DISPLAY.get(target, target)} & "
                f"{MODEL_DISPLAY.get(r['model'], r['model'])} & " +
                " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("grid-targets", needs={"grid"})
def table_grid_targets(results, ds, grid, **kw) -> str:
    """Target comparison — next_mask vs new_fires on a common metric basis.

    For each model, shows the best run per target.  When a next_mask run
    has ``nf_equiv_*`` fields, an extra row shows its new_fires-equivalent
    metrics for direct comparison.  Bold the best F1 per metric column.
    """
    models = sorted(set(r["model"] for r in grid))

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:grid-targets}")
    L.append(r"\begin{tabular}{l l " + "c " * 4 + "}")
    L.append(r"\toprule")
    L.append(rf"Model & Target & {GRID_METRIC_HEADER} \\")
    L.append(r"\midrule")

    for mi, model in enumerate(models):
        if mi > 0:
            L.append(r"\midrule")
        for target in ("next_mask", "new_fires"):
            best = _best_per_group(grid, "model", target).get(model)
            if best is None:
                continue
            cells = _grid_metrics_row(best)
            L.append(
                f"{MODEL_DISPLAY.get(model, model)} & "
                f"{TARGET_DISPLAY.get(target, target)} & " +
                " & ".join(cells) + r" \\")

            # nf_equiv row for next_mask runs
            if target == "next_mask" and "nf_equiv_f1" in best:
                eq = {
                    "test_f1":        best["nf_equiv_f1"],
                    "test_precision": best.get("nf_equiv_precision"),
                    "test_recall":    best.get("nf_equiv_recall"),
                    "test_iou":       best.get("nf_equiv_iou"),
                    "test_brier":     best.get("nf_equiv_brier"),
                }
                cells = _grid_metrics_row(eq)
                L.append(
                    f" & \\quad {tex_escape_text('nf_equiv')} & " +
                    " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("grid-regions", needs={"grid"})
def table_grid_regions(results, ds, grid, **kw) -> str:
    """Per-region F1 for the best model per target.

    Picks the single best run per target, then shows F1 per region.
    Bold the highest regional F1 per target.
    """
    targets = sorted(set(r["target"] for r in grid))

    # Best overall run per target
    best_runs: dict[str, dict] = {}
    for r in grid:
        t = r["target"]
        if t not in best_runs or r["test_f1"] > best_runs[t]["test_f1"]:
            best_runs[t] = r

    # Collect all region names across all runs (exclude unknown_0)
    all_regions: set[str] = set()
    for r in best_runs.values():
        all_regions |= {k for k in r.get("by_region", {}) if k != "unknown_0"}

    # Order regions by TRAIN_REGIONS order, then alphabetical for extras
    region_order = [r for r in TRAIN_REGIONS if r in all_regions]
    extras = sorted(all_regions - set(region_order))
    region_order += extras

    n_targets = len(targets)

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:grid-regions}")
    L.append(r"\begin{tabular}{l " + "c " * n_targets + "}")
    L.append(r"\toprule")

    # Header: Region & target1 & target2 ...
    target_hdrs = " & ".join(TARGET_DISPLAY.get(t, t) for t in targets)
    L.append(f"Region & {target_hdrs} \\\\")

    # Sub-header: show which model/loss was best
    sub_parts = []
    for t in targets:
        r = best_runs[t]
        sub_parts.append(
            f"\\scriptsize {MODEL_DISPLAY.get(r['model'], r['model'])}"
            f"/{_grid_run_label(r)}")
    L.append(" & " + " & ".join(sub_parts) + r" \\")
    L.append(r"\midrule")

    # Find best F1 per target for bolding
    col_best: dict[str, float] = {}
    for t in targets:
        by_r = best_runs[t].get("by_region", {})
        vals = [v for k, v in by_r.items() if k != "unknown_0" and v is not None]
        col_best[t] = max(vals) if vals else 0

    for region in region_order:
        short = SHORT.get(region, region)
        cells = []
        for t in targets:
            v = best_runs[t].get("by_region", {}).get(region)
            s = _fmt(v)
            if v is not None and v == col_best[t]:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        L.append(f"{short} & " + " & ".join(cells) + r" \\")

    # Global row
    L.append(r"\midrule")
    global_cells = []
    for t in targets:
        global_cells.append(_fmt(best_runs[t].get("test_f1")))
    L.append(f"Global & " + " & ".join(global_cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("grid-all", needs={"grid"})
def table_grid_all(results, ds, grid, **kw) -> str:
    """Full grid — every run, all metrics.  Supplementary material.

    Sorted by target then descending test_f1.  Shows model, loss config
    tag, threshold, and all test metrics.
    """
    sorted_runs = sorted(grid, key=lambda r: (r["target"], -r["test_f1"]))

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\scriptsize")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:grid-all}")
    L.append(r"\begin{tabular}{l l l c " + "c " * 4 + "}")
    L.append(r"\toprule")
    L.append(rf"Model & Target & Config & $\tau$ & {GRID_METRIC_HEADER} \\")
    L.append(r"\midrule")

    # Find best F1 per target
    target_best: dict[str, float] = {}
    for r in grid:
        t = r["target"]
        if t not in target_best or r["test_f1"] > target_best[t]:
            target_best[t] = r["test_f1"]

    prev_target = None
    for r in sorted_runs:
        target = r["target"]
        if prev_target is not None and target != prev_target:
            L.append(r"\midrule")
        prev_target = target

        cells = _grid_metrics_row(r)
        if r["test_f1"] == target_best.get(target):
            cells[0] = rf"\textbf{{{cells[0]}}}"

        label = _grid_run_label(r)
        threshold = _fmt_tau(r.get("threshold"))

        L.append(
            f"{MODEL_DISPLAY.get(r['model'], r['model'])} & "
            f"{TARGET_DISPLAY.get(target, target)} & "
            f"{label} & "
            f"{threshold} & " +
            " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def _grid_run_label(r: dict) -> str:
    """Short label for a grid run: 'BCE pw10' or 'Focal α.50'."""
    tag = r.get("tag", "")
    loss = r.get("loss", "")

    if loss == "bce":
        # Extract pos_weight from tag like "grid_unet_bce_pw10_newf"
        pw = "?"
        for part in tag.split("_"):
            if part.startswith("pw"):
                pw = part[2:]
                break
        return f"BCE pw{pw}"
    elif loss == "focal":
        # Extract alpha from tag like "grid_unet_focal_a25_newf"
        alpha = "?"
        for part in tag.split("_"):
            if part.startswith("a") and part[1:].isdigit():
                alpha = f".{part[1:]}"
                break
        return rf"Focal $\alpha${alpha}"
    return tag


# ═══════════════════════════════════════════════════════════════════════════
# Transfer learning tables
# ═══════════════════════════════════════════════════════════════════════════


@table("matrix-nm", needs={"matrix"})
def table_transfer_matrix_nm(results, ds, grid, matrix=None, **kw) -> str:
    """Transfer matrix for next_mask target."""
    nm = matrix.get("next_mask", {})
    if not nm:
        return "% No next_mask matrix data"
    return _table_transfer_matrix(nm, "next_mask")


@table("matrix-nf", needs={"matrix"})
def table_transfer_matrix_nf(results, ds, grid, matrix=None, **kw) -> str:
    """Transfer matrix for new_fires-equivalent metrics (from next_mask models).

    Reads the ``nf_equiv`` key produced by ``transfer_matrix_nf_equiv.json``.
    Falls back to ``new_fires`` for legacy compatibility.
    """
    nf = matrix.get("nf_equiv", matrix.get("new_fires", {}))
    if not nf:
        return "% No nf_equiv / new_fires matrix data"
    return _table_transfer_matrix(nf, "nf_equiv")


def _table_transfer_matrix(matrix: dict[str, dict[str, float]], target_type: str) -> str:
    """Train-region × eval-region F1 matrix.

    Rows: each training region + global.
    Columns: each eval region + global F1.
    **Bold**: column-best.  \\underline: self-eval diagonal.
    """
    eval_cols = [r for r in EVAL_REGIONS
                 if any(r in matrix[tl] for tl in matrix)]
    eval_cols.append("global")

    train_rows = [r for r in TRAIN_REGIONS if r in matrix]
    train_rows.append("global")

    # Column-best for bolding
    col_best: dict[str, str] = {}
    for col in eval_cols:
        best_v, best_r = -1.0, ""
        for row in train_rows:
            v = matrix.get(row, {}).get(col)
            if v is not None and v > best_v:
                best_v, best_r = v, row
        col_best[col] = best_r

    n = len(eval_cols)
    label = f"tab:transfer-{target_type.replace('_', '-')}"

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(f"\\label{{{label}}}")
    L.append(f"\\begin{{tabular}}{{l{'c' * n}}}")
    L.append(r"\toprule")

    # Header
    hdrs = " & ".join(SHORT.get(c, c) for c in eval_cols)
    L.append(rf"Train \textbackslash\ Eval & {hdrs} \\")
    L.append(r"\midrule")

    # Data rows
    for row in train_rows:
        if row == "global":
            L.append(r"\midrule")
        cells = []
        for col in eval_cols:
            v = matrix.get(row, {}).get(col)
            s = _fmt(v)
            if v is not None:
                is_diag = (row == col)
                is_best = (col_best.get(col) == row)
                if is_best and is_diag:
                    s = rf"\textbf{{\underline{{{s}}}}}"
                elif is_best:
                    s = rf"\textbf{{{s}}}"
                elif is_diag:
                    s = rf"\underline{{{s}}}"
            cells.append(s)
        L.append(f"{SHORT.get(row, row)} & " + " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("loo", needs={"loo"})
def table_loo(results, ds, grid, loo=None, matrix=None, **kw) -> str:
    """LOO data-pollution analysis: does excluding a region improve scores?

    For each excluded region X, the LOO model was trained AND evaluated on
    all-except-X.  Per-region F1s are compared against the global model
    (from transfer_matrix_*.json) to compute a macro-average Δ across the
    remaining regions.  Positive Δ = excluding X improved performance =
    X was polluting.

    Falls back to just showing LOO global F1 if matrix data is unavailable.
    """
    if not loo:
        return "% No LOO results"

    # Parse LOO results by (exclude_region, target_type)
    loo_by_tt = _loo_scores_from_file(loo)

    # Global model's per-region scores (from matrix JSON, "global" row)
    global_by_tt: dict[str, dict[str, float]] = {}
    if matrix:
        for tt, m in matrix.items():
            if "global" in m:
                global_by_tt[tt] = {
                    k: v for k, v in m["global"].items() if k != "unknown_0"
                }

    # Determine available target types
    target_types = sorted({tt for _, tt in loo_by_tt})
    if not target_types:
        return "% No LOO results parsed"

    # Regions that have LOO entries
    regions = [
        r for r in EVAL_REGIONS
        if any((r, tt) in loo_by_tt for tt in target_types)
    ]

    n_tt = len(target_types)
    # Columns per target: Full, LOO, Δ  (3 cols each)
    col_spec = " c ".join(["ccc"] * n_tt)
    spacer_cols = " c " * (n_tt - 1) if n_tt > 1 else ""

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:loo}")

    if n_tt == 1:
        L.append(r"\begin{tabular}{l ccc}")
    else:
        L.append(r"\begin{tabular}{l ccc c ccc}")
    L.append(r"\toprule")

    # Two-level header
    if n_tt > 1:
        hdr_parts = []
        for i, tt in enumerate(target_types):
            col_start = 2 + i * 4  # account for spacer columns
            hdr_parts.append(
                rf"\multicolumn{{3}}{{c}}{{{tex_escape_text(tt)}}}")
        L.append("& " + " & & ".join(hdr_parts) + r" \\")
        # cmidrules
        rules = []
        for i in range(n_tt):
            start = 2 + i * 4
            rules.append(rf"\cmidrule(lr){{{start}-{start + 2}}}")
        L.append(" ".join(rules))
    else:
        L.append(rf"& \multicolumn{{3}}{{c}}{{{tex_escape_text(target_types[0])}}} \\")
        L.append(r"\cmidrule(lr){2-4}")

    # Sub-header
    sub = "Excluded & " + " & & ".join(["Full & LOO & $\\Delta$"] * n_tt) + r" \\"
    L.append(sub)
    L.append(r"\midrule")

    for region in regions:
        parts = [SHORT.get(region, region)]
        for i, tt in enumerate(target_types):
            loo_entry = loo_by_tt.get((region, tt), {})
            global_scores = global_by_tt.get(tt, {})

            # Macro-average Δ: compare per-region F1s on the same regions
            # (fair because both models are eval'd on the same test samples
            # for each non-excluded region)
            loo_regions = {k: v for k, v in loo_entry.items()
                          if k not in ("global", "unknown_0")}
            if global_scores and loo_regions:
                deltas = []
                for rname, loo_f1 in loo_regions.items():
                    full_f1 = global_scores.get(rname)
                    if full_f1 is not None:
                        deltas.append(loo_f1 - full_f1)
                avg_delta = sum(deltas) / len(deltas) if deltas else None
                full_macro = (sum(global_scores.get(r, 0)
                                  for r in loo_regions) / len(loo_regions)
                              if loo_regions else None)
                loo_macro = (sum(loo_regions.values()) / len(loo_regions)
                             if loo_regions else None)
                parts += [_fmt(full_macro), _fmt(loo_macro),
                          _fmt_delta(avg_delta) if avg_delta is not None else "--"]
            else:
                # No global reference — just show LOO global F1
                loo_f1 = loo_entry.get("global")
                parts += ["--", _fmt(loo_f1), "--"]

            if i < n_tt - 1:
                parts.append("")  # spacer column

        L.append(" & ".join(parts) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("transfer-summary", needs={"matrix"})
def table_transfer_summary(results, ds, grid, matrix=None, **kw) -> str:
    """Compact comparison: next_mask vs new_fires per training region.

    For each training region (+ global), shows:
      All   — overall test F1 across all regions
      Self  — F1 evaluated only on the training region (diagonal)
    """
    nm = matrix.get("next_mask", {})
    nf = matrix.get("nf_equiv", matrix.get("new_fires", {}))

    train_rows = [r for r in TRAIN_REGIONS if r in nm or r in nf]
    train_rows.append("global")

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:transfer-summary}")
    L.append(r"\begin{tabular}{l cc c cc}")
    L.append(r"\toprule")
    L.append(
        r"& \multicolumn{2}{c}{next\_mask} "
        r"& & \multicolumn{2}{c}{nf\_equiv} \\"
    )
    L.append(r"\cmidrule(lr){2-3} \cmidrule(lr){5-6}")
    L.append(r"Train region & All & Self & & All & Self \\")
    L.append(r"\midrule")

    for row in train_rows:
        if row == "global":
            L.append(r"\midrule")
        parts = [SHORT.get(row, row)]
        for data in (nm, nf):
            by_region = data.get(row, {})
            all_f1 = by_region.get("global")
            self_f1 = by_region.get(row) if row != "global" else None
            parts += [_fmt(all_f1), _fmt(self_f1)]
            if data is nm:
                parts.append("")      # spacer
        L.append(" & ".join(parts) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════


def _fmt(v: float | None) -> str:
    """Format F1/Prec/Recall/IoU for tables: 3 dp, no leading zero (``.310``)."""
    if v is None:
        return "--"
    s = f"{v:.3f}"
    if s.startswith("0."):
        return s[1:]          # "0.310" → ".310"
    return s


def _fmt_n(v: float | int | None) -> str:
    """Format a count N as an integer with thousands separators (``130,649``)."""
    if v is None:
        return "--"
    return f"{int(round(float(v))):,}"


def _fmt_tau(v: float | None) -> str:
    """Format a decision threshold τ: 2 dp with leading zero (``0.48``)."""
    if v is None:
        return "--"
    return f"{v:.2f}"


def _fmt_pm(v: float | None, std: float | None) -> str:
    r"""Format value ± std for LaTeX: ``.310${\scriptstyle\pm}$.008``."""
    if v is None:
        return "--"
    base = _fmt(v)
    if std is not None and std > 0:
        s = _fmt(std)
        return rf"{base}${{\scriptstyle\pm}}${s}"
    return base


def _fmt_delta(v: float) -> str:
    """Format delta with explicit sign: +.012 or $-$.034."""
    if v >= 0:
        return "+" + _fmt(v)
    return "$-$" + _fmt(abs(v))


def _bold_row_best(vals: list[float | None]) -> list[str]:
    """Format a row of values, bolding the cell(s) with the maximum value."""
    cells = [_fmt(v) for v in vals]
    present = [v for v in vals if v is not None]
    if not present:
        return cells
    best = max(present)
    return [rf"\textbf{{{c}}}" if v is not None and v == best else c
            for c, v in zip(cells, vals)]


# ═══════════════════════════════════════════════════════════════════════════
# Data extraction helpers (transfer results JSON)
# ═══════════════════════════════════════════════════════════════════════════


def _build_transfer_matrix(
    results: list[dict], target_type: str,
) -> dict[str, dict[str, float]]:
    """Extract {train_label: {eval_region: F1}} for matrix (non-LOO) runs."""
    matrix: dict[str, dict[str, float]] = {}
    for r in results:
        if r["train_label"].startswith("loo_"):
            continue
        if r["target_type"] != target_type:
            continue
        by_region = {k: v for k, v in r["by_region"].items()
                     if k != "unknown_0"}
        matrix[r["train_label"]] = by_region
    return matrix


def _global_scores(results: list[dict]) -> dict[str, dict[str, float]]:
    """Extract {target_type: {region: F1}} for the global model."""
    out: dict[str, dict[str, float]] = {}
    for r in results:
        if r["train_label"] != "global":
            continue
        out[r["target_type"]] = {
            k: v for k, v in r["by_region"].items() if k != "unknown_0"
        }
    return out


def _loo_scores(
    results: list[dict],
) -> dict[tuple[str, str], dict[str, float]]:
    """Extract {(held_out, target_type): {region: F1}} for LOO runs
    from all_results.json (legacy)."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for r in results:
        if not r["train_label"].startswith("loo_"):
            continue
        held_out = r["train_label"][4:]       # strip "loo_"
        by_region = {k: v for k, v in r["by_region"].items()
                     if k != "unknown_0"}
        out[(held_out, r["target_type"])] = by_region
    return out


def _loo_scores_from_file(
    loo_data: dict[str, list[dict]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Extract {(exclude_region, target_type): {region: F1}} from LOO results.

    Args:
        loo_data: ``{target_type: [entries]}`` as returned by ``_load_loo_dir``.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for tt, entries in loo_data.items():
        for r in entries:
            excluded = r.get("exclude_region")
            if excluded is None:
                continue
            by_region = {k: v for k, v in r.get("by_region", {}).items()
                         if k != "unknown_0"}
            out[(excluded, tt)] = by_region
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Solidity / morphology tables
# ═══════════════════════════════════════════════════════════════════════════


@table("solidity", needs={"solidity"})
def table_solidity(results, ds, grid, solidity=None, **kw) -> str:
    """Fire solidity per region — median, IQR, mean component count.

    Shows how cohesive fire shapes are across regions.  Lower solidity
    means more scattered / fractal fire patterns.
    """
    if not solidity:
        return "% solidity data not loaded"

    # Group by region
    by_region: dict[str, list[dict]] = defaultdict(list)
    for entry in solidity:
        by_region[entry["region_name"]].append(entry)

    # Sort regions by median solidity ascending (most scattered first)
    region_order = sorted(
        by_region.keys(),
        key=lambda r: float(np.median([e["solidity"] for e in by_region[r]])),
    )

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:solidity}")
    L.append(r"\begin{tabular}{l r r r r r}")
    L.append(r"\toprule")
    L.append(r"Region & $N$ & Median & Q1 & Q3 & Comp. \\")
    L.append(r"\midrule")

    for rname in region_order:
        entries = by_region[rname]
        sols = np.array([e["solidity"] for e in entries])
        comps = np.array([e["n_components"] for e in entries])
        short = SHORT.get(rname, rname)
        q1, med, q3 = np.percentile(sols, [25, 50, 75])
        med_comp = np.median(comps)
        L.append(
            f"{short} & {_fmt_n(len(entries))} & {med:.3f} & {q1:.3f} & {q3:.3f} "
            f"& {_fmt_n(med_comp)} \\\\"
        )

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline tables
# ═══════════════════════════════════════════════════════════════════════════


# Display names for baselines in LaTeX
BASELINE_DISPLAY = {
    "persistence":   "Persistence",
    "morphological": "Morph.~dilation",
}


@table("baseline-summary", needs={"baselines"})
def table_baseline_summary(results, ds, grid, baselines=None, **kw) -> str:
    """All baselines side by side: F1, Precision, Recall, IoU, Brier.

    One row per (baseline, target).  If grid search results are also loaded,
    appends the best DL model per target for direct comparison.

    Main paper table — demonstrates that learned models significantly
    outperform simple baselines.
    """
    rows = _sort_baselines(baselines)

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:baseline-summary}")
    L.append(r"\begin{tabular}{l l " + "c " * 4 + "}")
    L.append(r"\toprule")
    L.append(rf"Method & Target & {GRID_METRIC_HEADER} \\")
    L.append(r"\midrule")

    # Find best F1 per target for bolding
    target_best: dict[str, float] = {}
    for r in rows:
        t = r.get("target", "")
        f1 = r.get("test_f1", 0)
        if t not in target_best or f1 > target_best[t]:
            target_best[t] = f1
    # Also include grid best if available
    if grid:
        for r in grid:
            t = r.get("target", "")
            f1 = r.get("test_f1", 0)
            if t not in target_best or f1 > target_best[t]:
                target_best[t] = f1

    prev_target = None
    for r in rows:
        target = r.get("target", "")
        name = r.get("baseline", "")

        if prev_target is not None and target != prev_target:
            L.append(r"\midrule")
        prev_target = target
        display = BASELINE_DISPLAY.get(name, name)
        if name == "morphological":
            radius = r.get("best_radius", "?")
            display += f" ($r$={radius})"

        cells = _grid_metrics_row(r)
        L.append(
            f"{display} & "
            f"{TARGET_DISPLAY.get(target, target)} & " +
            " & ".join(cells) + r" \\")

    # Append best DL model per target (if grid results available)
    if grid:
        L.append(r"\midrule")
        for target in sorted(target_best.keys()):
            best = _best_per_group(grid, "model", target)
            if not best:
                continue
            top = max(best.values(), key=lambda r: r["test_f1"])
            cells = _grid_metrics_row(top)
            if top["test_f1"] == target_best.get(target):
                cells[0] = rf"\textbf{{{cells[0]}}}"
            model_name = MODEL_DISPLAY.get(top["model"], top["model"])
            label = _grid_run_label(top)
            L.append(
                f"{model_name} ({label}) & "
                f"{TARGET_DISPLAY.get(target, target)} & " +
                " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("baseline-regions", needs={"baselines"})
def table_baseline_regions(results, ds, grid, baselines=None, **kw) -> str:
    """Per-region F1 for each baseline — next_mask and nf_equiv side by side.

    Rows = regions, columns = baseline methods grouped by scoring metric.
    Includes best DL model for comparison when grid results are available.
    """
    # Filter to next_mask baselines
    nm_baselines = [r for r in baselines if r.get("target") == "next_mask"]
    if not nm_baselines:
        nm_baselines = baselines

    # Collect all regions across all baselines
    all_regions: set[str] = set()
    for r in nm_baselines:
        all_regions |= {k for k in r.get("by_region", {}) if k != "unknown_0"}

    region_order = [r for r in EVAL_REGIONS if r in all_regions]
    extras = sorted(all_regions - set(region_order))
    region_order += extras

    if not region_order:
        return "% No regional data in baselines"

    # Build column groups: each method has (label, run_dict)
    methods: list[tuple[str, dict]] = []
    for r in _sort_baselines(nm_baselines):
        name = r.get("baseline", "")
        if name == "morphological":
            display = f"Morph ($r$={r.get('best_radius', '?')})"
        else:
            display = BASELINE_DISPLAY.get(name, name)
        methods.append((display, r))

    # Add best DL model if grid available
    if grid:
        best_nm = _best_per_group(grid, "model", "next_mask")
        if best_nm:
            top = max(best_nm.values(), key=lambda r: r["test_f1"])
            model_name = MODEL_DISPLAY.get(top["model"], top["model"])
            methods.append((f"{model_name} (DL)", top))

    # Check if any method has nf_equiv data
    has_nf = any(run.get("nf_equiv_by_region") for _, run in methods)

    n_methods = len(methods)

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:baseline-regions}")

    if has_nf:
        # Two column groups: next_mask and nf_equiv
        L.append(r"\begin{tabular}{l " + "c " * n_methods + " c " +
                 "c " * n_methods + "}")
        L.append(r"\toprule")
        L.append(
            rf"& \multicolumn{{{n_methods}}}{{c}}{{next\_mask F1}} "
            rf"& & \multicolumn{{{n_methods}}}{{c}}{{nf\_equiv F1}} \\")
        L.append(
            rf"\cmidrule(lr){{2-{1 + n_methods}}} "
            rf"\cmidrule(lr){{{1 + n_methods + 2}-{1 + 2 * n_methods + 1}}}")
        hdr = " & ".join(label for label, _ in methods)
        L.append(f"Region & {hdr} & & {hdr} \\\\")
    else:
        # Single column group: next_mask only
        L.append(f"\\begin{{tabular}}{{l {'c ' * n_methods}}}")
        L.append(r"\toprule")
        hdr = " & ".join(label for label, _ in methods)
        L.append(f"Region & {hdr} \\\\")
    L.append(r"\midrule")

    for region in region_order:
        short = SHORT.get(region, region)
        nm_vals = [run.get("by_region", {}).get(region) for _, run in methods]
        nm_cells = _bold_row_best(nm_vals)
        if has_nf:
            nf_vals = [run.get("nf_equiv_by_region", {}).get(region)
                       for _, run in methods]
            nf_cells = _bold_row_best(nf_vals)
            L.append(f"{short} & " + " & ".join(nm_cells) +
                     " & & " + " & ".join(nf_cells) + r" \\")
        else:
            L.append(f"{short} & " + " & ".join(nm_cells) + r" \\")

    # Global row
    L.append(r"\midrule")
    nm_global_vals = [run.get("test_f1") for _, run in methods]
    nm_globals = _bold_row_best(nm_global_vals)
    if has_nf:
        nf_global_vals = [run.get("nf_equiv_f1") for _, run in methods]
        nf_globals = _bold_row_best(nf_global_vals)
        L.append(f"Global & " + " & ".join(nm_globals) +
                 " & & " + " & ".join(nf_globals) + r" \\")
    else:
        L.append(f"Global & " + " & ".join(nm_globals) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def _sort_baselines(baselines: list[dict]) -> list[dict]:
    """Sort baselines: persistence first, then morphological, grouped by target."""
    order = {"persistence": 0, "morphological": 1}
    target_order = {"next_mask": 0, "new_fires": 1}
    return sorted(baselines, key=lambda r: (
        target_order.get(r.get("target", ""), 9),
        order.get(r.get("baseline", ""), 9),
    ))


# ═══════════════════════════════════════════════════════════════════════════
# Fire type tables
# ═══════════════════════════════════════════════════════════════════════════


# Display names for fire types in LaTeX
FIRE_TYPE_DISPLAY = {
    "vegetation": "Vegetation",
    "static":     "Static",
    "crop":       "Crop",
}
FIRE_TYPE_EVAL_ORDER = ["vegetation", "static", "crop"]

# Display names for training configs
TRAIN_FT_DISPLAY = {
    "vegetation": "Veg.~only",
    "all":        "All types",
    "static":     "Static only",
    "crop":       "Crop only",
}


@table("fire-type-perf", needs={"fire_type"})
def table_fire_type_perf(results, ds, grid, fire_type=None, **kw) -> str:
    """Per-fire-type prediction difficulty.

    For each fire type (vegetation, static, crop), shows F1, Precision,
    Recall for both next_mask and nf_equiv metrics.  Uses the model trained
    on all fire types (ft_all) for an unbiased comparison.

    Main paper table — demonstrates that static fires are trivially
    predictable (high persistence), while crop fires are hardest.
    """
    # Prefer ft_all; fall back to first available
    run = _find_ft_run(fire_type, prefer="all")
    if run is None:
        return "% No fire type study results"

    train_label = TRAIN_FT_DISPLAY.get(
        run.get("train_fire_type", ""), run.get("tag", "?"))
    by_ft = run.get("by_fire_type", {})
    g = run.get("global", {})

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:fire-type-perf}")
    L.append(r"\begin{tabular}{l r ccc c ccc}")
    L.append(r"\toprule")
    L.append(
        r"& & \multicolumn{3}{c}{next\_mask} "
        r"& & \multicolumn{3}{c}{new\_fires equiv.} \\"
    )
    L.append(r"\cmidrule(lr){3-5} \cmidrule(lr){7-9}")
    L.append(r"Fire type & $N$ & F1 & Prec & Recall "
             r"& & F1 & Prec & Recall \\")
    L.append(r"\midrule")

    # Find best nm F1 per fire type for bolding
    nm_best = max((by_ft.get(ft, {}).get("f1", 0)
                   for ft in FIRE_TYPE_EVAL_ORDER), default=0)

    def _fp(d, key):
        """Format value with optional ±std from aggregated results."""
        return _fmt_pm(d.get(key), d.get(f"{key}_std"))

    for ft in FIRE_TYPE_EVAL_ORDER:
        d = by_ft.get(ft, {})
        if not d:
            continue
        display = FIRE_TYPE_DISPLAY.get(ft, ft)
        n = d.get("n", 0)
        nm_f1 = _fp(d, "f1")
        if d.get("f1") == nm_best and nm_best > 0:
            nm_f1 = rf"\textbf{{{nm_f1}}}"
        L.append(
            f"{display} & {_fmt_n(n)} & "
            f"{nm_f1} & {_fp(d, 'precision')} & {_fp(d, 'recall')} & & "
            f"{_fp(d, 'nf_equiv_f1')} & {_fp(d, 'nf_equiv_precision')} & "
            f"{_fp(d, 'nf_equiv_recall')} \\\\")

    # Global row
    L.append(r"\midrule")
    n_total = sum(by_ft.get(ft, {}).get("n", 0) for ft in FIRE_TYPE_EVAL_ORDER)
    L.append(
        f"All & {_fmt_n(n_total)} & "
        f"{_fp(g, 'f1')} & {_fp(g, 'precision')} & "
        f"{_fp(g, 'recall')} & & "
        f"{_fp(g, 'nf_equiv_f1')} & {_fp(g, 'nf_equiv_precision')} & "
        f"{_fp(g, 'nf_equiv_recall')} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("fire-type-train", needs={"fire_type"})
def table_fire_type_train(results, ds, grid, fire_type=None, **kw) -> str:
    """Training composition effect: does including non-vegetation fires help?

    Rows = training configs (ft_veg, ft_all, etc.).  Columns = nf_equiv F1
    evaluated globally and per fire type.  Compact single-column table.

    Main paper table — shows whether including static/crop fires in
    training helps or hurts vegetation fire prediction.
    """
    if not fire_type:
        return "% No fire type study results"

    runs = fire_type

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:fire-type-train}")

    # Columns: train config | Global | per-fire-type nf_equiv F1s
    n_ft = len(FIRE_TYPE_EVAL_ORDER)
    ft_hdrs = " & ".join(FIRE_TYPE_DISPLAY.get(ft, ft)
                         for ft in FIRE_TYPE_EVAL_ORDER)
    L.append(r"\begin{tabular}{l " + "c " * (1 + n_ft) + "}")
    L.append(r"\toprule")
    L.append(f"Train data & Global & {ft_hdrs} \\\\")
    L.append(r"\midrule")

    # Find column-best nf_equiv F1 for bolding
    col_best: dict[str, float] = {"global": 0}
    for ft in FIRE_TYPE_EVAL_ORDER:
        col_best[ft] = 0
    for run in runs:
        g = run.get("global", {})
        by_ft = run.get("by_fire_type", {})
        v = g.get("nf_equiv_f1", 0)
        if v > col_best["global"]:
            col_best["global"] = v
        for ft in FIRE_TYPE_EVAL_ORDER:
            v = by_ft.get(ft, {}).get("nf_equiv_f1", 0)
            if v > col_best.get(ft, 0):
                col_best[ft] = v

    for run in runs:
        train_ft = run.get("train_fire_type", run.get("tag", "?"))
        display = TRAIN_FT_DISPLAY.get(train_ft, train_ft)
        g = run.get("global", {})
        by_ft = run.get("by_fire_type", {})

        cells = []
        v = g.get("nf_equiv_f1")
        s = _fmt_pm(v, g.get("nf_equiv_f1_std"))
        if v is not None and v == col_best["global"]:
            s = rf"\textbf{{{s}}}"
        cells.append(s)
        for ft in FIRE_TYPE_EVAL_ORDER:
            d = by_ft.get(ft, {})
            v = d.get("nf_equiv_f1")
            s = _fmt_pm(v, d.get("nf_equiv_f1_std"))
            if v is not None and v == col_best.get(ft):
                s = rf"\textbf{{{s}}}"
            cells.append(s)

        L.append(f"{display} & " + " & ".join(cells) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


@table("fire-type-detail", needs={"fire_type"})
def table_fire_type_detail(results, ds, grid, fire_type=None, **kw) -> str:
    """Full fire type breakdown — train config × eval type × all metrics.

    Supplementary table.  For each training config, shows full metrics
    (F1, Prec, Recall, IoU, Brier) per fire type for next_mask.
    """
    if not fire_type:
        return "% No fire type study results"

    runs = fire_type

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\scriptsize")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:fire-type-detail}")
    L.append(r"\begin{tabular}{l l r " + "c " * 4 + "c " + "c " * 3 + "}")
    L.append(r"\toprule")
    L.append(
        r"& & & \multicolumn{4}{c}{next\_mask} "
        r"& & \multicolumn{3}{c}{new\_fires equiv.} \\"
    )
    L.append(r"\cmidrule(lr){4-7} \cmidrule(lr){9-11}")
    L.append(r"Train data & Fire type & $N$ & F1 & Prec & Recall & IoU "
             r"& & F1 & Prec & Recall \\")
    L.append(r"\midrule")

    for ri, run in enumerate(runs):
        if ri > 0:
            L.append(r"\midrule")
        train_ft = run.get("train_fire_type", run.get("tag", "?"))
        display = TRAIN_FT_DISPLAY.get(train_ft, train_ft)
        by_ft = run.get("by_fire_type", {})
        g = run.get("global", {})

        for fi, ft in enumerate(FIRE_TYPE_EVAL_ORDER):
            d = by_ft.get(ft, {})
            if not d:
                continue
            ft_display = FIRE_TYPE_DISPLAY.get(ft, ft)
            train_col = display if fi == 0 else ""
            n = d.get("n", 0)
            L.append(
                f"{train_col} & {ft_display} & {_fmt_n(n)} & "
                f"{_fmt(d.get('f1'))} & {_fmt(d.get('precision'))} & "
                f"{_fmt(d.get('recall'))} & {_fmt(d.get('iou'))} & & "
                f"{_fmt(d.get('nf_equiv_f1'))} & "
                f"{_fmt(d.get('nf_equiv_precision'))} & "
                f"{_fmt(d.get('nf_equiv_recall'))} \\\\")

        # Global row for this training config
        L.append(
            f" & All & & "
            f"{_fmt(g.get('f1'))} & {_fmt(g.get('precision'))} & "
            f"{_fmt(g.get('recall'))} & {_fmt(g.get('iou'))} & & "
            f"{_fmt(g.get('nf_equiv_f1'))} & "
            f"{_fmt(g.get('nf_equiv_precision'))} & "
            f"{_fmt(g.get('nf_equiv_recall'))} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def _find_ft_run(fire_type_results: list[dict] | None,
                 prefer: str = "all") -> dict | None:
    """Find a fire type study run, preferring the given train_fire_type."""
    if not fire_type_results:
        return None
    for r in fire_type_results:
        if r.get("train_fire_type") == prefer:
            return r
    return fire_type_results[0]


# ═══════════════════════════════════════════════════════════════════════════
# Ablation tables
# ═══════════════════════════════════════════════════════════════════════════


# Display names for ablation groups
ABLATION_DISPLAY = {
    "baseline":    "All features",
    "no_era5":     r"$-$ ERA5 weather",
    "no_gfs":      r"$-$ GFS forecast",
    "no_ae":       r"$-$ Terrain embed.",
    "no_cur_mask": r"$-$ Current mask",
    "no_accum":    r"$-$ Accum.\ state",
}

# Display order (baseline first, then groups)
ABLATION_ORDER = ["baseline", "no_era5", "no_gfs", "no_ae", "no_cur_mask", "no_accum"]


@table("ablation", needs={"ablation"})
def table_ablation(results, ds, grid, ablation=None, **kw) -> str:
    """Feature group importance — baseline vs each excluded group.

    Shows how much removing each input feature group degrades test F1
    (and nf_equiv F1 when available).  Positive Δ = group helped.
    Bold the baseline row.
    """
    if not ablation:
        return "% No ablation results"

    # Index by short tag (strip 'ablation_' prefix)
    by_tag: dict[str, dict] = {}
    for r in ablation:
        short = r.get("tag", "").replace("ablation_", "")
        by_tag[short] = r

    baseline = by_tag.get("baseline", {})
    baseline_f1 = baseline.get("test_f1", 0)
    has_nf = any(r.get("nf_equiv_f1") is not None for r in ablation)

    L: list[str] = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering\small")
    L.append(r"\caption{#1}")
    L.append(r"\label{tab:ablation}")

    if has_nf:
        L.append(r"\begin{tabular}{l r " + "c " * 4 + "c c}")
        L.append(r"\toprule")
        L.append(rf"Config & Ch & {GRID_METRIC_HEADER} & NF F1 & $\Delta$ F1 \\")
    else:
        L.append(r"\begin{tabular}{l r " + "c " * 4 + "c}")
        L.append(r"\toprule")
        L.append(rf"Config & Ch & {GRID_METRIC_HEADER} & $\Delta$ F1 \\")
    L.append(r"\midrule")

    for tag in ABLATION_ORDER:
        r = by_tag.get(tag)
        if r is None:
            continue
        display = ABLATION_DISPLAY.get(tag, tag)
        ch = r.get("n_channels", "?")
        cells = _grid_metrics_row(r)

        # Delta vs baseline
        f1 = r.get("test_f1", 0)
        if tag == "baseline":
            delta_str = "--"
            # Bold the baseline row
            display = rf"\textbf{{{display}}}"
            cells = [rf"\textbf{{{c}}}" for c in cells]
        else:
            delta = baseline_f1 - f1 if baseline_f1 else 0
            delta_str = _fmt_delta(-delta) if delta != 0 else "--"

        parts = [display, str(ch)] + cells
        if has_nf:
            parts.append(_fmt(r.get("nf_equiv_f1")))
        parts.append(delta_str)

        L.append(" & ".join(parts) + r" \\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — JSON loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_json(path: str) -> list[dict]:
    """Load a JSON file (list of dicts)."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _load_matrix_dir(tdir: Path) -> dict[str, dict]:
    """Load all transfer_matrix_*.json from *tdir*.

    Returns ``{target_type: {train_label: {eval_region: F1}}}``.
    """
    result = {}
    for p in sorted(tdir.glob("transfer_matrix_*.json")):
        tt = p.stem.replace("transfer_matrix_", "")
        with open(p) as f:
            result[tt] = json.load(f)
    return result


def _load_loo_dir(tdir: Path) -> dict[str, list[dict]]:
    """Load all loo_results_*.json from *tdir*.

    Returns ``{target_type: [entries]}``.
    """
    result = {}
    for p in sorted(tdir.glob("loo_results_*.json")):
        tt = p.stem.replace("loo_results_", "")
        with open(p) as f:
            data = json.load(f)
        result[tt] = data if isinstance(data, list) else [data]
    return result


def _load_baselines(baselines_dir: Path) -> list[dict]:
    """Scan a baselines directory for result.json files.

    Expected layout:
        baselines/
          persistence_nm/result.json
          persistence_nf/result.json
          morphological_nm/result.json
          morphological_nf/result.json
    """
    results = []
    for d in sorted(baselines_dir.iterdir()):
        if d.is_dir() and (d / "result.json").exists():
            with open(d / "result.json") as f:
                r = json.load(f)
            # Persistence on new_fires is ~0 by definition — never useful.
            if r.get("baseline") == "persistence" and r.get("target") == "new_fires":
                continue
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# main + CLI
# ═══════════════════════════════════════════════════════════════════════════


DEFAULT_TRANSFER_DIR = "data/runs/transfer"
DEFAULT_GRID         = "data/runs/grid/eval_results.json"
DEFAULT_SOLIDITY     = "data/runs/solidity_test.json"
DEFAULT_BASELINES    = "data/runs/baselines"
DEFAULT_FIRE_TYPE    = "data/runs/fire_type/fire_type_results.json"
DEFAULT_ABLATION     = "data/runs/ablation/ablation_results.json"


def main():
    """Generate requested LaTeX tables."""
    args = _parse_args()

    if args.list:
        print("Available tables:")
        for key, (fn, needs) in _TABLES.items():
            tag = ", ".join(sorted(needs)) if needs else "nothing"
            print(f"  {key:22s}  (needs: {tag})")
        return

    # Determine which tables to generate
    targets = args.tables if args.tables else list(_TABLES.keys())
    is_full_run = not args.tables  # no explicit selection = all tables

    # Lazy-loaded data sources
    loaded: dict[str, object] = {}  # tag → data
    generated: list[tuple[str, str]] = []  # (key, latex) for .tex output

    for key in targets:
        if key not in _TABLES:
            print(f"Unknown table '{key}'. Use --list to see available.")
            continue

        fn, needs = _TABLES[key]

        # Load each required data source on first demand
        skip = False
        for src in needs:
            if src in loaded:
                continue

            if src == "matrix":
                tdir = Path(args.transfer_dir)
                md = _load_matrix_dir(tdir)
                if not md:
                    print(f"  skipping {key}: no transfer_matrix_*.json in "
                          f"{tdir}", file=sys.stderr)
                    skip = True
                    break
                loaded["matrix"] = md
                tts = ", ".join(md.keys())
                print(f"  loaded matrix ({tts}) from {tdir}",
                      file=sys.stderr)

            elif src == "loo":
                tdir = Path(args.transfer_dir)
                ld = _load_loo_dir(tdir)
                if not ld:
                    print(f"  skipping {key}: no loo_results_*.json in "
                          f"{tdir}", file=sys.stderr)
                    skip = True
                    break
                loaded["loo"] = ld
                # Also load matrix for global model comparison (optional)
                if "matrix" not in loaded:
                    md = _load_matrix_dir(tdir)
                    if md:
                        loaded["matrix"] = md
                        print(f"  loaded matrix for global reference",
                              file=sys.stderr)
                tts = ", ".join(ld.keys())
                print(f"  loaded LOO results ({tts}) from {tdir}",
                      file=sys.stderr)

            elif src == "grid":
                path = Path(args.grid)
                if not path.exists():
                    print(f"  skipping {key}: {path} not found", file=sys.stderr)
                    skip = True
                    break
                loaded["grid"] = _load_json(str(path))
                print(f"  loaded {len(loaded['grid'])} grid results "
                      f"from {path}", file=sys.stderr)

            elif src == "dataset":
                from firecomp.next_day.config import NextDayConfig
                from firecomp.next_day.dataset import NextDayDataset

                cfg = NextDayConfig(
                    **({"dataset_dir": args.dataset_dir} if args.dataset_dir else {}),
                    fire_type="all",
                )
                ds = NextDayDataset(cfg)
                loaded["dataset"] = ds
                print(f"  loaded dataset: {len(ds.train_samples)} train, "
                      f"{len(ds.val_samples)} val, {len(ds.test_samples)} test",
                      file=sys.stderr)

            elif src == "solidity":
                path = Path(args.solidity)
                if not path.exists():
                    print(f"  skipping {key}: {path} not found", file=sys.stderr)
                    skip = True
                    break
                loaded["solidity"] = _load_json(str(path))
                print(f"  loaded {len(loaded['solidity'])} solidity entries "
                      f"from {path}", file=sys.stderr)

            elif src == "baselines":
                baselines_dir = Path(args.baselines)
                if not baselines_dir.is_dir():
                    print(f"  skipping {key}: {baselines_dir} not found",
                          file=sys.stderr)
                    skip = True
                    break
                bl = _load_baselines(baselines_dir)
                if not bl:
                    print(f"  skipping {key}: no result.json in {baselines_dir}",
                          file=sys.stderr)
                    skip = True
                    break
                loaded["baselines"] = bl
                print(f"  loaded {len(bl)} baseline results from "
                      f"{baselines_dir}", file=sys.stderr)

            elif src == "fire_type":
                path = Path(args.fire_type)
                if not path.exists():
                    print(f"  skipping {key}: {path} not found",
                          file=sys.stderr)
                    skip = True
                    break
                loaded["fire_type"] = _load_json(str(path))
                print(f"  loaded {len(loaded['fire_type'])} fire type study "
                      f"results from {path}", file=sys.stderr)

            elif src == "ablation":
                path = Path(args.ablation)
                if not path.exists():
                    print(f"  skipping {key}: {path} not found",
                          file=sys.stderr)
                    skip = True
                    break
                loaded["ablation"] = _load_json(str(path))
                print(f"  loaded {len(loaded['ablation'])} ablation results "
                      f"from {path}", file=sys.stderr)

        if skip:
            continue

        tex = fn(loaded.get("results"), loaded.get("dataset"),
                 loaded.get("grid"), **{k: v for k, v in loaded.items()
                                        if k not in ("results", "dataset", "grid")})
        generated.append((key, tex))

        print(f"\n% {'=' * 60}")
        print(f"% Table: {key}")
        print(f"% {'=' * 60}\n")
        print(tex)
        print()

    # After a full run, write the LaTeX include file.
    if is_full_run and generated:
        import re
        from firecomp.next_day.latex import write_tex
        out_dir = Path("data/figures")
        sections = []
        for key, body in generated:
            # Extract label from body to use as the artifact key.
            m = re.search(r"\\label\{(tab:[^}]+)\}", body)
            label = m.group(1) if m else f"tab:{key}"
            sections.append((label, body))
        tex_path = out_dir / "tables.tex"
        write_tex(tex_path, sections)
        print(f"\nWrote {tex_path} ({len(sections)} tables)", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tables",
        description="Generate Paper 1 LaTeX tables.",
    )
    p.add_argument(
        "--tables", nargs="+", default=None,
        choices=list(_TABLES.keys()),
        metavar="KEY",
        help="One or more table keys to generate (default: all).",
    )
    p.add_argument("--list", action="store_true",
                   help="List available tables and exit.")
    p.add_argument(
        "--transfer-dir", default=DEFAULT_TRANSFER_DIR,
        help=f"Transfer results directory — auto-discovers "
             f"transfer_matrix_*.json and loo_results_*.json "
             f"(default: {DEFAULT_TRANSFER_DIR}).",
    )
    p.add_argument(
        "--grid", default=DEFAULT_GRID,
        help=f"Path to grid eval_results.json (default: {DEFAULT_GRID}).",
    )
    p.add_argument(
        "--solidity", default=DEFAULT_SOLIDITY,
        help=f"Path to solidity.json (default: {DEFAULT_SOLIDITY}).",
    )
    p.add_argument(
        "--dataset-dir", default=None,
        help="Path to next-day dataset directory (default: from config).",
    )
    p.add_argument(
        "--baselines", default=DEFAULT_BASELINES,
        help=f"Path to baselines run directory (default: {DEFAULT_BASELINES}).",
    )
    p.add_argument(
        "--fire-type", default=DEFAULT_FIRE_TYPE,
        help=f"Path to fire_type_results.json (default: {DEFAULT_FIRE_TYPE}).",
    )
    p.add_argument(
        "--ablation", default=DEFAULT_ABLATION,
        help=f"Path to ablation_eval_results.json (default: {DEFAULT_ABLATION}).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
