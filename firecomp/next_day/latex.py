"""
next_day/latex.py — LaTeX output helpers for figures and tables.

Generates .tex files that can be \\input{}'d from a paper.  The output
directory (default ``data/figures/``) is meant to be downloaded and
placed as a subdirectory of the LaTeX project.  All paths inside the
generated .tex files are relative to the *parent* directory (the paper
root), so usage is::

    % In the paper .tex file:
    \\input{figures/figures}   % defines all figure artifacts
    \\input{figures/tables}    % defines all table artifacts

    % Place where needed:
    \\placegeneratedartifact{fig:sample-density}{Caption text here.}
    \\placegeneratedartifact{tab:split-summary}{Caption text here.}

The ``figures/`` prefix in paths matches the subdir name.

Required preamble (add once in the main document)::

    \\usepackage{etoolbox}

    \\newcommand{\\definegeneratedartifact}[2]{%
      \\csdef{generatedartifact@#1}##1{#2}%
    }

    \\newcommand{\\placegeneratedartifact}[2]{%
      \\ifcsdef{generatedartifact@#1}
        {\\csuse{generatedartifact@#1}{#2}}
        {\\PackageError{generatedartifacts}{Unknown generated artifact `#1'}{Check generated figures/tables tex files}}%
    }
"""

from __future__ import annotations

from pathlib import Path

# Subdirectory name as it appears inside the LaTeX project.
# All \includegraphics paths are prefixed with this.
SUBDIR = "figures"


# ── Text escaping ────────────────────────────────────────────────────────

def tex_escape_text(s: str) -> str:
    """Escape data-derived text for safe inclusion in LaTeX.

    Use for table cells, model names, column headers, and any other
    human-visible text generated from data.  Do NOT use on trusted LaTeX
    fragments (``\\textdegree{}``, ``\\textbf{}``, math, paths).
    """
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _label_from_key(key: str) -> str:
    """Convert a figure/table key to a hyphenated label.

    File paths keep underscores; labels use hyphens.
    ``predictions_grid_unet_bce_pw10_next`` → ``pred-grid-unet-bce-pw10-next``
    """
    return key.replace("_", "-")


# ── Figure wrappers ──────────────────────────────────────────────────────

def figure_env(
    pdf_stem: str,
    label: str,
    *,
    width: str = r"\textwidth",
    placement: str = "t",
) -> str:
    r"""Single \includegraphics inside a figure environment.

    Caption is provided via ``#1`` (filled at call site).
    """
    path = f"{SUBDIR}/{pdf_stem}"
    return "\n".join([
        f"\\begin{{figure}}[{placement}]",
        r"  \centering",
        f"  \\includegraphics[width={width}]{{{path}}}",
        r"  \caption{#1}",
        f"  \\label{{{label}}}",
        r"\end{figure}",
    ])


def figure_wide_env(
    pdf_stem: str,
    label: str,
    *,
    width: str = r"\textwidth",
    placement: str = "t",
) -> str:
    r"""Single \includegraphics inside a figure* (two-column) environment.

    Caption is provided via ``#1`` (filled at call site).
    """
    path = f"{SUBDIR}/{pdf_stem}"
    return "\n".join([
        f"\\begin{{figure*}}[{placement}]",
        r"  \centering",
        f"  \\includegraphics[width={width}]{{{path}}}",
        r"  \caption{#1}",
        f"  \\label{{{label}}}",
        r"\end{figure*}",
    ])


def subfigure_env(
    items: list[tuple[str, str]],
    label: str,
    *,
    sub_width: str = "0.32",
    placement: str = "t",
    wide: bool = False,
) -> str:
    r"""Multiple subfigures (requires \usepackage{subcaption}).

    *items* is a list of ``(pdf_stem, subcaption)`` pairs.
    The main caption is provided via ``#1`` (filled at call site).
    """
    env = "figure*" if wide else "figure"
    lines = [
        f"\\begin{{{env}}}[{placement}]",
        r"  \centering",
    ]
    for pdf_stem, subcap in items:
        path = f"{SUBDIR}/{pdf_stem}"
        lines += [
            f"  \\begin{{subfigure}}{{{sub_width}\\textwidth}}",
            r"    \centering",
            f"    \\includegraphics[width=\\textwidth]{{{path}}}",
            f"    \\caption{{{subcap}}}",
            f"  \\end{{subfigure}}",
            r"  \hfill",
        ]
    # Remove trailing \hfill
    if lines[-1] == r"  \hfill":
        lines.pop()
    lines += [
        r"  \caption{#1}",
        f"  \\label{{{label}}}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines)


# ── File writers ─────────────────────────────────────────────────────────

def write_tex(path: Path, sections: list[tuple[str, str]]):
    r"""Write a .tex file from (label_key, latex_body) sections.

    Each section is wrapped in ``\definegeneratedartifact{key}{body}``.
    Adds a generation comment at the top and separates sections with
    clear dividers for readability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"% Auto-generated by FireComp.  Do not edit manually.",
        f"% Re-generate: python -m firecomp.next_day.run_figures  (figures)",
        f"%              python -m firecomp.next_day.tables        (tables)",
        "",
    ]
    for key, body in sections:
        lines += [
            f"% {'=' * 60}",
            f"% {key}",
            f"% {'=' * 60}",
            f"\\definegeneratedartifact{{{key}}}{{%",
            body,
            "}",
            "",
        ]
    path.write_text("\n".join(lines))


# ── Figure manifest ──────────────────────────────────────────────────────

# Maps figure registry keys to the LaTeX they produce.
# Each entry: (key, generator_fn(out_dir) -> str)
# This is called by run_figures.py after all PDFs are generated.

def build_figures_tex(out_dir: Path) -> list[tuple[str, str]]:
    """Return (label_key, latex_body) sections for all figures that exist on disk."""
    sections = []

    def _add(label, tex):
        sections.append((label, tex))

    def _exists(stem):
        return (out_dir / f"{stem}.pdf").exists()

    # --- Dataset (Section 3) ---

    if _exists("sample_density"):
        _add("fig:sample-density", figure_wide_env(
            "sample_density", "fig:sample-density"))

    if _exists("regions"):
        _add("fig:regions", figure_wide_env(
            "regions", "fig:regions"))

    if _exists("fire_types"):
        _add("fig:fire-types", figure_wide_env(
            "fire_types", "fig:fire-types"))

    if _exists("inputs_grid"):
        _add("fig:inputs-grid", figure_wide_env(
            "inputs_grid", "fig:inputs-grid"))

    if _exists("num_fire_hist"):
        _add("fig:num-fire-hist", figure_wide_env(
            "num_fire_hist", "fig:num-fire-hist"))

    # --- Results (Section 5) ---

    # Predictions — collect all that exist
    pred_stems = sorted(
        p.stem for p in out_dir.glob("predictions_*.pdf"))
    for s in pred_stems:
        label = "fig:" + _label_from_key(s.replace("predictions_", "pred-"))
        _add(label, figure_wide_env(s, label))

    if _exists("size_and_region_f1"):
        _add("fig:size-and-region-f1", figure_env(
            "size_and_region_f1", "fig:size-and-region-f1"))

    if _exists("size_and_region_f1_nf"):
        _add("fig:size-and-region-f1-nf", figure_env(
            "size_and_region_f1_nf", "fig:size-and-region-f1-nf"))

    # --- LC analysis (Supplemental) ---

    if _exists("cluster_lc"):
        _add("fig:cluster-lc", figure_wide_env(
            "cluster_lc", "fig:cluster-lc"))

    return sections
