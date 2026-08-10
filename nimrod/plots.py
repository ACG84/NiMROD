"""Figures for the NiMROD results.

Every figure is emitted in both a light and a dark variant from the same code
path, using a categorical palette that has been machine-validated for
colour-vision deficiency separation rather than chosen by eye.

Two rules are enforced structurally because they are the easiest way to make a
scientific figure lie:

* **No dual axes.**  The three sensor readouts carry different units (eV,
  cm^-1, electrons).  Overlaying them on twinned y-axes would let the reader
  infer a correlation from an arbitrary choice of scale, so multi-observable
  figures are drawn as stacked small multiples sharing one x-axis instead.

* **Series are direct-labelled.**  Two of the light-mode palette slots sit below
  3:1 contrast against the page, so identity is never carried by colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .config import FIGURE_DIR

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e2e1dd",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#3a3a37",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
)

THEMES = (LIGHT, DARK)

#: Reserved status colours, never reused as a series hue.
STATUS = {"reference": "#52514e", "warning": "#eda100", "critical": "#e34948"}


def _style_axes(ax, theme: Theme, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor(theme.surface)
    ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.text_secondary, labelsize=9, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme.text_secondary, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme.text_secondary, fontsize=10)
    if title:
        ax.set_title(title, color=theme.text_primary, fontsize=11, loc="left", pad=10)


def _direct_label(ax, x: float, y: float, text: str, colour: str, theme: Theme) -> None:
    """Label a series at its right-hand end.

    This is the relief required by the low-contrast palette slots: identity is
    carried by the text, with the colour only reinforcing it.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        color=theme.text_primary,
        fontsize=9,
        va="center",
        ha="left",
        fontweight="medium",
    )


def _finish(fig: Figure, theme: Theme, stem: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or FIGURE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(theme.surface)
    path = outdir / f"{stem}-{theme.name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=theme.surface)
    plt.close(fig)
    return path


def _clean(xs: Sequence[float], ys: Sequence[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Drop points where the observable failed to compute."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y)]
    if not pairs:
        return np.array([]), np.array([])
    x_arr, y_arr = zip(*pairs)
    return np.asarray(x_arr, dtype=float), np.asarray(y_arr, dtype=float)


# --------------------------------------------------------------------------
# Figure 1: validation against experiment
# --------------------------------------------------------------------------


def plot_validation(
    records: Sequence[Mapping[str, object]],
    *,
    stem: str = "validation",
    outdir: Path | None = None,
) -> list[Path]:
    """Computed spin-state gaps against experiment.

    ``records`` entries need ``system``, ``experiment`` (the reference value),
    and ``values`` (a mapping of method name to computed value), all in the same
    unit.  Each system is drawn as a horizontal strip: experiment as a reference
    rule, computed points scattered against it, so the reader sees the spread
    across functionals rather than one cherry-picked number.
    """
    method_order: list[str] = []
    for record in records:
        for method in record.get("values", {}):  # type: ignore[union-attr]
            if method not in method_order:
                method_order.append(method)
    if not method_order:
        return []

    paths = []
    for theme in THEMES:
        # One panel per molecule; functionals are y-axis rows.  Identity is
        # carried by the row label, not by hue, so this stays readable with
        # eight methods and in greyscale.  Colour is left to encode one thing
        # only: the sign of the deviation.
        over, under = theme.series[0], theme.series[1]
        fig, axes = plt.subplots(
            1, len(records), figsize=(3.3 * len(records) + 1.2, 0.42 * len(method_order) + 2.4),
            sharey=True,
        )
        if len(records) == 1:
            axes = [axes]
        fig.patch.set_facecolor(theme.surface)

        for ax, record in zip(axes, records):
            reference = record.get("experiment")
            values = record.get("values") or {}
            for row, method in enumerate(method_order):
                y = len(method_order) - row - 1
                value = values.get(method)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    ax.annotate("no convergence", xy=(0.5, y), xycoords=("axes fraction", "data"),
                                ha="center", va="center", fontsize=7.5,
                                color=theme.text_secondary, style="italic")
                    continue
                colour = over if (isinstance(reference, (int, float))
                                  and value >= reference) else under
                if isinstance(reference, (int, float)):
                    # Lollipop from the reference to the computed value: the
                    # stem length *is* the error.
                    ax.plot([reference, value], [y, y], color=colour,
                            linewidth=2.0, zorder=3, solid_capstyle="round")
                ax.plot(value, y, marker="o", markersize=7, color=colour,
                        markeredgecolor=theme.surface, markeredgewidth=1.8,
                        zorder=4, linestyle="none")

            if isinstance(reference, (int, float)):
                ax.axvline(reference, color=STATUS["reference"], linewidth=1.6,
                           zorder=2)
                ax.annotate(f"ref {reference:.1f}", xy=(reference, len(method_order) - 0.35),
                            xytext=(4, 0), textcoords="offset points",
                            color=theme.text_secondary, fontsize=8, va="center")

            _style_axes(ax, theme, "gap (kcal/mol)", "", str(record["system"]))
            ax.set_ylim(-0.8, len(method_order) - 0.2)
            ax.margins(x=0.18)

        axes[0].set_yticks(range(len(method_order)))
        axes[0].set_yticklabels(list(reversed(method_order)),
                                color=theme.text_primary, fontsize=9)

        fig.suptitle(
            "Validation: computed spin-state gaps against the single-determinant reference",
            color=theme.text_primary, fontsize=11, x=0.02, ha="left", y=1.0,
        )
        handles = [
            plt.Line2D([], [], color=over, linewidth=2.0, marker="o", markersize=7,
                       label="overestimates"),
            plt.Line2D([], [], color=under, linewidth=2.0, marker="o", markersize=7,
                       label="underestimates"),
            plt.Line2D([], [], color=STATUS["reference"], linewidth=1.6, label="reference"),
        ]
        legend = fig.legend(handles=handles, loc="lower center", ncol=3,
                            frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)

        fig.tight_layout()
        paths.append(_finish(fig, theme, stem, outdir))
    return paths


# --------------------------------------------------------------------------
# Figure 2: spin-state gap along the degradation coordinate
# --------------------------------------------------------------------------


def plot_spin_gap_scan(
    distances: Sequence[float],
    series: Mapping[str, Sequence[float | None]],
    *,
    ylabel: str = "S=1 minus S=0 gap (kcal/mol)",
    title: str = "Metal spin-state gap collapses as the imine arm dissociates",
    stem: str = "spin-gap-scan",
    outdir: Path | None = None,
) -> list[Path]:
    """One line per density functional, with the spread as the honest error bar."""
    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        fig.patch.set_facecolor(theme.surface)

        endpoints: list[tuple[float, float, str, str]] = []
        for i, (name, values) in enumerate(series.items()):
            x, y = _clean(distances, values)
            if x.size == 0:
                continue
            colour = theme.series[i % len(theme.series)]
            ax.plot(x, y, color=colour, linewidth=2.0, marker="o", markersize=5,
                    markeredgecolor=theme.surface, markeredgewidth=1.5, zorder=3,
                    solid_capstyle="round")
            endpoints.append((float(x[-1]), float(y[-1]), name, colour))

        # Series that finish close together would print their labels on top of
        # each other; nudge them apart so identity stays legible.
        endpoints.sort(key=lambda e: e[1])
        if endpoints:
            # Base the spacing on the drawn axis range, not the spread of the
            # endpoints: a label needs a fixed slice of the *figure*, and two
            # series finishing 1 kcal/mol apart still collide when the axis
            # spans thirty.
            low, high = ax.get_ylim()
            minimum_gap = (high - low) * 0.052
            placed: list[float] = []
            for x_end, y_end, name, colour in endpoints:
                y_label = y_end
                if placed and y_label - placed[-1] < minimum_gap:
                    y_label = placed[-1] + minimum_gap
                placed.append(y_label)
                if abs(y_label - y_end) > 1e-9:
                    ax.plot([x_end, x_end], [y_end, y_label], color=colour,
                            linewidth=0.8, alpha=0.5, zorder=2)
                _direct_label(ax, x_end, y_label, name, colour, theme)

        # The spin-flip crossing: where the gap changes sign the ground state
        # of the metal has changed, which is the physical detection threshold.
        ax.axhline(0.0, color=STATUS["reference"], linewidth=1.2, linestyle=(0, (4, 3)),
                   zorder=2)
        ax.annotate("S=0 / S=1 crossing", xy=(float(np.min(distances)), 0.0),
                    xytext=(2, 5), textcoords="offset points",
                    color=theme.text_secondary, fontsize=8.5)

        _style_axes(ax, theme, "Ni–N distance (Å)", ylabel, title)
        ax.margins(x=0.14)
        paths.append(_finish(fig, theme, stem, outdir))
    return paths


# --------------------------------------------------------------------------
# Figure 3: the three sensor readouts as small multiples
# --------------------------------------------------------------------------


def plot_sensor_readouts(
    distances: Sequence[float],
    panels: Sequence[Mapping[str, object]],
    *,
    title: str = "Three independent readouts of catalyst degradation",
    stem: str = "sensor-readouts",
    outdir: Path | None = None,
) -> list[Path]:
    """Stacked small multiples sharing the degradation coordinate.

    ``panels`` entries need ``label`` (the y-axis text) and ``values``; an
    optional ``note`` is annotated on the panel.  Small multiples rather than
    twinned axes: these observables have different units and putting them on one
    pair of axes would manufacture an apparent correlation.
    """
    paths = []
    for theme in THEMES:
        fig, axes = plt.subplots(
            len(panels), 1, figsize=(7.4, 2.35 * len(panels) + 0.8), sharex=True
        )
        if len(panels) == 1:
            axes = [axes]
        fig.patch.set_facecolor(theme.surface)

        for i, (ax, panel) in enumerate(zip(axes, panels)):
            colour = theme.series[i % len(theme.series)]
            x, y = _clean(distances, panel.get("values", []))  # type: ignore[arg-type]
            if x.size:
                ax.plot(x, y, color=colour, linewidth=2.0, marker="o", markersize=5,
                        markeredgecolor=theme.surface, markeredgewidth=1.5, zorder=3,
                        solid_capstyle="round")
            _style_axes(ax, theme, "", str(panel.get("label", "")),
                        title if i == 0 else "")
            if panel.get("note"):
                ax.annotate(str(panel["note"]), xy=(0.99, 0.06),
                            xycoords="axes fraction", ha="right",
                            color=theme.text_secondary, fontsize=8.5)
            ax.margins(x=0.05)

        axes[-1].set_xlabel("Ni–N distance (Å)  →  increasing degradation",
                            color=theme.text_secondary, fontsize=10)
        fig.align_ylabels(axes)
        fig.subplots_adjust(hspace=0.18)
        paths.append(_finish(fig, theme, stem, outdir))
    return paths


# --------------------------------------------------------------------------
# Figure 4: spin redistribution between fragments
# --------------------------------------------------------------------------


def plot_spin_density(
    distances: Sequence[float],
    fragments: Mapping[str, Sequence[float | None]],
    *,
    title: str = "Unpaired spin redistributes from the colour centre onto the metal",
    stem: str = "spin-density",
    outdir: Path | None = None,
) -> list[Path]:
    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.patch.set_facecolor(theme.surface)

        for i, (name, values) in enumerate(fragments.items()):
            x, y = _clean(distances, values)
            if x.size == 0:
                continue
            colour = theme.series[i % len(theme.series)]
            ax.plot(x, y, color=colour, linewidth=2.0, marker="o", markersize=5,
                    markeredgecolor=theme.surface, markeredgewidth=1.5, zorder=3,
                    solid_capstyle="round")
            _direct_label(ax, x[-1], y[-1], name, colour, theme)

        _style_axes(ax, theme, "Ni–N distance (Å)",
                    "Mulliken spin population (electrons)", title)
        ax.margins(x=0.16)
        paths.append(_finish(fig, theme, stem, outdir))
    return paths


# --------------------------------------------------------------------------
# Figure 5: SAPT decomposition of solvent capture at the open site
# --------------------------------------------------------------------------


def plot_sapt_components(
    distances: Sequence[float],
    components: Mapping[str, Sequence[float | None]],
    *,
    title: str = "Character of water binding at the vacated coordination site",
    stem: str = "sapt-components",
    outdir: Path | None = None,
) -> list[Path]:
    """Grouped bars per scan point, one bar per SAPT component.

    A 2px surface gap separates adjacent bars so neighbouring fills never touch.
    """
    paths = []
    names = list(components)
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        fig.patch.set_facecolor(theme.surface)

        n = len(names)
        width = 0.8 / max(n, 1)
        centres = np.arange(len(distances), dtype=float)

        for i, name in enumerate(names):
            values = [
                v if isinstance(v, (int, float)) and np.isfinite(v) else np.nan
                for v in components[name]
            ]
            offset = (i - (n - 1) / 2) * width
            ax.bar(
                centres + offset, values, width=width * 0.9,
                color=theme.series[i % len(theme.series)], label=name,
                zorder=3, linewidth=1.5, edgecolor=theme.surface,
            )

        ax.axhline(0.0, color=theme.grid, linewidth=1.0, zorder=2)
        ax.set_xticks(centres)
        ax.set_xticklabels([f"{d:.2f}" for d in distances])
        _style_axes(ax, theme, "Ni–N distance (Å)",
                    "interaction energy (kcal/mol)", title)
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                           ncol=min(n, 5), frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)
        paths.append(_finish(fig, theme, stem, outdir))
    return paths


# --------------------------------------------------------------------------
# Figure 6: the colour centre's optical signature
# --------------------------------------------------------------------------


def plot_absorption_comparison(
    spectra: Mapping[str, Sequence[tuple[float, float]]],
    *,
    broadening_ev: float = 0.15,
    title: str = "Colour-centre absorption: pristine pyrene vs sp3 aryl defect",
    stem: str = "absorption",
    outdir: Path | None = None,
) -> list[Path]:
    """Gaussian-broadened stick spectra.

    ``spectra`` maps a label to a list of ``(energy_eV, oscillator_strength)``.
    """
    paths = []
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.patch.set_facecolor(theme.surface)

        all_energies = [e for states in spectra.values() for e, _ in states]
        if not all_energies:
            plt.close(fig)
            continue
        grid = np.linspace(max(0.0, min(all_energies) - 1.0), max(all_energies) + 1.0, 800)

        for i, (name, states) in enumerate(spectra.items()):
            colour = theme.series[i % len(theme.series)]
            profile = np.zeros_like(grid)
            for energy, strength in states:
                profile += strength * np.exp(
                    -((grid - energy) ** 2) / (2.0 * broadening_ev**2)
                )
            ax.plot(grid, profile, color=colour, linewidth=2.0, zorder=3)
            for energy, strength in states:
                ax.plot([energy, energy], [0, strength], color=colour,
                        linewidth=1.0, alpha=0.45, zorder=2)
            peak = int(np.argmax(profile))
            _direct_label(ax, grid[peak], profile[peak], name, colour, theme)

        _style_axes(ax, theme, "excitation energy (eV)",
                    "oscillator strength (broadened)", title)
        ax.margins(x=0.12)
        ax.set_ylim(bottom=0)
        paths.append(_finish(fig, theme, stem, outdir))
    return paths
