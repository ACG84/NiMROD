"""Turn stored results into figures.

Reads whatever is present in ``data/results/`` and builds the corresponding
figures, skipping quietly when a result file has not been produced yet.  Keeping
this separate from :mod:`nimrod.plots` means the plotting primitives stay pure
(data in, figure out) and all the file-format knowledge lives in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR, FIGURE_DIR
from . import plots

RESULTS_DIR = DATA_DIR / "results"


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------


def build_validation_figure() -> list[Path]:
    data = _load("validation.json")
    if not data:
        return []

    benchmarks = {b["name"]: b for b in data.get("benchmarks", [])}
    grouped: dict[str, dict[str, float]] = {}
    for record in data.get("results", []):
        if record.get("gap_kcal") is None:
            continue
        grouped.setdefault(record["system"], {})[record["functional"]] = record["gap_kcal"]

    records = []
    for system, values in grouped.items():
        benchmark = benchmarks.get(system, {})
        records.append({
            "system": system,
            # Compare against the reference a single determinant can legitimately
            # reproduce; see docs/THEORY.md section 3.
            "experiment": benchmark.get("sd_reference_kcal", benchmark.get("experiment_kcal")),
            "values": values,
        })
    if not records:
        return []
    return plots.plot_validation(records)


def build_catalyst_scan_figures() -> list[Path]:
    data = _load("catalyst_scan.json")
    if not data:
        return []

    records = [r for r in data.get("records", []) if "distance" in r and "error" not in r]
    if not records:
        return []

    distances = [r["distance"] for r in records]
    functionals = data.get("functionals", [])

    series = {}
    for functional in functionals:
        values = [r.get(f"gap_{functional}") for r in records]
        if any(v is not None for v in values):
            series[functional] = values
    made: list[Path] = []
    if series:
        made += plots.plot_spin_gap_scan(distances, series)

    spin_on_ni = {}
    for functional in functionals:
        values = [r.get(f"{functional}_spin_on_Ni") for r in records]
        if any(v is not None for v in values):
            spin_on_ni[f"Ni ({functional})"] = values
    if spin_on_ni:
        made += plots.plot_spin_density(
            distances, spin_on_ni,
            title="Spin population on nickel in the S=1 state",
        )
    return made


def build_sensor_figures() -> list[Path]:
    data = _load("sensor_scan.json")
    if not data:
        return []
    points = data.get("points", [])
    if not points:
        return []

    distances = [p["distance"] for p in points]

    def series(key: str) -> list[float | None]:
        out = []
        for point in points:
            props = point.get("properties", {})
            value = props.get(key, point.get("energies", {}).get(key))
            out.append(value if isinstance(value, (int, float)) else None)
        return out

    panels = []
    if any(v is not None for v in series("spin_gap_kcal")):
        panels.append({
            "label": "metal spin gap\n(kcal/mol)",
            "values": series("spin_gap_kcal"),
            "note": "S=1 minus S=0",
        })
    if any(v is not None for v in series("J_cm")):
        panels.append({
            "label": "exchange coupling\nJ (cm$^{-1}$)",
            "values": series("J_cm"),
            "note": "J < 0 antiferromagnetic",
        })
    if any(v is not None for v in series("bright_state_ev")):
        panels.append({
            "label": "bright transition\n(eV)",
            "values": series("bright_state_ev"),
            "note": "lowest bright defect state",
        })

    made: list[Path] = []
    if panels:
        made += plots.plot_sensor_readouts(distances, panels)

    fragments = {}
    for name, key in (("colour centre", "spin_colour_centre"), ("catalyst", "spin_catalyst")):
        values = series(key)
        if any(v is not None for v in values):
            fragments[name] = values
    if fragments:
        made += plots.plot_spin_density(distances, fragments)
    return made


def build_occ_figure() -> list[Path]:
    data = _load("occ_photophysics.json")
    if not data:
        return []

    def spectrum(key: str) -> list[tuple[float, float]]:
        return [
            (s["energy_ev"], s.get("oscillator_strength") or 0.0)
            for s in data.get(key, [])
            if s.get("energy_ev") is not None
        ]

    spectra = {}
    pristine = spectrum("pristine")
    defect = spectrum("defect")
    if pristine:
        spectra["pristine pyrene"] = pristine
    if defect:
        spectra["sp$^3$ aryl defect"] = defect
    if not spectra:
        return []
    return plots.plot_absorption_comparison(spectra)


def build_sapt_figure() -> list[Path]:
    data = _load("sapt_scan.json")
    if not data:
        return []
    points = data.get("points", [])
    if not points:
        return []
    distances = [p["distance"] for p in points]
    components = {}
    for key, label in (
        ("electrostatics", "electrostatics"),
        ("exchange", "exchange"),
        ("induction", "induction"),
        ("dispersion", "dispersion"),
    ):
        values = [p.get(key) for p in points]
        if any(isinstance(v, (int, float)) for v in values):
            components[label] = values
    if not components:
        return []
    return plots.plot_sapt_components(distances, components)


def build_all() -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for builder in (
        build_validation_figure,
        build_catalyst_scan_figures,
        build_occ_figure,
        build_sensor_figures,
        build_sapt_figure,
    ):
        try:
            made.extend(builder())
        except Exception as exc:  # one bad result file must not block the rest
            print(f"  ! {builder.__name__}: {type(exc).__name__}: {exc}")
    return made
