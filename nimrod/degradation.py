"""The degradation coordinate: ligand loss at the metal centre.

Catalyst decay is modelled as progressive dissociation of one imine arm of the
truncated salen complex — the Ni-N bond on the chelate arm *opposite* the
reporting colour centre is stretched from its equilibrium 1.87 A out to 4.5 A,
where the arm is fully decoordinated and the metal has an open site.

Two things make this a physically meaningful scan rather than a cartoon:

**It is relaxed, not rigid.**  At each point only the Ni-N distance is frozen;
every other degree of freedom is optimised.  A rigid pull would report the
energy of a strained artefact rather than of the degraded catalyst.

**It uses sequential continuation.**  Each scan point starts from the *previous*
converged geometry rather than from the intact complex.  Displacing the nitrogen
rigidly from the equilibrium structure drives it into other atoms — at a 3.8 A
target the closest contact falls to 0.84 A, which is not a geometry any SCF
should be asked to start from.  Walking the coordinate outward one step at a
time keeps every starting guess physical.

The observables harvested along the way are the three sensor readouts: the
spin-state energetics of the metal, the exchange coupling between the metal and
the colour-centre spin, and the optical transition of the colour centre itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import (
    BASIS_TIERS,
    DATA_DIR,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL,
    REFERENCE_FUNCTIONAL,
)
from .geometry import Structure, arm_atoms, elongate_bond
from .psi4_driver import JobResult, JobSpec, run_energy, run_optimize

#: Default Ni-N separations (angstrom) spanning intact to fully dissociated.
DEFAULT_COORDINATE = (1.87, 2.10, 2.35, 2.60, 2.90, 3.30, 3.80, 4.50)


@dataclass
class ScanPoint:
    """One point on the degradation coordinate."""

    distance: float
    geometry: str | None = None
    converged: bool = False
    energies: dict[str, float] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScanResult:
    """A complete degradation scan."""

    label: str
    functional: str
    basis: str
    points: list[ScanPoint] = field(default_factory=list)

    @property
    def distances(self) -> list[float]:
        return [p.distance for p in self.points]

    def series(self, key: str) -> list[float | None]:
        """Extract one named observable across the scan, with gaps as None."""
        out: list[float | None] = []
        for point in self.points:
            value = point.energies.get(key, point.properties.get(key))
            out.append(float(value) if isinstance(value, (int, float)) else None)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "functional": self.functional,
            "basis": self.basis,
            "points": [p.to_json() for p in self.points],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScanResult":
        data = json.loads(Path(path).read_text())
        result = cls(data["label"], data["functional"], data["basis"])
        result.points = [ScanPoint(**p) for p in data["points"]]
        return result


# --------------------------------------------------------------------------
# Constrained relaxation
# --------------------------------------------------------------------------


def frozen_distance_option(anchor: int, moving: int) -> str:
    """Psi4 optking ``frozen_distance`` string for a 1-indexed atom pair."""
    return f"{anchor + 1} {moving + 1}"


def relax_at_distance(
    structure: Structure,
    anchor: int,
    moving: int,
    distance: float,
    *,
    functional: str = REFERENCE_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
    charge: int = 0,
    multiplicity: int = 1,
    carry: Sequence[int] = (),
    label: str = "",
    max_iter: int = 40,
) -> tuple[JobResult, Structure | None]:
    """Optimise ``structure`` with the anchor-moving distance frozen.

    The incoming structure is first displaced so that the constrained distance
    is already satisfied, because optking constrains a coordinate at its
    *starting* value.
    """
    displaced = elongate_bond(structure, anchor, moving, distance, carry=carry)

    spec = JobSpec(
        geometry=displaced.to_psi4(),
        method=functional,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        options={"frozen_distance": frozen_distance_option(anchor, moving)},
        label=label or f"relax@{distance:.2f}",
    )
    result, geometry = run_optimize(spec, max_iter=max_iter)
    relaxed = Structure.from_psi4(geometry) if geometry else None

    # Guard against the constraint silently slipping.
    if relaxed is not None:
        actual = relaxed.distance(anchor, moving)
        if abs(actual - distance) > 0.05:
            result.properties["constraint_slip"] = actual - distance
    return result, relaxed


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------


def run_scan(
    structure: Structure,
    ni_index: int,
    n_index: int,
    *,
    distances: Sequence[float] = DEFAULT_COORDINATE,
    functional: str = REFERENCE_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
    charge: int = 0,
    multiplicity: int = 1,
    observers: Iterable[Callable[[Structure, float], dict[str, Any]]] = (),
    label: str = "degradation",
    relax: bool = True,
    progress: Callable[[str], None] | None = print,
) -> ScanResult:
    """Walk the degradation coordinate with sequential continuation.

    ``observers`` are callables ``f(relaxed_structure, distance) -> dict`` run at
    each converged point; their outputs are merged into the point's properties.
    This is how the spin, exchange-coupling and optical readouts are attached
    without hard-wiring them into the scan.

    If a point fails to converge the scan does not abort: the failure is
    recorded and continuation falls back to the last good geometry, so one bad
    point cannot silently truncate the coordinate.
    """
    result = ScanResult(label=label, functional=functional, basis=basis)
    carry = arm_atoms(structure, ni_index, n_index)
    current = structure
    last_good = structure

    for distance in distances:
        if progress:
            progress(f"[{label}] Ni-N = {distance:.2f} A")

        point = ScanPoint(distance=distance)

        if relax:
            job, relaxed = relax_at_distance(
                current,
                ni_index,
                n_index,
                distance,
                functional=functional,
                basis=basis,
                charge=charge,
                multiplicity=multiplicity,
                carry=carry,
                label=f"{label}@{distance:.2f}",
            )
            if job.ok and relaxed is not None:
                point.converged = True
                point.geometry = relaxed.to_psi4()
                point.energies[f"E_{functional}"] = job.energy
                if "constraint_slip" in job.properties:
                    point.properties["constraint_slip"] = job.properties["constraint_slip"]
                current = relaxed
                last_good = relaxed
            else:
                point.error = job.error or "optimisation did not converge"
                # Continue from the last good geometry rather than a bad one.
                current = last_good
                result.points.append(point)
                if progress:
                    progress(f"    FAILED: {point.error}")
                continue
        else:
            relaxed = elongate_bond(structure, ni_index, n_index, distance, carry=carry)
            point.converged = True
            point.geometry = relaxed.to_psi4()
            current = relaxed

        for observer in observers:
            try:
                point.properties.update(observer(current, distance) or {})
            except Exception as exc:  # an observer must never kill the scan
                point.properties.setdefault("observer_errors", []).append(
                    f"{type(exc).__name__}: {exc}"
                )

        result.points.append(point)

    return result


# --------------------------------------------------------------------------
# Observers: the three sensor readouts
# --------------------------------------------------------------------------


def spin_state_observer(
    *,
    charge: int = 0,
    low_multiplicity: int = 1,
    high_multiplicity: int = 3,
    functional: str = REFERENCE_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
) -> Callable[[Structure, float], dict[str, Any]]:
    """Readout 1: the metal's own spin-state splitting.

    Returns the low-spin/high-spin gap in eV and kcal/mol.  A positive gap means
    the low-spin state is the ground state; the gap collapsing and changing sign
    along the coordinate *is* the degradation signal at the metal.
    """

    def observe(structure: Structure, distance: float) -> dict[str, Any]:
        from .spin import spin_properties

        geometry = structure.to_psi4()
        out: dict[str, Any] = {}
        energies: dict[int, float] = {}

        for multiplicity in (low_multiplicity, high_multiplicity):
            spec = JobSpec(
                geometry=geometry,
                method=functional,
                basis=basis,
                charge=charge,
                multiplicity=multiplicity,
                reference="uks",  # UKS for both, so the comparison is like-for-like
                label=f"spin-m{multiplicity}@{distance:.2f}",
            )
            job = run_energy(spec, property_hook=spin_properties)
            if not job.ok:
                out[f"spin_m{multiplicity}_error"] = job.error
                continue
            energies[multiplicity] = job.energy
            out[f"E_m{multiplicity}"] = job.energy
            for key in ("s2", "spin_contamination"):
                if key in job.properties:
                    out[f"{key}_m{multiplicity}"] = job.properties[key]

        if low_multiplicity in energies and high_multiplicity in energies:
            gap = energies[high_multiplicity] - energies[low_multiplicity]
            out["spin_gap_ev"] = gap * HARTREE_TO_EV
            out["spin_gap_kcal"] = gap * HARTREE_TO_KCAL
        return out

    return observe


def exchange_observer(
    *,
    charge: int = 0,
    functional: str = REFERENCE_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
) -> Callable[[Structure, float], dict[str, Any]]:
    """Readout 2: exchange coupling between the colour-centre spin and the metal.

    Uses broken-symmetry DFT with the Yamaguchi projection.  This is the
    magnetic channel of the sensor: as the metal turns paramagnetic on ligand
    loss, a coupling that was essentially zero becomes measurable.
    """

    def observe(structure: Structure, distance: float) -> dict[str, Any]:
        from .spin import exchange_coupling

        coupling = exchange_coupling(
            structure.to_psi4(),
            method=functional,
            basis=basis,
            charge=charge,
            label=f"J@{distance:.2f}",
        )
        return {
            "J_cm": coupling.j_cm,
            "J_kcal": coupling.j_kcal,
            "E_hs": coupling.e_hs,
            "E_bs": coupling.e_bs,
            "s2_hs": coupling.s2_hs,
            "s2_bs": coupling.s2_bs,
        }

    return observe


def optical_observer(
    *,
    charge: int = 0,
    multiplicity: int = 2,
    functional: str = REFERENCE_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
    n_states: int = 6,
) -> Callable[[Structure, float], dict[str, Any]]:
    """Readout 3: the colour centre's optical transition.

    Returns the lowest bright transition of the defect.  Shifts in this number
    are the optical channel of the sensor and the one that maps onto a real
    photoluminescence measurement.
    """

    def observe(structure: Structure, distance: float) -> dict[str, Any]:
        from .excited import occ_absorption

        absorption = occ_absorption(
            structure.to_psi4(),
            functional=functional,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            n_states=n_states,
        )
        state = absorption.lowest_bright
        if state is None:
            return {"optical_error": "no bright state found"}
        return {
            "bright_state_ev": state.energy_ev,
            "bright_state_nm": state.energy_nm,
            "oscillator_strength": state.oscillator_strength,
            "reference_spin_contamination": absorption.spin_contamination,
        }

    return observe


def spin_density_observer(
    fragments: dict[str, Sequence[int]],
    *,
    charge: int = 0,
    multiplicity: int = 2,
    functional: str = REFERENCE_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
) -> Callable[[Structure, float], dict[str, Any]]:
    """Readout 3b: how the unpaired spin redistributes between fragments.

    ``fragments`` maps a name (e.g. ``"colour_centre"``, ``"catalyst"``) to the
    atom indices it contains.  Spin leaking off the defect and onto the metal is
    the microscopic mechanism behind both other readouts.
    """

    def observe(structure: Structure, distance: float) -> dict[str, Any]:
        from .spin import fragment_spin, spin_properties

        spec = JobSpec(
            geometry=structure.to_psi4(),
            method=functional,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            reference="uks",
            label=f"rho@{distance:.2f}",
        )
        job = run_energy(spec, property_hook=spin_properties)
        if not job.ok:
            return {"spin_density_error": job.error}

        populations = np.asarray(job.properties.get("mulliken_spin", []), dtype=float)
        if populations.size == 0:
            return {"spin_density_error": "no spin populations returned"}

        out: dict[str, Any] = {"s2_doublet": job.properties.get("s2")}
        for name, indices in fragments.items():
            out[f"spin_{name}"] = fragment_spin(populations, indices)
        return out

    return observe


# --------------------------------------------------------------------------
# Convenience entry point
# --------------------------------------------------------------------------


def default_scan_path(label: str) -> Path:
    return DATA_DIR / "results" / f"{label}.json"
