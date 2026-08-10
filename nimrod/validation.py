"""Validation tier: does our spin machinery reproduce known experiment?

Nothing in this project is believable unless the spin-state energetics are
trustworthy, and spin-state gaps are exactly where density functionals are least
reliable — the answer can swing by tens of kcal/mol across the functional
ladder.  So before touching the sensor we compute a set of small molecules whose
singlet–triplet gaps are known to high precision experimentally, across the
whole ladder, and report the spread.

The spread is the point.  It is the honest, physically-motivated uncertainty
that should be attached to every spin-state number later in the project, and it
tells us which functionals to trust for the nickel complex.

Reference values (all adiabatic, i.e. each spin state at its own optimised
geometry, which is what these calculations produce):

===============  ==================================  ====================
System           Transition                          Experiment
===============  ==================================  ====================
CH2 methylene    X 3B1 -> a 1A1                      9.0 kcal/mol
O2 dioxygen      X 3Sg- -> a 1Dg                     22.6 kcal/mol (0.98 eV)
NH imidogen      X 3S- -> a 1D                       36.0 kcal/mol (1.561 eV)
===============  ==================================  ====================

All three have triplet ground states, so a *positive* gap is correct and a
negative one means the functional has the ordering wrong.  The experimental
numbers are term energies (T0) and therefore include zero-point energy, which
these calculations do not; the residual ZPE difference is a few tenths of a
kcal/mol for these molecules and is small compared with the functional spread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    BASIS_TIERS,
    DATA_DIR,
    FUNCTIONAL_LADDER,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL,
)
from .psi4_driver import JobSpec, run_energy, run_optimize
from .spin import spin_properties


# --------------------------------------------------------------------------
# Benchmark definitions
# --------------------------------------------------------------------------


CM_PER_KCAL = 349.7551  # cm^-1 per kcal/mol


@dataclass(frozen=True)
class SpinGapBenchmark:
    """A molecule with an experimentally known singlet–triplet gap.

    Two reference numbers are carried, and the distinction between them is the
    whole reason this tier exists.

    ``experiment_kcal``
        The true, spectroscopically measured gap to the lowest singlet.

    ``sd_reference_kcal``
        What a *single closed-shell determinant* can legitimately be compared
        against.  For a molecule whose lowest singlet is one of a degenerate
        pair — O2 and NH both have a pi**2 configuration giving 1D and 1S+ —
        the closed-shell determinant (pi_x)^2 is not the 1D state at all: it is
        an equal mixture of 1D and 1S+.  Its energy is therefore the *average*
        of the two term energies, and comparing it against 1D alone charges DFT
        for an error it did not make.  For a genuine single-reference singlet
        such as the a 1A1 state of methylene the two numbers coincide.

    Getting this wrong inflates the apparent DFT error by 8–12 kcal/mol, which
    is larger than the real functional spread.
    """

    key: str
    name: str
    geometry: str
    charge: int
    low_multiplicity: int
    high_multiplicity: int
    experiment_kcal: float
    sd_reference_kcal: float
    transition: str
    source: str
    degenerate_pair: bool = False
    note: str = ""

    @property
    def experiment_ev(self) -> float:
        return self.experiment_kcal / HARTREE_TO_KCAL * HARTREE_TO_EV


#: Starting geometries are near-equilibrium; each spin state is then optimised
#: independently so the computed gap is adiabatic and comparable to T0.
#: Term energies are the primary spectroscopic quantities, quoted in cm^-1.
BENCHMARKS: tuple[SpinGapBenchmark, ...] = (
    SpinGapBenchmark(
        key="ch2",
        name="CH2 methylene",
        geometry="C  0.000000  0.000000  0.110000\nH  0.000000  0.982000 -0.330000\nH  0.000000 -0.982000 -0.330000",
        charge=0,
        low_multiplicity=1,
        high_multiplicity=3,
        experiment_kcal=9.0,
        sd_reference_kcal=9.0,
        transition="X 3B1 -> a 1A1",
        source="T0 = 9.0 kcal/mol (3147 cm^-1)",
        degenerate_pair=False,
        note="a 1A1 is a genuine closed-shell sigma^2 state, so a single "
             "determinant targets it directly.",
    ),
    SpinGapBenchmark(
        key="o2",
        name="O2 dioxygen",
        geometry="O  0.000000  0.000000  0.000000\nO  0.000000  0.000000  1.208000",
        charge=0,
        low_multiplicity=1,
        high_multiplicity=3,
        experiment_kcal=7918.0 / CM_PER_KCAL,          # a 1Dg, 22.64 kcal/mol
        sd_reference_kcal=(7918.0 + 13195.0) / 2 / CM_PER_KCAL,  # 30.18 kcal/mol
        transition="X 3Sg- -> a 1Dg",
        source="a1Dg 7918 cm^-1 (1270 nm); b1Sg+ 13195 cm^-1 (762 nm)",
        degenerate_pair=True,
        note="The closed-shell (pi*_x)^2 determinant is an equal mixture of "
             "a 1Dg and b 1Sg+, so it should reproduce their mean, not a 1Dg.",
    ),
    SpinGapBenchmark(
        key="nh",
        name="NH imidogen",
        geometry="N  0.000000  0.000000  0.000000\nH  0.000000  0.000000  1.036000",
        charge=0,
        low_multiplicity=1,
        high_multiplicity=3,
        experiment_kcal=12566.0 / CM_PER_KCAL,          # a 1D, 35.93 kcal/mol
        sd_reference_kcal=(12566.0 + 21202.0) / 2 / CM_PER_KCAL,  # 48.27 kcal/mol
        transition="X 3S- -> a 1D",
        source="a1D 12566 cm^-1; b1S+ 21202 cm^-1",
        degenerate_pair=True,
        note="Same pi^2 degeneracy as O2: the single determinant averages "
             "a 1D and b 1S+.",
    ),
)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class GapResult:
    """One functional's answer for one benchmark."""

    system: str
    functional: str
    basis: str
    gap_kcal: float | None = None
    gap_ev: float | None = None
    #: Deviation from the reference a single determinant can legitimately be
    #: held to.  This is the number that measures functional quality.
    error_kcal: float | None = None
    #: Deviation from the true lowest-singlet term energy.  For a degenerate
    #: pair this is dominated by the state-averaging artefact, not by the
    #: functional, and is reported only for completeness.
    error_vs_experiment_kcal: float | None = None
    e_low: float | None = None
    e_high: float | None = None
    s2_low: float | None = None
    s2_high: float | None = None
    ordering_correct: bool | None = None
    failure: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    basis: str
    results: list[GapResult] = field(default_factory=list)

    def for_system(self, system: str) -> list[GapResult]:
        return [r for r in self.results if r.system == system and r.gap_kcal is not None]

    def spread(self, system: str) -> dict[str, float] | None:
        """Min / max / mean / stdev of the computed gap across functionals."""
        values = [r.gap_kcal for r in self.for_system(system)]
        values = [v for v in values if v is not None]
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "stdev": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "range": float(arr.max() - arr.min()),
            "n": int(arr.size),
        }

    def mean_absolute_error(self) -> dict[str, float]:
        """Mean absolute error per functional across all benchmarks."""
        per_functional: dict[str, list[float]] = {}
        for result in self.results:
            if result.error_kcal is None:
                continue
            per_functional.setdefault(result.functional, []).append(abs(result.error_kcal))
        return {
            name: float(np.mean(values))
            for name, values in sorted(per_functional.items())
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "basis": self.basis,
            "benchmarks": [
                {
                    "key": b.key,
                    "name": b.name,
                    "transition": b.transition,
                    "experiment_kcal": b.experiment_kcal,
                    "sd_reference_kcal": b.sd_reference_kcal,
                    "degenerate_pair": b.degenerate_pair,
                    "source": b.source,
                    "note": b.note,
                }
                for b in BENCHMARKS
            ],
            "results": [r.to_json() for r in self.results],
            "mean_absolute_error": self.mean_absolute_error(),
            "spread": {b.name: self.spread(b.name) for b in BENCHMARKS},
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


# --------------------------------------------------------------------------
# The calculation
# --------------------------------------------------------------------------


def _optimised_state_energy(
    benchmark: SpinGapBenchmark,
    multiplicity: int,
    functional: str,
    basis: str,
    optimise: bool,
    reference: str,
) -> tuple[float | None, float | None, str | None]:
    """Return ``(energy, <S^2>, error)`` for one spin state.

    The ``reference`` matters more than it looks.  Running the closed-shell
    singlet unrestricted lets the SCF slide into a broken-symmetry Ms = 0
    solution, which is roughly a 50:50 singlet/triplet mixture and therefore far
    too low: in the first pass of this tier that collapse produced gaps of
    7.6 kcal/mol for O2 and 14 kcal/mol for NH, errors of -23 and -34, purely as
    an artefact of the reference rather than of the functional.  The intended
    closed-shell state is obtained with a restricted reference, which cannot
    spin-break by construction.
    """
    spec = JobSpec(
        geometry=benchmark.geometry,
        method=functional,
        basis=basis,
        charge=benchmark.charge,
        multiplicity=multiplicity,
        reference=reference,
        label=f"{benchmark.key}-m{multiplicity}-{functional}-{reference}",
    )

    geometry = benchmark.geometry
    if optimise:
        opt_result, optimised = run_optimize(spec, max_iter=50)
        if not opt_result.ok or optimised is None:
            return None, None, opt_result.error or "optimisation failed"
        geometry = optimised

    energy_spec = JobSpec(
        geometry=geometry,
        method=functional,
        basis=basis,
        charge=benchmark.charge,
        multiplicity=multiplicity,
        reference=reference,
        label=f"{benchmark.key}-m{multiplicity}-{functional}-{reference}-sp",
    )
    result = run_energy(energy_spec, property_hook=spin_properties)
    if not result.ok:
        return None, None, result.error
    return result.energy, result.properties.get("s2"), None


def run_spin_gap_validation(
    *,
    functionals: Sequence[str] | None = None,
    basis: str = BASIS_TIERS["production"],
    benchmarks: Iterable[SpinGapBenchmark] = BENCHMARKS,
    optimise: bool = True,
    progress=print,
) -> ValidationReport:
    """Compute every benchmark gap with every functional on the ladder."""
    names = list(functionals) if functionals else [f.name for f in FUNCTIONAL_LADDER]
    report = ValidationReport(basis=basis)

    for benchmark in benchmarks:
        for functional in names:
            if progress:
                progress(f"  {benchmark.key:5s} {functional:10s} ...", end="", flush=True)

            # High spin: unrestricted, the only option for Ms > 0.
            e_high, s2_high, err_high = _optimised_state_energy(
                benchmark, benchmark.high_multiplicity, functional, basis,
                optimise, reference="uks",
            )
            # Low spin: restricted, so the closed-shell state cannot collapse
            # into a broken-symmetry mixture.
            e_low, s2_low, err_low = _optimised_state_energy(
                benchmark, benchmark.low_multiplicity, functional, basis,
                optimise, reference="rks",
            )

            result = GapResult(
                system=benchmark.name,
                functional=functional,
                basis=basis,
                e_low=e_low,
                e_high=e_high,
                s2_low=s2_low,
                s2_high=s2_high,
            )

            if e_low is None or e_high is None:
                result.failure = err_high or err_low or "unknown failure"
                if progress:
                    progress(" FAILED")
            else:
                # Positive gap = triplet is the ground state, as experiment says.
                gap = (e_low - e_high) * HARTREE_TO_KCAL
                result.gap_kcal = gap
                result.gap_ev = gap / HARTREE_TO_KCAL * HARTREE_TO_EV
                result.error_kcal = gap - benchmark.sd_reference_kcal
                result.error_vs_experiment_kcal = gap - benchmark.experiment_kcal
                result.ordering_correct = gap > 0
                if progress:
                    progress(f" {gap:7.2f} kcal/mol  (ref {benchmark.sd_reference_kcal:.2f},"
                             f" err {result.error_kcal:+.2f})")

            report.results.append(result)

    return report


def default_report_path() -> Path:
    return DATA_DIR / "results" / "validation.json"
