"""Global compute configuration for NiMROD.

Centralises everything that has to be consistent across every calculation in
the project: scratch handling, memory/thread budgets, and the basis-set and
density-functional registries used by the multi-method protocols.

The scratch settings here are not cosmetic.  Psi4's ``detci`` (CASSCF) module
writes large scratch files through libpsio and will abort with

    PSIO_ERROR: 18 (Incorrect block end address)

if ``PSI_SCRATCH`` is left pointing at a shared or full ``/tmp``.  We therefore
always point it at a project-local directory that we own.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"
SCRATCH_DIR = PROJECT_ROOT / ".psi4_scratch"
OUTPUT_DIR = PROJECT_ROOT / ".psi4_output"

for _d in (DATA_DIR, FIGURE_DIR, SCRATCH_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Physical constants (CODATA 2018, as used by Psi4)
# --------------------------------------------------------------------------

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_CM = 219474.6313632
HARTREE_TO_NM = 45.56335252767   # nm = HARTREE_TO_NM / E_hartree
BOHR_TO_ANGSTROM = 0.529177210903


# --------------------------------------------------------------------------
# Compute budget
# --------------------------------------------------------------------------


@dataclass
class ComputeConfig:
    """Resource budget handed to Psi4.

    Defaults are tuned for the 4-core / 15 GB container this project was
    developed in; override via ``NIMROD_THREADS`` / ``NIMROD_MEMORY_GB``.
    """

    threads: int = field(default_factory=lambda: int(os.environ.get("NIMROD_THREADS", os.cpu_count() or 4)))
    memory_gb: int = field(default_factory=lambda: int(os.environ.get("NIMROD_MEMORY_GB", 10)))
    scratch: Path = SCRATCH_DIR
    quiet: bool = True

    def apply(self) -> None:
        """Push this configuration into the Psi4 core.

        Safe to call repeatedly; ``psi4`` is imported lazily so that importing
        :mod:`nimrod.config` does not require a Psi4 installation.
        """
        import psi4

        # PSI_SCRATCH must be set before the IO subsystem is first touched.
        self.scratch.mkdir(parents=True, exist_ok=True)
        os.environ["PSI_SCRATCH"] = str(self.scratch)
        psi4.core.IOManager.shared_object().set_default_path(str(self.scratch))

        psi4.set_memory(f"{self.memory_gb} GB")
        psi4.set_num_threads(self.threads)
        psi4.core.set_output_file(str(OUTPUT_DIR / "psi4.log"), True)
        if self.quiet:
            psi4.core.be_quiet()

    def clean_scratch(self) -> None:
        """Remove leftover scratch files between large jobs."""
        if self.scratch.exists():
            for item in self.scratch.iterdir():
                try:
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
                except OSError:
                    pass


DEFAULT_COMPUTE = ComputeConfig()


# --------------------------------------------------------------------------
# Basis sets
# --------------------------------------------------------------------------

#: Basis sets by cost tier.  All of these were verified to carry Ni parameters.
BASIS_TIERS = {
    "screen": "def2-svp",
    "production": "def2-tzvp",
    "benchmark": "def2-tzvpp",
    "correlated": "cc-pvdz",
    "diffuse": "def2-svpd",
}


# --------------------------------------------------------------------------
# Density functionals
# --------------------------------------------------------------------------
#
# Spin-state energetics of 3d transition metals are notoriously sensitive to
# the exact-exchange admixture.  We therefore never report a single functional:
# every spin-state quantity is computed across this ladder and the spread is
# reported as an honest, physically-motivated uncertainty.  The functionals are
# ordered by nominal Hartree-Fock exchange fraction, which is the dominant
# variable controlling high-spin/low-spin splittings.

@dataclass(frozen=True)
class Functional:
    name: str          #: Psi4 method string
    hf_exchange: float  #: nominal global HF exchange fraction
    rung: str          #: Jacob's-ladder rung
    dispersion: bool = False


FUNCTIONAL_LADDER = (
    Functional("bp86", 0.00, "GGA"),
    Functional("pbe", 0.00, "GGA"),
    Functional("tpss", 0.00, "meta-GGA"),
    Functional("tpssh", 0.10, "hybrid meta-GGA"),
    Functional("b3lyp", 0.20, "hybrid GGA"),
    Functional("pbe0", 0.25, "hybrid GGA"),
    Functional("m06", 0.27, "hybrid meta-GGA"),
    Functional("wb97x-d", 0.222, "range-separated hybrid", dispersion=True),
)

#: Cheap subset for dense scans where the full ladder is unaffordable.
FUNCTIONAL_SCREEN = ("bp86", "b3lyp", "pbe0", "wb97x-d")

#: The functional used whenever a single representative number is needed.
#: TPSSh is the usual recommendation for 3d spin-state energetics because its
#: 10% exact exchange sits near the empirical sweet spot.
REFERENCE_FUNCTIONAL = "tpssh"


def functional_by_name(name: str) -> Functional:
    for f in FUNCTIONAL_LADDER:
        if f.name == name:
            return f
    raise KeyError(f"unknown functional {name!r}")


# --------------------------------------------------------------------------
# SCF convergence presets
# --------------------------------------------------------------------------
#
# Open-shell transition-metal SCF is the least reliable step in this whole
# project.  These presets are tried in order by the driver until one converges.

#: Meta-GGA functionals.  Psi4's second-order SCF cannot build rotated exchange
#: -correlation potentials for these -- it aborts with
#:
#:     Vx: RKS does not support rotated V builds for MGGA's
#:
#: and the failure additionally leaves a Psi4 timer in a bad state, so the
#: *next* attempt dies with "Timer RV: Form Vx is already on" even under a
#: preset that would otherwise work.  The SOSCF rung is therefore skipped for
#: these functionals rather than tried and recovered from.
METAGGA_FUNCTIONALS = frozenset({
    "tpss", "tpssh", "revtpss", "m06", "m06-l", "m06-2x", "m06-hf",
    "m11", "mn15", "scan", "r2scan", "b97m-v", "wb97m-v", "mgga_ms0",
})


def is_metagga(method: str) -> bool:
    return method.lower().removeprefix("td-") in METAGGA_FUNCTIONALS


SCF_PRESETS = (
    {
        "label": "default",
        "options": {
            "scf_type": "df",
            "guess": "sad",
            "e_convergence": 1e-8,
            "d_convergence": 1e-7,
            "maxiter": 200,
        },
    },
    {
        "label": "soscf",
        "options": {
            "scf_type": "df",
            "guess": "sad",
            "soscf": True,
            "e_convergence": 1e-8,
            "d_convergence": 1e-7,
            "maxiter": 300,
        },
    },
    {
        "label": "damped-core",
        "options": {
            "scf_type": "df",
            "guess": "core",
            "damping_percentage": 40.0,
            "e_convergence": 1e-7,
            "d_convergence": 1e-6,
            "maxiter": 400,
        },
    },
    {
        "label": "fractional",
        "options": {
            "scf_type": "df",
            "guess": "gwh",
            "damping_percentage": 60.0,
            "diis_start": 5,
            "e_convergence": 1e-7,
            "d_convergence": 1e-6,
            "maxiter": 500,
        },
    },
)


def applicable_presets(method: str) -> tuple[dict, ...]:
    """The SCF preset ladder, minus rungs that cannot work for this method."""
    if is_metagga(method):
        return tuple(p for p in SCF_PRESETS if p["label"] != "soscf")
    return SCF_PRESETS
