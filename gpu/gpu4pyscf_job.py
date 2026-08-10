#!/usr/bin/env python3
"""Remote GPU worker: UKS/RKS + TDDFT for the NiMROD sensor, on a Colab VM.

This file is deliberately **self-contained**.  It is shipped to a freshly
allocated Colab virtual machine by ``gpu/launch_colab.sh`` (``colab run``),
where none of the NiMROD package, none of its configuration and none of Psi4
exist.  Everything it needs -- unit conversions, the ``<S^2>`` expression, the
population analysis, the output schema -- is therefore duplicated here rather
than imported.  The duplication is intentional and the values are copied
verbatim from :mod:`nimrod.config` and :mod:`nimrod.spin`; if those change, this
file must be changed with them.

Why a GPU at all
----------------
Psi4 is CPU-only.  No amount of GPU hardware makes a Psi4 job faster, so the
offload path cannot be "run Psi4 somewhere else" -- it has to be a different
program.  The program is `GPU4PySCF <https://github.com/pyscf/gpu4pyscf>`_, a
CUDA reimplementation of PySCF's integral and exchange-correlation kernels that
exposes the ordinary PySCF object API.  The parts of NiMROD that hurt on four
CPU cores are exactly the parts GPU4PySCF is good at:

* the 54-atom ``sensor_assembly()`` at UKS/def2-TZVP -- roughly 900 basis
  functions, an open-shell SCF that has to be converged at every point of the
  degradation scan; and
* the linear-response TDDFT on top of it, whose cost is dominated by repeated
  Coulomb/exchange builds on trial vectors -- the same J/K engine the SCF uses,
  called tens of times more often.

Both are dense-linear-algebra bound with density fitting, which is where a
single A100 replaces a large number of CPU cores.

What this script guarantees
---------------------------
1. **It says which path it took.**  GPU4PySCF, or plain CPU PySCF, is reported
   in ``properties.gpu_provenance.backend`` and on stderr.  It never pretends a
   CPU run was a GPU run.
2. **It returns the same observables as the Psi4 path.**  The output JSON has
   the exact field names of :class:`nimrod.psi4_driver.JobResult`, and
   ``properties`` carries the exact keys produced by
   :func:`nimrod.spin.spin_properties` and :func:`nimrod.excited._tdscf_hook`.
   ``<S^2>``, the Mulliken and Löwdin spin populations are recomputed here from
   the density matrices with the *same* formulae, not taken from a library
   routine that might define them differently.
3. **It fails loudly.**  Any failure exits non-zero so that ``colab run``
   propagates the exit code to :mod:`nimrod.gpu_bridge`.

Getting the answer back off the VM
----------------------------------
``colab run`` provisions a VM, runs the script and *destroys the VM*.  A file
written to ``--out`` on that VM is therefore unreachable the moment the job
finishes.  The result is consequently also printed to **stdout**, framed by
:data:`JSON_BEGIN` / :data:`JSON_END` sentinels, and that stdout copy is what
the launcher actually harvests.  The sentinels are required because the CLI
mixes its own chatter and (when unauthenticated) an OAuth prompt into the same
streams.

Exit codes
----------
==== =====================================================================
0    success
1    unexpected error (traceback printed to stderr)
2    invalid arguments (argparse's own convention, also used by our checks)
3    SCF did not converge
4    TDDFT was requested but failed or did not converge
5    PySCF could not be imported or installed at all
==== =====================================================================
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

# --------------------------------------------------------------------------
# Constants copied from nimrod.config (CODATA 2018, as used by Psi4)
# --------------------------------------------------------------------------

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_CM = 219474.6313632
HARTREE_TO_NM = 45.56335252767  # nm = HARTREE_TO_NM / E_hartree

#: Copied from nimrod.excited.DARK_THRESHOLD.
DARK_THRESHOLD = 1.0e-3

#: Copied from nimrod.excited._MULTIPLICITY_NAME.
MULTIPLICITY_NAME = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}

#: stdout framing for the result document.  ``colab run`` sends the script's
#: stdout to our stdout and its own ``[colab]`` chatter to stderr, but an
#: unauthenticated CLI also writes ``Enter the authorization code:`` to stdout,
#: so the payload must be unambiguously delimited.
JSON_BEGIN = "===NIMROD-GPU-JSON-BEGIN==="
JSON_END = "===NIMROD-GPU-JSON-END==="

#: Element symbol -> atomic number, for Mulliken charges.  Copied from
#: nimrod.geometry.ATOMIC_NUMBER and extended to cover a generic .xyz file.
ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
}


# --------------------------------------------------------------------------
# Functional and basis translation, Psi4 -> PySCF
# --------------------------------------------------------------------------
#
# Most names are shared, but three of the functionals on
# nimrod.config.FUNCTIONAL_LADDER carry real definitional differences between
# the two codes.  Silently running a different functional than the Psi4 branch
# ran would poison every comparison this offload path exists to make, so each
# known difference is translated to an explicit libxc identifier where that
# removes the ambiguity, and warned about where it does not.

#: Psi4 method string -> (PySCF ``xc`` string, warning or "").
FUNCTIONAL_MAP: dict[str, tuple[str, str]] = {
    # Pure GGAs: identical definitions, nothing to say.
    "bp86": ("bp86", ""),
    "pbe": ("pbe", ""),
    # B3LYP is the classic trap: the hybrid is defined with an LDA correlation
    # component that is VWN3(RPA) in the original Gaussian/Psi4 definition and
    # VWN5 in the "B3LYP5" variant.  PySCF has shipped both as the meaning of
    # the bare string "b3lyp" at different times.  Naming the libxc functional
    # explicitly pins it to the VWN(RPA) form that Psi4's "b3lyp" uses.
    "b3lyp": (
        "HYB_GGA_XC_B3LYP",
        "b3lyp mapped to the explicit libxc HYB_GGA_XC_B3LYP (VWN_RPA "
        "correlation) to match Psi4's b3lyp; the bare PySCF string 'b3lyp' has "
        "denoted the VWN5 variant in some releases and would differ by "
        "~1 mEh/atom.",
    ),
    "b3lyp5": ("HYB_GGA_XC_B3LYP5", ""),
    "pbe0": ("pbe0", ""),
    # Range-separated hybrid plus an empirical dispersion tail.  libxc's
    # WB97X_D supplies only the XC part; the -D2 damped dispersion is applied
    # by the host code and PySCF does not add it automatically.
    "wb97x-d": (
        "wb97x-d",
        "wb97x-d: libxc supplies the range-separated XC only. Psi4 adds the "
        "-D2 empirical dispersion on top; PySCF does not unless dftd3 is "
        "installed and attached. Total energies are therefore NOT comparable "
        "with the Psi4 wb97x-d numbers, though excitation energies (a "
        "difference of two states with the same dispersion term) very nearly "
        "are.",
    ),
    # Meta-GGAs.  Psi4's TDSCF kernel refuses these entirely (see the
    # nimrod.excited docstring), which is precisely why they are interesting
    # here: PySCF's TDDFT does support meta-GGA kernels.
    "tpss": ("tpss", ""),
    "tpssh": (
        "tpssh",
        "tpssh is a meta-GGA hybrid. Psi4's TDSCF Vx kernel cannot run it at "
        "all, so any excited-state number produced here has no Psi4 "
        "counterpart to be checked against.",
    ),
    "m06": (
        "m06",
        "m06 is a meta-GGA hybrid; Psi4's TDSCF kernel cannot run it, so "
        "excited-state results have no Psi4 counterpart.",
    ),
    "hf": ("hf", ""),
}

#: Basis-set names that differ between the two libraries.  Both codes read the
#: Basis Set Exchange naming convention, so this is short by design.
BASIS_MAP: dict[str, str] = {
    "def2-svp": "def2-svp",
    "def2-svpd": "def2-svpd",
    "def2-tzvp": "def2-tzvp",
    "def2-tzvpp": "def2-tzvpp",
    "cc-pvdz": "ccpvdz",
    "cc-pvtz": "ccpvtz",
    "aug-cc-pvdz": "augccpvdz",
    "6-31g": "631g",
    "6-31g*": "631g*",
    "6-31gs": "631g*",
    "sto-3g": "sto3g",
}

#: pip requirements attempted, in order, when gpu4pyscf is missing on the VM.
#: Colab images have been CUDA 12 for a long while; the cuda11x wheel is kept
#: as a second chance rather than as a probe of the driver version.
GPU_PACKAGES: tuple[str, ...] = ("gpu4pyscf-cuda12x", "cutensor-cu12")
GPU_PACKAGES_FALLBACK: tuple[str, ...] = ("gpu4pyscf-cuda11x",)
CPU_PACKAGES: tuple[str, ...] = ("pyscf",)


class JobError(RuntimeError):
    """A failure that should be reported as a structured result, not a crash."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# --------------------------------------------------------------------------
# Geometry input
# --------------------------------------------------------------------------


def parse_xyz(text: str) -> tuple[list[str], list[tuple[float, float, float]], str]:
    """Parse a standard .xyz file into symbols, coordinates (angstrom), comment.

    Tolerant of a missing count line: if the first line is not an integer the
    whole file is treated as a bare Cartesian block, which is the format
    :meth:`nimrod.geometry.Structure.to_psi4` emits.
    """
    lines = [ln for ln in text.strip().splitlines()]
    if not lines:
        raise JobError("geometry is empty", 2)

    comment = ""
    body = lines
    try:
        count = int(lines[0].split()[0])
    except (ValueError, IndexError):
        count = None
    if count is not None:
        if len(lines) < count + 2:
            raise JobError(
                f"xyz header declares {count} atoms but only {len(lines) - 2} "
                "coordinate lines follow",
                2,
            )
        comment = lines[1].strip()
        body = lines[2 : 2 + count]

    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for lineno, line in enumerate(body, start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 4:
            raise JobError(f"malformed coordinate line {lineno}: {line!r}", 2)
        symbol = parts[0].capitalize()
        if symbol not in ATOMIC_NUMBER:
            raise JobError(f"unknown element {parts[0]!r} on line {lineno}", 2)
        try:
            xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError as exc:
            raise JobError(f"non-numeric coordinate on line {lineno}: {exc}", 2) from None
        symbols.append(symbol)
        coords.append(xyz)

    if not symbols:
        raise JobError("geometry contains no atoms", 2)
    return symbols, coords, comment


def decode_xyz_b64(blob: str) -> str:
    """Decode the gzip+base64 geometry the launcher passes on the command line.

    ``colab run`` uploads no data files -- it ships one script and its argv.
    Compressing the geometry into a single argv element is what lets the
    ephemeral one-shot mode work at all; a 54-atom structure comes to well under
    two kilobytes, far inside any argv limit.
    """
    try:
        raw = base64.b64decode(blob.encode("ascii"), validate=True)
    except Exception as exc:
        raise JobError(f"--xyz-b64 is not valid base64: {exc}", 2) from None
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
            return fh.read().decode("utf-8")
    except OSError:
        # Allow an uncompressed base64 payload too; cheap and forgiving.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JobError(f"--xyz-b64 decodes to neither gzip nor utf-8: {exc}", 2) from None


def geometry_block(symbols: Sequence[str], coords: Sequence[Sequence[float]]) -> str:
    """Cartesian block in the same layout as ``Structure.to_psi4()``."""
    return "\n".join(
        f"{s:<3s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}"
        for s, c in zip(symbols, coords)
    )


# --------------------------------------------------------------------------
# Argument handling and the plan
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu4pyscf_job.py",
        description="UKS/RKS + TDDFT on a GPU (GPU4PySCF) or CPU (PySCF) backend, "
                    "returning NiMROD-compatible JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--xyz", metavar="FILE", help="path to a .xyz file on this machine")
    source.add_argument(
        "--xyz-b64",
        metavar="BLOB",
        help="gzip+base64 of the .xyz contents, for shipping the geometry through argv",
    )

    parser.add_argument("--charge", type=int, default=0, help="total charge")
    parser.add_argument(
        "--multiplicity", "--mult", dest="multiplicity", type=int, default=1,
        help="spin multiplicity 2S+1; >1 forces an unrestricted (UKS) reference",
    )
    parser.add_argument("--functional", default="pbe0", help="Psi4-style functional name")
    parser.add_argument("--basis", default="def2-svp", help="Psi4-style basis-set name")
    parser.add_argument(
        "--n-states", "--nstates", dest="n_states", type=int, default=0,
        help="number of TDDFT roots; 0 skips the excited-state step entirely",
    )
    parser.add_argument(
        "--triplets", choices=("none", "also", "only"), default="none",
        help="closed-shell references only: whether to solve the triplet branch "
             "(mirrors Psi4's TDSCF_TRIPLETS)",
    )
    parser.add_argument("--tda", action="store_true", help="Tamm-Dancoff approximation instead of full TDDFT")
    parser.add_argument("--out", metavar="FILE", default="result.json", help="where to write the JSON result on this machine")
    parser.add_argument("--label", default="", help="free-text label carried into the result")

    parser.add_argument("--no-df", action="store_true", help="disable density fitting (Psi4's SCF_TYPE=DF is the default there too)")
    parser.add_argument("--grid-level", type=int, default=3, help="DFT quadrature grid level")
    parser.add_argument("--conv-tol", type=float, default=1e-9, help="SCF energy convergence threshold")
    parser.add_argument("--max-cycle", type=int, default=200, help="SCF iteration limit per attempt")
    parser.add_argument("--memory-mb", type=int, default=0, help="PySCF max_memory in MB; 0 leaves the PySCF default")

    parser.add_argument("--cpu-only", action="store_true", help="never attempt GPU4PySCF, even if a GPU is present")
    parser.add_argument("--no-install", action="store_true", help="never pip-install anything; fail if imports are missing")
    parser.add_argument("--verbose", type=int, default=3, help="PySCF verbosity")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate arguments and print the plan without importing pyscf; "
             "this is the only mode runnable on a machine with no GPU stack",
    )
    return parser


def resolve_functional(name: str) -> tuple[str, list[str]]:
    """Translate a Psi4 functional name to a PySCF ``xc`` string plus warnings."""
    key = name.strip().lower()
    if key in FUNCTIONAL_MAP:
        xc, warning = FUNCTIONAL_MAP[key]
        return xc, ([warning] if warning else [])
    return key, [
        f"functional {name!r} is not in this script's Psi4->PySCF translation "
        "table; it is passed to PySCF verbatim and may denote a different "
        "functional than Psi4's name of the same spelling."
    ]


def resolve_basis(name: str) -> tuple[str, list[str]]:
    key = name.strip().lower()
    if key in BASIS_MAP:
        return BASIS_MAP[key], []
    return key, [
        f"basis {name!r} is not in this script's translation table; passed to "
        "PySCF verbatim."
    ]


def validate(args: argparse.Namespace) -> None:
    """Reject argument combinations that cannot describe a real calculation."""
    if args.multiplicity < 1:
        raise JobError(f"multiplicity must be >= 1, got {args.multiplicity}", 2)
    if args.n_states < 0:
        raise JobError(f"--n-states must be >= 0, got {args.n_states}", 2)
    if args.grid_level not in range(0, 10):
        raise JobError(f"--grid-level must be 0-9, got {args.grid_level}", 2)
    if args.conv_tol <= 0:
        raise JobError(f"--conv-tol must be positive, got {args.conv_tol}", 2)
    if args.max_cycle < 1:
        raise JobError(f"--max-cycle must be >= 1, got {args.max_cycle}", 2)
    if args.xyz is not None and not os.path.exists(args.xyz):
        raise JobError(f"geometry file not found: {args.xyz}", 2)
    if args.triplets != "none" and args.multiplicity != 1:
        raise JobError(
            "--triplets applies only to a closed-shell (restricted) reference; "
            f"multiplicity {args.multiplicity} gives an unrestricted reference "
            "whose roots are not spin-classified branches",
            2,
        )


def load_geometry(args: argparse.Namespace) -> tuple[list[str], list[tuple[float, float, float]], str]:
    if args.xyz_b64:
        return parse_xyz(decode_xyz_b64(args.xyz_b64))
    with open(args.xyz, encoding="utf-8") as fh:
        return parse_xyz(fh.read())


def electron_count(symbols: Sequence[str], charge: int) -> int:
    return sum(ATOMIC_NUMBER[s] for s in symbols) - charge


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Everything the job will do, decided before any heavy import.

    Producing this separately from executing it is what makes ``--dry-run``
    meaningful: the plan is the *same* object the real run consumes.
    """
    symbols, coords, comment = load_geometry(args)
    xc, xc_warnings = resolve_functional(args.functional)
    basis, basis_warnings = resolve_basis(args.basis)

    n_elec = electron_count(symbols, args.charge)
    n_unpaired = args.multiplicity - 1
    if (n_elec - n_unpaired) % 2 != 0:
        raise JobError(
            f"{n_elec} electrons cannot form multiplicity {args.multiplicity}: "
            f"n_electrons - (2S) = {n_elec - n_unpaired} must be even. "
            f"For this formula use multiplicity "
            f"{1 if n_elec % 2 == 0 else 2} (or any value of the same parity).",
            2,
        )
    if n_unpaired > n_elec:
        raise JobError(
            f"multiplicity {args.multiplicity} needs {n_unpaired} unpaired "
            f"electrons but the system has only {n_elec}",
            2,
        )

    restricted = args.multiplicity == 1
    reference = "rks" if restricted else "uks"
    if xc == "hf":
        reference = "rhf" if restricted else "uhf"

    warnings = xc_warnings + basis_warnings
    if args.triplets == "also":
        # Psi4's TDSCF_TRIPLETS=ALSO solves both branches in one job. PySCF's
        # `td.singlet` is a boolean selecting one diagonalisation, so "also"
        # cannot be honoured -- and quietly returning singlets under a flag
        # that asked for both is exactly the kind of silent wrongness this
        # script is written to avoid.
        warnings.append(
            "--triplets also is not supported by PySCF: td.singlet selects a "
            "single branch, so ONLY SINGLETS were computed. Run the job twice, "
            "with --triplets none and --triplets only, to get both."
        )

    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    formula = "".join(
        f"{el}{counts[el]}" if counts[el] > 1 else el
        for el in sorted(counts, key=lambda e: (e != "C", e != "H", e))
    )

    return {
        "natom": len(symbols),
        "formula": formula,
        "comment": comment,
        "symbols": symbols,
        "coords": [list(c) for c in coords],
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "n_electrons": n_elec,
        "n_unpaired": n_unpaired,
        "restricted": restricted,
        "reference": reference,
        "functional": args.functional.strip().lower(),
        "xc": xc,
        "basis_requested": args.basis.strip().lower(),
        "basis": basis,
        "density_fitting": not args.no_df,
        "grid_level": args.grid_level,
        "conv_tol": args.conv_tol,
        "max_cycle": args.max_cycle,
        "n_states": args.n_states,
        "tddft": args.n_states > 0,
        "tda": bool(args.tda),
        "triplets": args.triplets,
        "cpu_only": bool(args.cpu_only),
        "allow_install": not args.no_install,
        "out": args.out,
        "label": args.label,
        "warnings": warnings,
    }


def fingerprint(plan: dict[str, Any]) -> str:
    """A cache key with the same construction as :meth:`JobSpec.fingerprint`.

    ``__backend__`` is included deliberately.  Without it a GPU4PySCF result and
    a Psi4 result for the same specification would hash identically and the GPU
    answer could be served out of ``data/cache`` as though Psi4 had produced it.
    The two are different programs with different integral thresholds; they must
    never share a cache slot.
    """
    payload = {
        "geometry": " ".join(geometry_block(plan["symbols"], plan["coords"]).split()),
        "method": plan["functional"],
        "basis": plan["basis_requested"],
        "charge": plan["charge"],
        "multiplicity": plan["multiplicity"],
        "reference": plan["reference"],
        "options": {
            "df": plan["density_fitting"],
            "grid_level": plan["grid_level"],
            "n_states": plan["n_states"],
            "tda": plan["tda"],
            "triplets": plan["triplets"],
        },
        "__backend__": "gpu4pyscf",
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def format_plan(plan: dict[str, Any]) -> str:
    """Human-readable plan, for the dry run and for the stderr preamble."""
    tddft = (
        f"{plan['n_states']} roots, "
        f"{'TDA' if plan['tda'] else 'full TDDFT'}, triplets={plan['triplets']}"
        if plan["tddft"] else "skipped (--n-states 0)"
    )
    lines = [
        "NiMROD GPU offload plan",
        "=======================",
        f"  system          : {plan['formula']}  ({plan['natom']} atoms"
        + (f", {plan['comment']}" if plan["comment"] else "") + ")",
        f"  electrons       : {plan['n_electrons']}  "
        f"(charge {plan['charge']:+d}, multiplicity {plan['multiplicity']}, "
        f"{plan['n_unpaired']} unpaired)",
        f"  reference       : {plan['reference'].upper()}"
        f"  [{'restricted' if plan['restricted'] else 'unrestricted'}]",
        f"  functional      : {plan['functional']}  ->  PySCF xc={plan['xc']!r}",
        f"  basis           : {plan['basis_requested']}  ->  PySCF {plan['basis']!r}",
        f"  density fitting : {'on' if plan['density_fitting'] else 'off'}",
        f"  grid level      : {plan['grid_level']}",
        f"  SCF             : conv_tol={plan['conv_tol']:g}, max_cycle={plan['max_cycle']},"
        " DIIS then SOSCF(newton) on failure",
        f"  TDDFT           : {tddft}",
        f"  backend order   : " + ("CPU PySCF only (--cpu-only)" if plan["cpu_only"]
                                   else "GPU4PySCF if a CUDA device is present, else CPU PySCF"),
        f"  install allowed : {'yes' if plan['allow_install'] else 'no (--no-install)'}",
        f"  result JSON     : {plan['out']}  (and stdout, between sentinels)",
        f"  fingerprint     : {fingerprint(plan)}",
    ]
    for warning in plan["warnings"]:
        lines.append(f"  WARNING         : {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def detect_nvidia_gpu() -> tuple[bool, str]:
    """Is there a usable CUDA device?  Answered without importing anything heavy.

    ``nvidia-smi`` is asked rather than cupy because cupy may not be installed
    yet -- and on a CPU Colab runtime, installing the CUDA wheels only to find
    out there is no device wastes several minutes of billable VM time.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, "nvidia-smi not found on PATH: this runtime has no NVIDIA driver"
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi could not be executed: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip()[:200]}"
    names = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not names:
        return False, "nvidia-smi reported no devices"
    return True, "; ".join(names)


def pip_install(packages: Sequence[str]) -> tuple[bool, str]:
    """Install packages with the running interpreter's pip."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    log(f"installing: {' '.join(packages)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
        return False, "\n".join(tail)
    return True, ""


def import_backend(*, cpu_only: bool, allow_install: bool) -> dict[str, Any]:
    """Import GPU4PySCF if we can, plain PySCF otherwise.

    Returns a provenance dict.  ``backend`` is ``"gpu4pyscf"`` or ``"pyscf-cpu"``
    and ``backend_reason`` always explains the choice, including when the GPU
    path was attempted and abandoned.  Nothing here is silent.
    """
    provenance: dict[str, Any] = {
        "backend": None,
        "backend_reason": "",
        "gpu_present": False,
        "gpu_name": "",
        "attempted_gpu": False,
        "install_performed": [],
        "versions": {},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    gpu_present, gpu_detail = detect_nvidia_gpu()
    provenance["gpu_present"] = gpu_present
    provenance["gpu_name"] = gpu_detail if gpu_present else ""

    if cpu_only:
        provenance["backend_reason"] = "--cpu-only was requested"
    elif not gpu_present:
        provenance["backend_reason"] = f"no GPU: {gpu_detail}"
    else:
        provenance["attempted_gpu"] = True
        log(f"CUDA device detected: {gpu_detail}")
        gpu_error = _try_import_gpu4pyscf(provenance, allow_install)
        if gpu_error is None:
            provenance["backend"] = "gpu4pyscf"
            provenance["backend_reason"] = f"GPU4PySCF imported; device: {gpu_detail}"
            _record_versions(provenance)
            log(f"BACKEND = gpu4pyscf ({gpu_detail})")
            return provenance
        provenance["backend_reason"] = (
            f"GPU present ({gpu_detail}) but GPU4PySCF is unusable, falling back "
            f"to CPU PySCF: {gpu_error}"
        )
        log(f"GPU4PySCF unusable, falling back to CPU: {gpu_error}")

    # CPU path.
    try:
        import pyscf  # noqa: F401
    except ImportError:
        if not allow_install:
            raise JobError(
                "pyscf is not installed and --no-install forbids installing it", 5
            ) from None
        ok, detail = pip_install(CPU_PACKAGES)
        if not ok:
            raise JobError(f"could not install pyscf: {detail}", 5)
        provenance["install_performed"].append(list(CPU_PACKAGES))
        try:
            import pyscf  # noqa: F401,F811
        except ImportError as exc:
            raise JobError(f"pyscf installed but will not import: {exc}", 5) from None

    provenance["backend"] = "pyscf-cpu"
    _record_versions(provenance)
    log(f"BACKEND = pyscf-cpu ({provenance['backend_reason']})")
    return provenance


def _try_import_gpu4pyscf(provenance: dict[str, Any], allow_install: bool) -> str | None:
    """Import gpu4pyscf, installing if permitted.  Returns None on success."""
    try:
        import gpu4pyscf  # noqa: F401
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return "cupy reports zero CUDA devices"
        return None
    except Exception as first_exc:  # ImportError, CUDARuntimeError, ...
        if not allow_install:
            return f"import failed and --no-install forbids installing: {first_exc}"

    for packages in (GPU_PACKAGES, GPU_PACKAGES_FALLBACK):
        ok, detail = pip_install(packages)
        if not ok:
            log(f"install of {packages} failed: {detail}")
            continue
        provenance["install_performed"].append(list(packages))
        try:
            import gpu4pyscf  # noqa: F401,F811
            import cupy  # noqa: F811
            if cupy.cuda.runtime.getDeviceCount() < 1:
                return "cupy reports zero CUDA devices after install"
            return None
        except Exception as exc:
            log(f"gpu4pyscf still unusable after installing {packages}: {exc}")
            continue
    return "every gpu4pyscf install/import attempt failed (see stderr)"


def _record_versions(provenance: dict[str, Any]) -> None:
    for name in ("pyscf", "gpu4pyscf", "cupy", "numpy"):
        try:
            module = __import__(name)
        except Exception:
            continue
        provenance["versions"][name] = str(getattr(module, "__version__", "unknown"))


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------


def to_numpy(array: Any):
    """Bring a possibly-CuPy array back to the host.

    GPU4PySCF returns ``cupy.ndarray`` from most of its accessors.  Every
    analysis routine below is written against numpy, so everything crosses this
    function exactly once, on the way out of the SCF object.
    """
    import numpy as np

    if array is None:
        return None
    if isinstance(array, (tuple, list)):
        return type(array)(to_numpy(a) for a in array)
    get = getattr(array, "get", None)
    if get is not None and not isinstance(array, np.ndarray):
        try:
            return np.asarray(get())
        except Exception:
            pass
    return np.asarray(array)


def ideal_s2(multiplicity: int) -> float:
    """S(S+1) for a pure spin state.  Copied from :func:`nimrod.spin.ideal_s2`."""
    s = 0.5 * (multiplicity - 1)
    return s * (s + 1.0)


def spin_squared(mo_a, mo_b, n_alpha: int, n_beta: int, overlap) -> float:
    """``<S^2>`` of a single unrestricted determinant.

    The identical expression to :func:`nimrod.spin.spin_squared`::

        <S^2> = Sz(Sz + 1) + N_beta - sum_ij |<phi_i^alpha | phi_j^beta>|^2

    computed here from MO coefficients and the AO overlap rather than taken from
    ``mf.spin_square()``, so that the GPU and Psi4 branches are provably
    evaluating the same formula and not two library conventions.
    """
    import numpy as np

    sz = 0.5 * (n_alpha - n_beta)
    occ_a = np.asarray(mo_a)[:, :n_alpha]
    occ_b = np.asarray(mo_b)[:, :n_beta]
    s_ab = occ_a.T @ np.asarray(overlap) @ occ_b
    return float(sz * (sz + 1.0) + n_beta - np.sum(s_ab * s_ab))


def _accumulate_on_atoms(diagonal, ao_slices, natom: int):
    """Sum an AO-resolved diagonal onto atoms, as :mod:`nimrod.spin` does."""
    import numpy as np

    out = np.zeros(natom)
    for atom in range(natom):
        start, stop = int(ao_slices[atom][2]), int(ao_slices[atom][3])
        out[atom] = float(np.sum(diagonal[start:stop]))
    return out


def population_analysis(mol, dm, *, unrestricted: bool) -> dict[str, Any]:
    """Mulliken and Löwdin spin populations plus Mulliken charges.

    Same partitions, same formulae and same output key names as
    :func:`nimrod.spin.atomic_spin_populations`:

    * Mulliken diagonal ``(P S)_mu,mu``, summed over the AOs of each atom;
    * Löwdin diagonal ``(S^1/2 P S^1/2)_mu,mu``, likewise.

    Mulliken is reported because it is what everyone quotes; Löwdin because it
    is far less basis-set-sensitive, and agreement between them is the cheapest
    check that a localisation claim is real.
    """
    import numpy as np

    overlap = np.asarray(to_numpy(mol.intor_symmetric("int1e_ovlp")))
    natom = mol.natm
    ao_slices = mol.aoslice_by_atom()
    symbols = [mol.atom_symbol(i) for i in range(natom)]

    if unrestricted:
        dm_a, dm_b = np.asarray(dm[0]), np.asarray(dm[1])
    else:
        dm_a = np.asarray(dm) * 0.5
        dm_b = dm_a
    dm_total = dm_a + dm_b
    dm_spin = dm_a - dm_b

    mulliken_diag = np.einsum("ij,ji->i", dm_spin, overlap)
    mulliken = _accumulate_on_atoms(mulliken_diag, ao_slices, natom)

    # Symmetric orthogonalisation S^1/2 from the eigendecomposition of S.
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    eigenvalues = np.clip(eigenvalues, 1e-14, None)
    s_half = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    lowdin_diag = np.einsum("ij,jk,ki->i", s_half, dm_spin, s_half)
    lowdin = _accumulate_on_atoms(lowdin_diag, ao_slices, natom)

    charge_diag = np.einsum("ij,ji->i", dm_total, overlap)
    gross = _accumulate_on_atoms(charge_diag, ao_slices, natom)
    charges = np.array([float(mol.atom_charge(i)) for i in range(natom)]) - gross

    return {
        "symbols": symbols,
        "natom": natom,
        "mulliken": mulliken,
        "lowdin": lowdin,
        "charges": charges,
    }


# --------------------------------------------------------------------------
# The calculation
# --------------------------------------------------------------------------


def build_mol(plan: dict[str, Any], *, verbose: int, memory_mb: int):
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [
        (symbol, tuple(coord))
        for symbol, coord in zip(plan["symbols"], plan["coords"])
    ]
    mol.unit = "Angstrom"
    mol.basis = plan["basis"]
    mol.charge = plan["charge"]
    # PySCF's `spin` is 2S = n_alpha - n_beta, NOT the multiplicity.  Getting
    # this wrong is the single most common PySCF mistake and it fails silently
    # for closed-shell systems, where both conventions give the same answer.
    mol.spin = plan["multiplicity"] - 1
    mol.verbose = verbose
    if memory_mb > 0:
        mol.max_memory = memory_mb
    mol.build()
    return mol


def make_scf(mol, plan: dict[str, Any], backend: str):
    """Construct the SCF object on the requested backend."""
    restricted = plan["restricted"]
    xc = plan["xc"]

    if backend == "gpu4pyscf":
        from gpu4pyscf import dft as gpu_dft

        mf = gpu_dft.RKS(mol, xc=xc) if restricted else gpu_dft.UKS(mol, xc=xc)
    else:
        from pyscf import dft as cpu_dft

        mf = cpu_dft.RKS(mol, xc=xc) if restricted else cpu_dft.UKS(mol, xc=xc)

    if plan["density_fitting"]:
        # Matches Psi4's SCF_TYPE=DF, which every preset in
        # nimrod.config.SCF_PRESETS uses; on the GPU it is also the only path
        # with well-optimised kernels.
        mf = mf.density_fit()
    mf.grids.level = plan["grid_level"]
    mf.conv_tol = plan["conv_tol"]
    mf.max_cycle = plan["max_cycle"]
    return mf


def run_scf(mol, plan: dict[str, Any], backend: str) -> tuple[Any, float, str]:
    """Converge the SCF, walking a short preset ladder.

    The ladder is the PySCF counterpart of
    :data:`nimrod.config.SCF_PRESETS`: ordinary DIIS first, then a second-order
    (Newton / SOSCF) solver restarted from the DIIS orbitals, which is what
    rescues most of the open-shell 3d-metal cases the Psi4 driver also has to
    fight.  Two rungs, not four, because a Colab VM is billed by the minute and
    a job that needs the desperate end of the Psi4 ladder should be diagnosed
    interactively rather than ground through on rented hardware.
    """
    label = "diis-df" if plan["density_fitting"] else "diis"
    mf = make_scf(mol, plan, backend)

    log(f"SCF attempt [{label}] ...")
    energy = mf.kernel()
    if mf.converged:
        log(f"SCF converged [{label}]: E = {float(energy):.10f} Eh")
        return mf, float(energy), label

    log(f"SCF did NOT converge [{label}]; restarting with the second-order (newton) solver")
    try:
        newton = mf.newton()
    except (AttributeError, NotImplementedError) as exc:
        raise JobError(
            f"SCF did not converge with {label} and this backend offers no "
            f"second-order solver ({type(exc).__name__}: {exc})",
            3,
        ) from None

    newton.max_cycle = plan["max_cycle"]
    newton.conv_tol = plan["conv_tol"]
    # The documented PySCF restart: hand the unconverged DIIS orbitals to the
    # Newton solver as its starting guess.
    energy = newton.kernel(mf.mo_coeff, mf.mo_occ)
    if newton.converged:
        log(f"SCF converged [{label}+soscf]: E = {float(energy):.10f} Eh")
        return newton, float(energy), f"{label}+soscf"

    raise JobError(
        "SCF failed to converge with DIIS and with the second-order solver "
        f"(last energy {float(energy):.10f} Eh)",
        3,
    )


def make_state(
    index: int,
    omega_hartree: float,
    oscillator_strength: float | None,
    spin_label: str,
    *,
    transition_dipole: list[float] | None = None,
) -> dict[str, Any]:
    """One root, with exactly the fields of :class:`nimrod.excited.ExcitedState`."""
    omega = float(omega_hartree)
    nm = HARTREE_TO_NM / omega if omega > 1e-12 else math.inf
    return {
        "index": index,
        "energy_ev": omega * HARTREE_TO_EV,
        "energy_nm": nm,
        "energy_cm": omega * HARTREE_TO_CM,
        "oscillator_strength": None if oscillator_strength is None else float(oscillator_strength),
        "spin_label": spin_label,
        "energy_hartree": omega,
        "symmetry": "A",   # every NiMROD molecule is built in C1
        "dominant_pair": None,
        "dominant_weight": None,
        "transition_dipole": transition_dipole,
        "oscillator_strength_available": oscillator_strength is not None,
    }


def run_tddft(mf, plan: dict[str, Any], backend: str) -> dict[str, Any]:
    """Linear-response TDDFT on the converged reference.

    GPU4PySCF's TDDFT coverage lags its SCF coverage and varies by release.  If
    the GPU object cannot produce a TDDFT solver we move the *converged*
    wavefunction to the CPU with ``to_cpu()`` and solve there: the SCF, which is
    the part that had to be iterated to convergence, still ran on the GPU, and
    the fallback is recorded in ``tddft_backend`` rather than hidden.
    """
    import numpy as np

    restricted = plan["restricted"]
    n_states = plan["n_states"]
    tddft_backend = backend
    fallback_reason = ""

    def build(solver_mf):
        td = solver_mf.TDA() if plan["tda"] else solver_mf.TDDFT()
        td.nstates = n_states
        if restricted:
            # Mirrors Psi4's TDSCF_TRIPLETS: 'none' -> singlets only,
            # 'only' -> the triplet branch.  PySCF has no 'also' mode; the two
            # branches are separate diagonalisations, and this script runs one.
            td.singlet = plan["triplets"] != "only"
        return td

    td = None
    try:
        td = build(mf)
        td.kernel()
    except Exception as exc:
        if backend != "gpu4pyscf":
            raise JobError(
                f"TDDFT failed on the CPU backend: {type(exc).__name__}: {exc}", 4
            ) from None
        fallback_reason = f"{type(exc).__name__}: {exc}"
        log(f"GPU TDDFT unavailable ({fallback_reason}); moving the converged SCF to CPU")
        try:
            cpu_mf = mf.to_cpu()
            td = build(cpu_mf)
            td.kernel()
            tddft_backend = "pyscf-cpu (fallback from gpu4pyscf)"
        except Exception as exc2:
            raise JobError(
                f"TDDFT failed on the GPU ({fallback_reason}) and on the CPU "
                f"fallback ({type(exc2).__name__}: {exc2})",
                4,
            ) from None

    energies = np.atleast_1d(np.asarray(to_numpy(td.e), dtype=float))
    if energies.size == 0:
        raise JobError("TDDFT returned no roots", 4)

    converged = getattr(td, "converged", None)
    if converged is None:
        converged_list = [True] * energies.size
    else:
        converged_arr = np.atleast_1d(np.asarray(to_numpy(converged)))
        converged_list = [bool(x) for x in converged_arr]
    if len(converged_list) != energies.size:
        converged_list = (converged_list + [False] * energies.size)[: energies.size]

    strengths: list[float | None]
    try:
        strengths = [float(f) for f in np.atleast_1d(np.asarray(to_numpy(td.oscillator_strength())))]
    except Exception as exc:
        log(f"oscillator strengths unavailable: {type(exc).__name__}: {exc}")
        strengths = [None] * energies.size

    dipoles: list[list[float] | None]
    try:
        raw = np.atleast_2d(np.asarray(to_numpy(td.transition_dipole()), dtype=float))
        dipoles = [[float(x) for x in row] for row in raw]
    except Exception:
        dipoles = [None] * energies.size

    if restricted:
        spin_label = "triplet" if plan["triplets"] == "only" else "singlet"
    else:
        # An unrestricted reference has one solver branch, and its roots carry
        # the reference's own multiplicity (with higher-spin contamination).
        # This is the same override nimrod.excited._spin_label_for applies.
        spin_label = MULTIPLICITY_NAME.get(
            plan["multiplicity"], f"M={plan['multiplicity']}"
        )

    states = [
        make_state(
            i + 1,
            energies[i],
            strengths[i] if i < len(strengths) else None,
            spin_label,
            transition_dipole=dipoles[i] if i < len(dipoles) else None,
        )
        for i in range(energies.size)
    ]

    result: dict[str, Any] = {
        "states": states,
        "restricted": restricted,
        "tddft_backend": tddft_backend,
        "tddft_converged": all(converged_list),
        "tddft_root_converged": converged_list,
        "n_states_requested": n_states,
        "n_states_returned": len(states),
    }
    if fallback_reason:
        result["tddft_gpu_fallback_reason"] = fallback_reason
    return result


def log(message: str) -> None:
    """Progress goes to stderr; only the JSON document goes to stdout."""
    print(f"[nimrod-gpu] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Result assembly
# --------------------------------------------------------------------------


def empty_result(plan: dict[str, Any]) -> dict[str, Any]:
    """A JobResult-shaped document with nothing filled in yet.

    The key names and their order are those of
    :class:`nimrod.psi4_driver.JobResult`, so that
    ``JobResult(**json.load(fh))`` works with no translation layer at all.
    Everything that has no JobResult counterpart lives inside ``properties``.
    """
    return {
        "spec_fingerprint": fingerprint(plan),
        "label": plan["label"] or f"gpu:{plan['formula']}",
        "method": plan["functional"],
        "basis": plan["basis_requested"],
        "charge": plan["charge"],
        "multiplicity": plan["multiplicity"],
        "reference": plan["reference"],
        "converged": False,
        "energy": None,
        "preset": None,
        "wall_seconds": 0.0,
        "error": None,
        "properties": {},
    }


def emit(result: dict[str, Any], out_path: str) -> None:
    """Write the result to ``--out`` and to stdout between the sentinels.

    ``allow_nan`` is left at its default of True, matching
    :func:`nimrod.psi4_driver.store_cached` exactly.  That emits the bare
    ``Infinity`` / ``NaN`` literals, which are a Python extension rather than
    strict RFC 8259 JSON -- but Python's own ``json.load`` reads them straight
    back as floats, and the alternative (encoding them as strings) would make
    ``energy_nm`` a ``str`` on precisely the roots that matter.  A zero
    excitation energy is the signature of a triplet instability; it has to
    reach the report as a float infinity, the same way the Psi4 branch delivers
    it, not as text that blows up the first time anything formats it.
    """
    document = json.dumps(result, indent=2, default=str)

    if out_path:
        try:
            directory = os.path.dirname(os.path.abspath(out_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(document + "\n")
            log(f"wrote {out_path}")
        except OSError as exc:
            # Not fatal: the stdout copy is the one the launcher harvests, and
            # on an ephemeral `colab run` VM the file is discarded anyway.
            log(f"could not write {out_path}: {exc}")

    print(JSON_BEGIN, flush=True)
    print(document, flush=True)
    print(JSON_END, flush=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def execute(plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run the whole calculation and return the JobResult-shaped document."""
    started = time.time()
    result = empty_result(plan)

    # Import the backend FIRST. numpy arrives as a dependency of pyscf, so
    # importing it up here would turn "pyscf is missing and --no-install
    # forbids installing it" (exit 5, actionable) into a bare
    # ModuleNotFoundError for numpy (exit 1, misleading).
    provenance = import_backend(cpu_only=args.cpu_only, allow_install=not args.no_install)

    import numpy as np

    backend = provenance["backend"]
    provenance["warnings"] = list(plan["warnings"])
    provenance["xc"] = plan["xc"]
    provenance["basis"] = plan["basis"]
    provenance["density_fitting"] = plan["density_fitting"]
    provenance["grid_level"] = plan["grid_level"]

    mol = build_mol(plan, verbose=args.verbose, memory_mb=args.memory_mb)
    provenance["nbf"] = int(mol.nao_nr())
    log(f"{plan['formula']}: {mol.natm} atoms, {provenance['nbf']} basis functions")

    mf, energy, preset = run_scf(mol, plan, backend)
    result["converged"] = True
    result["energy"] = energy
    result["preset"] = f"{backend}:{preset}"

    properties: dict[str, Any] = {}

    unrestricted = not plan["restricted"]
    dm = to_numpy(mf.make_rdm1())
    populations = population_analysis(mol, dm, unrestricted=unrestricted)

    n_alpha = int(mol.nelec[0])
    n_beta = int(mol.nelec[1])
    if unrestricted:
        mo = to_numpy(mf.mo_coeff)
        overlap = to_numpy(mol.intor_symmetric("int1e_ovlp"))
        s2 = spin_squared(mo[0], mo[1], n_alpha, n_beta, overlap)
    else:
        s2 = 0.0

    max_index = int(np.argmax(np.abs(populations["mulliken"]))) if populations["natom"] else -1

    # These key names are exactly nimrod.spin.spin_properties'.
    properties.update({
        "s2": float(s2),
        "s2_ideal": ideal_s2(plan["multiplicity"]),
        "spin_contamination": float(s2) - ideal_s2(plan["multiplicity"]),
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "sz": 0.5 * (n_alpha - n_beta),
        "multiplicity": plan["multiplicity"],
        "natom": populations["natom"],
        "symbols": list(populations["symbols"]),
        "mulliken_spin": [float(x) for x in populations["mulliken"]],
        "lowdin_spin": [float(x) for x in populations["lowdin"]],
        "mulliken_charge": [float(x) for x in populations["charges"]],
        "mulliken_spin_total": float(populations["mulliken"].sum()),
        "lowdin_spin_total": float(populations["lowdin"].sum()),
        "max_spin_atom": max_index,
        "max_spin_value": float(populations["mulliken"][max_index]) if populations["natom"] else 0.0,
    })

    # The identity that validates the AO->atom mapping: the spin populations
    # must sum to N_alpha - N_beta.  nimrod.spin makes the same check.
    expected = float(n_alpha - n_beta)
    for scheme in ("mulliken", "lowdin"):
        total = properties[f"{scheme}_spin_total"]
        if abs(total - expected) > 1e-6:
            properties.setdefault("population_warnings", []).append(
                f"{scheme} spin populations sum to {total:.6f}, expected "
                f"{expected:.6f}: the AO-to-atom mapping or the density "
                "matrices are not what this script assumes"
            )

    if plan["tddft"]:
        excited = run_tddft(mf, plan, backend)
        properties.update(excited)
        # nimrod.excited stores the reference <S^2> under these names.
        if unrestricted:
            properties["s_squared"] = float(s2)
            properties["s_squared_ideal"] = ideal_s2(plan["multiplicity"])

    properties["gpu_provenance"] = provenance
    result["properties"] = properties
    result["wall_seconds"] = time.time() - started
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate(args)
        plan = make_plan(args)
    except JobError as exc:
        print(f"[nimrod-gpu] ERROR: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.dry_run:
        # Deliberately does not import pyscf, numpy or cupy, so this path runs
        # on a machine with no quantum-chemistry stack and no GPU at all.
        print(format_plan(plan))
        print()
        print("dry run: no calculation performed, nothing imported beyond the standard library")
        return 0

    log(format_plan(plan).replace("\n", "\n[nimrod-gpu] "))

    result = empty_result(plan)
    try:
        result = execute(plan, args)
    except JobError as exc:
        result["error"] = str(exc)
        result["converged"] = False
        log(f"ERROR: {exc}")
        emit(result, args.out)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - the traceback is the deliverable
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["converged"] = False
        traceback.print_exc()
        emit(result, args.out)
        return 1

    emit(result, args.out)
    log(f"done in {result['wall_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
