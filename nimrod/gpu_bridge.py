"""Offload a NiMROD calculation to a rented GPU, and read the answer back.

Psi4 is a CPU code.  There is no build of it, and no option in it, that puts
work on a GPU, so the project's expensive jobs cannot be accelerated in place —
they have to be handed to a different program on different hardware.  This
module is the seam.  It takes a :class:`nimrod.geometry.Structure`, ships it to
a freshly rented Colab VM running :mod:`gpu.gpu4pyscf_job` under GPU4PySCF, and
returns an ordinary :class:`nimrod.psi4_driver.JobResult` — the same object the
Psi4 path produces, with the same ``properties`` keys, so that downstream code
in :mod:`nimrod.spin` and :mod:`nimrod.excited` cannot tell the difference.

What is actually worth sending away
-----------------------------------
Renting a GPU is not free and the round trip costs minutes, so this is not a
general accelerator.  :func:`worth_offloading` encodes the judgement:

* **Worth it.**  The 54-atom ``sensor_assembly()`` at UKS/def2-TZVP (~900 basis
  functions), and the TDDFT on top of it.  These are the jobs that make a
  4-core box useless: the SCF has to converge an open-shell 3d-metal
  determinant at every point of the degradation scan, and the linear-response
  step then calls the same J/K engine tens of times more.
* **Not worth it.**  Anything in the validation tier — H2O, O2, CH2, benzene at
  6-31G or def2-SVP.  These finish locally in seconds, and the VM allocation
  alone costs longer than the calculation.
* **Impossible.**  CASSCF.  Psi4's ``detci`` is CPU-only, and the GPU4PySCF
  stack has no CASSCF at all, so :mod:`nimrod.casscf` has nothing to offload
  to.  :func:`worth_offloading` says so rather than letting a caller discover
  it on a rented VM.

Honest limits of this seam
--------------------------
The two codes are not the same program and this module does not pretend they
are.  GPU4PySCF results come back with ``preset`` naming the backend
(``"gpu4pyscf:diis-df"``) and a full ``properties["gpu_provenance"]`` block
recording the device, the library versions, the resolved ``xc`` string and any
functional-translation warnings.  Two differences are known and material:

* **B3LYP** is defined with VWN3(RPA) correlation in Psi4 and has meant the
  VWN5 variant in some PySCF releases; the job script pins the libxc name to
  remove the ambiguity, and records that it did.
* **wB97X-D** total energies are *not* comparable, because Psi4 adds the -D2
  dispersion term and PySCF does not.  Excitation energies largely survive,
  being a difference of two states that carry the same dispersion term.

The cache is deliberately not shared with Psi4: the fingerprint computed here
includes a backend tag, so a GPU answer can never be served out of
``data/cache`` as though Psi4 had produced it.  See :func:`gpu_fingerprint`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import BASIS_TIERS, DATA_DIR, PROJECT_ROOT
from .geometry import Structure
from .psi4_driver import JobResult, store_cached

__all__ = [
    "GPUOffloadError",
    "ColabNotInstalled",
    "ColabAuthError",
    "ColabQuotaError",
    "ColabResultError",
    "ColabStatus",
    "OffloadAdvice",
    "DEFAULT_LAUNCHER",
    "GPU_DIR",
    "colab_status",
    "gpu_fingerprint",
    "worth_offloading",
    "write_xyz",
    "load_gpu_result",
    "run_gpu_energy",
]

#: The wrapper that owns every Colab-specific detail (accelerator fallback,
#: timeouts, session teardown).  This module never calls the CLI directly.
DEFAULT_LAUNCHER = PROJECT_ROOT / "gpu" / "launch_colab.sh"

#: Where exported geometries and returned JSON documents are kept.
GPU_DIR = DATA_DIR / "gpu"

#: Exit codes defined by ``gpu/launch_colab.sh``.  Anything not listed here is
#: the remote job's own code (3 SCF failure, 4 TDDFT failure, 5 no PySCF).
LAUNCHER_USAGE = 2
LAUNCHER_NOAUTH = 3
LAUNCHER_NOACCEL = 4
LAUNCHER_NOJSON = 5

#: Basis sets large enough that the 54-atom assembly is genuinely painful on
#: four cores.  Used by :func:`worth_offloading`.
_LARGE_BASES = {"def2-tzvp", "def2-tzvpp", "cc-pvtz", "aug-cc-pvdz", "def2-svpd"}

#: Methods with no GPU implementation anywhere in the PySCF stack.
_NO_GPU_METHODS = {
    "casscf", "rasscf", "detci", "fci", "cisd",
    "eom-ccsd", "eom-cc2", "ccsd", "ccsd(t)", "cc2", "cc3",
    "sapt0", "ssapt0", "sapt2",
}


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class GPUOffloadError(RuntimeError):
    """Base class for every failure of the offload path itself.

    Chemistry failures are *not* raised: a job that ran but whose SCF diverged
    comes back as a :class:`~nimrod.psi4_driver.JobResult` with ``converged``
    False and ``error`` set, exactly as the Psi4 driver does.  These exceptions
    mean the calculation never happened.
    """


class ColabNotInstalled(GPUOffloadError):
    """The launcher or the colab CLI is missing from this machine."""


class ColabAuthError(GPUOffloadError):
    """The colab CLI is present but has no usable credentials."""


class ColabQuotaError(GPUOffloadError):
    """No accelerator in the requested list could be allocated."""


class ColabResultError(GPUOffloadError):
    """The remote job returned nothing parseable."""


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColabStatus:
    """Whether an offload could run right now, and if not, why not."""

    launcher_present: bool
    cli_present: bool
    authenticated: bool
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.launcher_present and self.cli_present and self.authenticated

    def explain(self) -> str:
        if self.usable:
            return "GPU offload available"
        return f"GPU offload unavailable: {self.detail}"

    def raise_if_unusable(self) -> None:
        if self.usable:
            return
        if not self.launcher_present or not self.cli_present:
            raise ColabNotInstalled(self.explain())
        raise ColabAuthError(self.explain() + "\n\n" + AUTH_INSTRUCTIONS)


#: The one-time operator action, repeated here so a traceback carries the fix.
AUTH_INSTRUCTIONS = """\
Authenticate the Colab CLI once, on a machine with a browser and the gcloud SDK:

  gcloud auth application-default login \\
    --scopes=openid,\\
https://www.googleapis.com/auth/cloud-platform,\\
https://www.googleapis.com/auth/userinfo.email,\\
https://www.googleapis.com/auth/colaboratory

All four scopes are required:
  userinfo.email    the session backend (colab.research.google.com) 401s without it
  colaboratory      the keep-alive RPC (colab.pa.googleapis.com) 403s without it
  openid            mandated by gcloud
  cloud-platform    mandated by gcloud; it rejects scope lists that omit it

Then confirm with:  colab --auth=adc sessions
See gpu/README.md for the full guide."""

#: Substrings that identify an unauthenticated CLI.  Both strategies fail
#: distinctly: ADC raises DefaultCredentialsError, while the oauth2 default
#: prints a consent URL and then blocks reading a code from stdin.
_UNAUTH_MARKERS = (
    "DefaultCredentialsError",
    "default credentials were not found",
    "Enter the authorization code",
    "To authorize colab-cli",
)


def _colab_command() -> list[str] | None:
    """How to invoke the colab CLI, mirroring the launcher's own resolution."""
    override = os.environ.get("NIMROD_COLAB_CMD")
    if override:
        return override.split()
    micromamba = PROJECT_ROOT / "bin" / "micromamba"
    if micromamba.is_file() and os.access(micromamba, os.X_OK):
        return [
            "env",
            f"MAMBA_ROOT_PREFIX={PROJECT_ROOT / '.mamba'}",
            str(micromamba), "run", "-n", "colabcli", "colab",
        ]
    found = shutil.which("colab")
    return [found] if found else None


def colab_status(*, timeout: float = 120.0) -> ColabStatus:
    """Check the offload path without allocating anything.

    Runs the read-only ``colab sessions`` with stdin closed.  Closing stdin
    matters: with the default ``oauth2`` strategy and a terminal attached the
    CLI blocks forever on ``Enter the authorization code:``, which in an
    unattended scan is indistinguishable from a hung calculation.
    """
    launcher_present = DEFAULT_LAUNCHER.is_file() and os.access(DEFAULT_LAUNCHER, os.X_OK)
    command = _colab_command()
    if command is None:
        return ColabStatus(
            launcher_present, False, False,
            "the colab CLI was not found (set NIMROD_COLAB_CMD to override)",
        )
    if not launcher_present:
        return ColabStatus(
            False, True, False,
            f"launcher missing or not executable: {DEFAULT_LAUNCHER}",
        )

    auth = os.environ.get("NIMROD_COLAB_AUTH", "adc")
    try:
        proc = subprocess.run(
            [*command, f"--auth={auth}", "sessions"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return ColabStatus(launcher_present, False, False, f"cannot execute {command[0]!r}")
    except subprocess.TimeoutExpired:
        return ColabStatus(
            launcher_present, True, False,
            f"`colab sessions` did not return within {timeout:g}s; it is probably "
            "waiting for an interactive OAuth code",
        )

    combined = f"{proc.stdout}\n{proc.stderr}"
    if any(marker in combined for marker in _UNAUTH_MARKERS):
        return ColabStatus(
            launcher_present, True, False,
            f"the colab CLI has no usable credentials (--auth={auth})",
        )
    if proc.returncode != 0:
        return ColabStatus(
            launcher_present, True, False,
            f"`colab sessions` exited {proc.returncode}: "
            f"{combined.strip().splitlines()[-1][:200] if combined.strip() else 'no output'}",
        )
    return ColabStatus(launcher_present, True, True, "authenticated")


# --------------------------------------------------------------------------
# Should this job be offloaded at all?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OffloadAdvice:
    """Verdict on whether a given job belongs on a rented GPU."""

    worth_it: bool
    possible: bool
    reason: str

    def __bool__(self) -> bool:
        return self.worth_it and self.possible


def worth_offloading(
    natom: int,
    basis: str,
    *,
    method: str = "pbe0",
    n_states: int = 0,
) -> OffloadAdvice:
    """Judge whether ``natom`` atoms at ``basis`` justify renting a GPU.

    The thresholds are calibrated on the systems this project actually runs:
    the 37-atom colour centre and the 54-atom assembly are the only structures
    big enough to matter, and only at the production basis.
    """
    method_key = method.lower().removeprefix("td-")
    if method_key in _NO_GPU_METHODS:
        return OffloadAdvice(
            False, False,
            f"{method} has no GPU implementation: Psi4's detci and coupled-cluster "
            "modules are CPU-only and the GPU4PySCF stack does not provide them. "
            "Run it locally.",
        )

    basis_key = basis.lower()
    large_basis = basis_key in _LARGE_BASES

    if natom < 20:
        return OffloadAdvice(
            False, True,
            f"{natom} atoms is a validation-tier system; it finishes locally in "
            "less time than a Colab VM takes to allocate.",
        )
    if natom >= 45 and (large_basis or n_states > 0):
        what = "TDDFT" if n_states else "the SCF"
        return OffloadAdvice(
            True, True,
            f"{natom} atoms at {basis}: {what} on this is the job the offload "
            "path exists for.",
        )
    if large_basis:
        return OffloadAdvice(
            True, True,
            f"{natom} atoms at {basis} is a large single point; offloading is "
            "worthwhile if the local queue is busy.",
        )
    return OffloadAdvice(
        False, True,
        f"{natom} atoms at {basis} is affordable locally; offload only if you "
        f"raise the basis to {BASIS_TIERS['production']} or add excited states.",
    )


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def _reference_for(multiplicity: int, functional: str) -> str:
    if functional.lower() == "hf":
        return "rhf" if multiplicity == 1 else "uhf"
    return "rks" if multiplicity == 1 else "uks"


def gpu_fingerprint(
    structure: Structure,
    *,
    functional: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 0,
    tda: bool = False,
    triplets: str = "none",
    density_fitting: bool = True,
    grid_level: int = 3,
) -> str:
    """Reproduce :func:`gpu.gpu4pyscf_job.fingerprint` without a round trip.

    Needed *before* launching, so that a repeated scan point is served from
    ``data/cache`` instead of renting a VM to recompute it.  The two
    implementations must agree exactly; ``tests`` pins that by comparing this
    value against the one the job script prints in ``--dry-run``.

    The ``__backend__`` entry is what keeps GPU and Psi4 results in separate
    cache slots even when every other field is identical.
    """
    payload = {
        "geometry": " ".join(structure.to_psi4().split()),
        "method": functional.strip().lower(),
        "basis": basis.strip().lower(),
        "charge": charge,
        "multiplicity": multiplicity,
        "reference": _reference_for(multiplicity, functional),
        "options": {
            "df": density_fitting,
            "grid_level": grid_level,
            "n_states": n_states,
            "tda": tda,
            "triplets": triplets,
        },
        "__backend__": "gpu4pyscf",
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# --------------------------------------------------------------------------
# Geometry export and result import
# --------------------------------------------------------------------------


def write_xyz(structure: Structure, path: str | Path, comment: str = "") -> Path:
    """Write ``structure`` as a standard .xyz file for the remote job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(structure.to_xyz(comment or structure.name), encoding="utf-8")
    return path


def load_gpu_result(path: str | Path) -> JobResult:
    """Read a JSON document written by the remote job into a :class:`JobResult`.

    The remote script emits exactly the field set of :class:`JobResult`, with
    everything extra tucked inside ``properties``, so this is a plain
    constructor call.  Any drift between the two files shows up here as a
    :class:`TypeError`, which is the intended alarm.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ColabResultError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ColabResultError(f"{path} does not contain a JSON object")
    try:
        return JobResult(**data)
    except TypeError as exc:
        raise ColabResultError(
            f"{path} does not match the JobResult schema ({exc}); "
            "gpu/gpu4pyscf_job.py and nimrod/psi4_driver.py have drifted apart"
        ) from None


# --------------------------------------------------------------------------
# The offload itself
# --------------------------------------------------------------------------


def run_gpu_energy(
    structure: Structure,
    *,
    functional: str = "pbe0",
    basis: str = "def2-tzvp",
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 0,
    tda: bool = False,
    triplets: str = "none",
    label: str = "",
    gpus: Sequence[str] = ("A100", "T4"),
    timeout_seconds: int = 7200,
    use_cache: bool = True,
    launcher: str | Path | None = None,
    extra_job_args: Sequence[str] = (),
) -> JobResult:
    """Run a UKS/RKS (+ optional TDDFT) job on a rented GPU and return the result.

    Returns the same :class:`~nimrod.psi4_driver.JobResult` shape as
    :func:`nimrod.psi4_driver.run_energy`, cached in the same place under a
    backend-namespaced fingerprint.

    Raises :class:`GPUOffloadError` (or a subclass) when the *offload* fails —
    no CLI, no credentials, no accelerator, no parseable answer.  A job that
    ran and failed chemically is returned, not raised, so a scan can record the
    failure and continue.

    ``timeout_seconds`` is passed to the launcher, which passes it to ``colab
    run``.  The CLI's own default there is 30 seconds, which would kill every
    real calculation in this project; never leave it unset.
    """
    launcher_path = Path(launcher) if launcher is not None else DEFAULT_LAUNCHER

    fingerprint = gpu_fingerprint(
        structure,
        functional=functional,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        n_states=n_states,
        tda=tda,
        triplets=triplets,
    )

    if use_cache:
        cached = _load_by_fingerprint(fingerprint)
        if cached is not None:
            return cached

    status = colab_status()
    status.raise_if_unusable()

    stem = label or structure.name or structure.formula
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stem) or "job"
    GPU_DIR.mkdir(parents=True, exist_ok=True)
    xyz_path = write_xyz(structure, GPU_DIR / f"{stem}-{fingerprint}.xyz", comment=stem)
    out_path = GPU_DIR / f"{stem}-{fingerprint}.json"

    command = [
        str(launcher_path),
        "--xyz", str(xyz_path),
        "--out", str(out_path),
        "--charge", str(charge),
        "--mult", str(multiplicity),
        "--functional", functional,
        "--basis", basis,
        "--n-states", str(n_states),
        "--triplets", triplets,
        "--timeout", str(int(timeout_seconds)),
        "--gpu", " ".join(gpus),
    ]
    if tda:
        command.append("--tda")
    if label:
        command.extend(["--label", label])
    if extra_job_args:
        command.append("--")
        command.extend(extra_job_args)

    try:
        proc = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            # Allow the launcher to enforce its own remote timeout first; only
            # step in if the whole wrapper wedges.
            timeout=timeout_seconds + 1800,
            check=False,
        )
    except FileNotFoundError:
        raise ColabNotInstalled(
            f"launcher not found or not executable: {launcher_path}"
        ) from None
    except subprocess.TimeoutExpired:
        raise ColabResultError(
            f"{launcher_path.name} did not return within "
            f"{timeout_seconds + 1800}s. A VM may still be running: check with "
            "`colab sessions` and stop it, or it will keep burning compute units."
        ) from None

    _raise_for_launcher_status(proc, launcher_path)

    if not out_path.exists():
        raise ColabResultError(
            f"{launcher_path.name} exited {proc.returncode} but wrote no result "
            f"to {out_path}.\n{_tail(proc.stderr)}"
        )

    result = load_gpu_result(out_path)
    # Trust the local fingerprint over the remote one: they are computed by two
    # separate implementations, and the local one is the cache key.
    if result.spec_fingerprint != fingerprint:
        result.properties.setdefault("gpu_provenance", {})
        result.properties["gpu_provenance"]["remote_fingerprint"] = result.spec_fingerprint
        result.properties["gpu_provenance"]["fingerprint_mismatch"] = (
            "the remote job computed a different fingerprint than nimrod.gpu_bridge; "
            "gpu_fingerprint() and gpu4pyscf_job.fingerprint() have drifted apart"
        )
        result.spec_fingerprint = fingerprint

    store_cached(result)
    return result


def _raise_for_launcher_status(proc: subprocess.CompletedProcess[str], launcher: Path) -> None:
    """Turn the launcher's infrastructure exit codes into typed exceptions."""
    if proc.returncode == LAUNCHER_NOAUTH:
        raise ColabAuthError(
            "the Colab CLI is not authenticated, so nothing was run.\n\n"
            + AUTH_INSTRUCTIONS
        )
    if proc.returncode == LAUNCHER_NOACCEL:
        raise ColabQuotaError(
            "no requested accelerator could be allocated. Accelerator "
            "availability is gated by Colab subscription tier and most accounts "
            "get CPU only. Retry with gpus=('T4',) or gpus=('T4', 'CPU') to "
            "accept a CPU runtime.\n" + _tail(proc.stderr)
        )
    if proc.returncode == LAUNCHER_NOJSON:
        raise ColabResultError(
            "the remote job returned no parseable JSON.\n" + _tail(proc.stderr)
        )
    if proc.returncode == LAUNCHER_USAGE:
        raise GPUOffloadError(
            f"{launcher.name} rejected its arguments.\n" + _tail(proc.stderr)
        )
    # Any other non-zero code is the remote job's own verdict (SCF or TDDFT
    # failure).  Those are chemistry, and are carried in the returned
    # JobResult rather than raised.


def _tail(text: str, lines: int = 20) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])


def _load_by_fingerprint(fingerprint: str) -> JobResult | None:
    """Cache lookup by fingerprint alone, without rebuilding a JobSpec."""
    from .psi4_driver import CACHE_DIR

    path = CACHE_DIR / f"{fingerprint}.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return JobResult(**json.load(fh))
    except (json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None
