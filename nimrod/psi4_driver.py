"""A defensive wrapper around the Psi4 driver.

Three problems make raw ``psi4.energy()`` calls unsuitable for an unattended
multi-hundred-job campaign, and this module exists to solve all three:

1. **Option leakage.**  Psi4 options are global and persist between calls.  A
   ``tdscf_states`` left over from a previous job will make the next one fail
   with a confusing irrep-mismatch error.  Every job here runs inside a
   :func:`clean_context` that resets global state first.

2. **Scratch corruption.**  libpsio aborts the *whole process* on some IO
   errors.  We pin ``PSI_SCRATCH`` to a project-local directory and purge it
   between heavy jobs.

3. **SCF convergence.**  Open-shell 3d-metal SCF frequently fails from the
   default guess.  :func:`run_energy` walks the preset ladder in
   :data:`nimrod.config.SCF_PRESETS` until something converges, and records
   *which* preset succeeded so the provenance ends up in the results.

On top of that everything is cached on disk, keyed by a hash of the full job
specification, so an interrupted scan resumes instead of restarting.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import DEFAULT_COMPUTE, SCF_PRESETS, DATA_DIR, applicable_presets

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_INITIALISED = False


def ensure_initialised() -> None:
    """Apply the compute configuration exactly once per process."""
    global _INITIALISED
    if not _INITIALISED:
        DEFAULT_COMPUTE.apply()
        _INITIALISED = True


@contextmanager
def clean_context():
    """Run a Psi4 job with a clean global option stack."""
    import psi4

    ensure_initialised()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    try:
        yield
    finally:
        try:
            psi4.core.clean()
        except Exception:  # pragma: no cover - psi4 internal state
            pass


# --------------------------------------------------------------------------
# Job specification and result
# --------------------------------------------------------------------------


@dataclass
class JobSpec:
    """Everything that uniquely determines a single-point calculation."""

    geometry: str          #: Psi4 geometry block (Cartesian, angstrom)
    method: str            #: e.g. "b3lyp", "eom-ccsd", "casscf"
    basis: str
    charge: int = 0
    multiplicity: int = 1
    reference: str | None = None   #: rhf/uhf/rks/uks/rohf; inferred if None
    options: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def resolved_reference(self) -> str:
        """Pick a sensible reference if the caller did not specify one."""
        if self.reference:
            return self.reference
        is_dft = _is_dft(self.method)
        if self.multiplicity == 1:
            return "rks" if is_dft else "rhf"
        return "uks" if is_dft else "uhf"

    def fingerprint(self) -> str:
        payload = {
            "geometry": " ".join(self.geometry.split()),
            "method": self.method,
            "basis": self.basis,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "reference": self.resolved_reference(),
            "options": {k.lower(): v for k, v in sorted(self.options.items())},
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass
class JobResult:
    """Outcome of a single calculation."""

    spec_fingerprint: str
    label: str
    method: str
    basis: str
    charge: int
    multiplicity: int
    reference: str
    converged: bool
    energy: float | None = None
    preset: str | None = None
    wall_seconds: float = 0.0
    error: str | None = None
    #: Extra scalars harvested from the wavefunction (``<S^2>``, populations,
    #: excitation energies, SAPT components, ...).
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.converged and self.energy is not None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _is_dft(method: str) -> bool:
    """Heuristic: is this method string a density functional?"""
    m = method.lower().removeprefix("td-")
    wavefunction_methods = {
        "hf", "scf", "mp2", "mp3", "ccsd", "ccsd(t)", "cc2", "cc3",
        "eom-ccsd", "eom-cc2", "cisd", "fci", "casscf", "rasscf", "detci",
        "sapt0", "ssapt0", "sapt2", "omp2", "bccd", "cepa(0)", "lccd",
    }
    return m not in wavefunction_methods


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_path(fingerprint: str) -> Path:
    return CACHE_DIR / f"{fingerprint}.json"


def load_cached(spec: JobSpec) -> JobResult | None:
    path = _cache_path(spec.fingerprint())
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return JobResult(**json.load(fh))
    except (json.JSONDecodeError, TypeError, ValueError):
        path.unlink(missing_ok=True)
        return None


def store_cached(result: JobResult) -> None:
    with _cache_path(result.spec_fingerprint).open("w") as fh:
        json.dump(result.to_json(), fh, indent=2, default=str)


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def build_molecule(spec: JobSpec):
    """Construct a Psi4 molecule from a :class:`JobSpec`.

    ``symmetry c1`` and ``no_reorient``/``no_com`` are forced: symmetry
    detection interacts badly with the broken-symmetry and TDSCF machinery, and
    reorientation would invalidate externally-supplied fragment geometries.
    """
    import psi4

    body = spec.geometry.strip()
    header = f"{spec.charge} {spec.multiplicity}"
    # A multi-fragment geometry (SAPT) carries its own per-fragment charge
    # lines and a "--" separator; do not prepend a global charge line.
    if "--" in body:
        block = f"{body}\nsymmetry c1\nno_reorient\nno_com\n"
    else:
        block = f"{header}\n{body}\nsymmetry c1\nno_reorient\nno_com\n"
    return psi4.geometry(block)


def run_energy(
    spec: JobSpec,
    *,
    use_cache: bool = True,
    property_hook: Callable[[Any, Any], dict[str, Any]] | None = None,
    presets: Sequence[dict[str, Any]] | None = None,
) -> JobResult:
    """Run a single-point energy, walking the SCF preset ladder on failure.

    Parameters
    ----------
    property_hook
        Called as ``hook(wfn, molecule)`` after a successful calculation; the
        returned mapping is merged into :attr:`JobResult.properties`.  This is
        how the spin, excited-state and SAPT modules harvest their observables
        without re-running the SCF.
    """
    import psi4

    fingerprint = spec.fingerprint()
    if use_cache:
        cached = load_cached(spec)
        if cached is not None:
            return cached

    reference = spec.resolved_reference()
    ladder = presets if presets is not None else applicable_presets(spec.method)
    last_error: str | None = None
    started = time.time()

    for preset in ladder:
        with clean_context():
            try:
                mol = build_molecule(spec)
                opts: dict[str, Any] = {
                    "basis": spec.basis,
                    "reference": reference,
                    **preset["options"],
                }
                opts.update(spec.options)  # caller options win
                psi4.set_options(opts)

                energy, wfn = psi4.energy(spec.method, return_wfn=True, molecule=mol)

                props: dict[str, Any] = {}
                if property_hook is not None:
                    try:
                        props = property_hook(wfn, mol) or {}
                    except Exception as exc:  # property failure must not lose the energy
                        props = {"property_error": f"{type(exc).__name__}: {exc}"}

                result = JobResult(
                    spec_fingerprint=fingerprint,
                    label=spec.label,
                    method=spec.method,
                    basis=spec.basis,
                    charge=spec.charge,
                    multiplicity=spec.multiplicity,
                    reference=reference,
                    converged=True,
                    energy=float(energy),
                    preset=preset["label"],
                    wall_seconds=time.time() - started,
                    properties=props,
                )
                store_cached(result)
                return result

            except Exception as exc:
                last_error = f"[{preset['label']}] {type(exc).__name__}: {str(exc)[:400]}"
                continue

    result = JobResult(
        spec_fingerprint=fingerprint,
        label=spec.label,
        method=spec.method,
        basis=spec.basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        reference=reference,
        converged=False,
        wall_seconds=time.time() - started,
        error=last_error,
    )
    store_cached(result)
    return result


#: Independent SCF starting guesses.  SAD is Psi4's default and is usually
#: best, but it is a superposition of *atomic* densities and in a stretched,
#: near-degenerate complex it can bias the SCF straight into an excited
#: closed-shell solution and converge there happily.
GUESS_LADDER = ("sad", "gwh")


def run_energy_multiguess(
    spec: JobSpec,
    *,
    guesses: Sequence[str] = GUESS_LADDER,
    property_hook: Callable[[Any, Any], dict[str, Any]] | None = None,
    use_cache: bool = True,
) -> JobResult:
    """Run a single point from several starting guesses and keep the lowest.

    A converged SCF is not necessarily the *ground-state* SCF.  Along the
    ligand-loss coordinate the default SAD guess drove the closed-shell
    wB97X-D solution to a state 0.13 Eh — about 85 kcal/mol — above the true
    minimum, and did it without any convergence failure: the energies looked
    fine until plotted against the coordinate, where they jumped by 105
    kcal/mol between adjacent points while the triplet stayed smooth.  A
    single-determinant excited solution is exactly what a variational method
    cannot warn you about.

    Trying independent guesses and taking the minimum is the cheap defence.
    The winning guess is recorded in ``properties['winning_guess']`` and the
    spread over guesses in ``properties['guess_spread_kcal']``, so a large
    spread flags a point whose SCF solution is not robust.
    """
    from .config import HARTREE_TO_KCAL

    best: JobResult | None = None
    energies: dict[str, float] = {}

    for guess in guesses:
        trial = JobSpec(
            geometry=spec.geometry,
            method=spec.method,
            basis=spec.basis,
            charge=spec.charge,
            multiplicity=spec.multiplicity,
            reference=spec.reference,
            options={**spec.options, "guess": guess},
            label=f"{spec.label}-{guess}" if spec.label else guess,
        )
        result = run_energy(trial, use_cache=use_cache, property_hook=property_hook)
        if not result.ok or result.energy is None:
            continue
        energies[guess] = result.energy
        if best is None or result.energy < best.energy:  # type: ignore[operator]
            best = result

    if best is None:
        # Every guess failed; return the plain attempt so the caller sees why.
        return run_energy(spec, use_cache=use_cache, property_hook=property_hook)

    best.properties = dict(best.properties)
    best.properties["winning_guess"] = min(energies, key=energies.__getitem__)
    if len(energies) > 1:
        spread = (max(energies.values()) - min(energies.values())) * HARTREE_TO_KCAL
        best.properties["guess_spread_kcal"] = spread
    return best


def run_optimize(
    spec: JobSpec,
    *,
    use_cache: bool = True,
    max_iter: int = 60,
) -> tuple[JobResult, str | None]:
    """Geometry-optimise and return ``(result, optimised_geometry_xyz)``.

    The optimised geometry is stored in ``result.properties['geometry']`` so it
    survives caching.
    """
    import psi4

    opt_spec = JobSpec(
        geometry=spec.geometry,
        method=spec.method,
        basis=spec.basis,
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        reference=spec.reference,
        options={**spec.options, "__opt__": True, "geom_maxiter": max_iter},
        label=spec.label or "optimise",
    )

    if use_cache:
        cached = load_cached(opt_spec)
        if cached is not None:
            return cached, cached.properties.get("geometry")

    reference = opt_spec.resolved_reference()
    started = time.time()
    last_error = None

    # Walk the full ladder here too.  Truncating it to the first two rungs
    # saved time on easy cases and silently lost the hard ones: the closed-shell
    # meta-GGA optimisations of O2 and NH only converge on the heavily damped
    # rung, and stopping early reported them as failures.
    for preset in applicable_presets(opt_spec.method):
        with clean_context():
            try:
                mol = build_molecule(opt_spec)
                opts = {
                    "basis": opt_spec.basis,
                    "reference": reference,
                    "geom_maxiter": max_iter,
                    "g_convergence": "gau_loose",
                    **preset["options"],
                }
                opts.update({k: v for k, v in spec.options.items() if not k.startswith("__")})
                psi4.set_options(opts)

                energy = psi4.optimize(opt_spec.method, molecule=mol)
                geom = _molecule_to_xyz(mol)

                result = JobResult(
                    spec_fingerprint=opt_spec.fingerprint(),
                    label=opt_spec.label,
                    method=opt_spec.method,
                    basis=opt_spec.basis,
                    charge=opt_spec.charge,
                    multiplicity=opt_spec.multiplicity,
                    reference=reference,
                    converged=True,
                    energy=float(energy),
                    preset=preset["label"],
                    wall_seconds=time.time() - started,
                    properties={"geometry": geom},
                )
                store_cached(result)
                return result, geom

            except Exception as exc:
                last_error = f"[{preset['label']}] {type(exc).__name__}: {str(exc)[:400]}"
                continue

    result = JobResult(
        spec_fingerprint=opt_spec.fingerprint(),
        label=opt_spec.label,
        method=opt_spec.method,
        basis=opt_spec.basis,
        charge=opt_spec.charge,
        multiplicity=opt_spec.multiplicity,
        reference=reference,
        converged=False,
        wall_seconds=time.time() - started,
        error=last_error,
    )
    store_cached(result)
    return result, None


def _molecule_to_xyz(mol) -> str:
    """Serialise a Psi4 molecule to a bare Cartesian block in angstrom."""
    from .config import BOHR_TO_ANGSTROM

    lines = []
    geom = mol.geometry().to_array()
    for i in range(mol.natom()):
        x, y, z = (float(c) * BOHR_TO_ANGSTROM for c in geom[i])
        lines.append(f"{mol.symbol(i):<3s} {x:14.8f} {y:14.8f} {z:14.8f}")
    return "\n".join(lines)


def molecule_to_xyz(mol) -> str:
    """Public alias for :func:`_molecule_to_xyz`."""
    return _molecule_to_xyz(mol)
