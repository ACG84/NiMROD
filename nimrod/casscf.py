"""CASSCF spin manifolds and the multireference audit of the DFT results.

Everything else in NiMROD is single-reference: Kohn-Sham DFT for the spin-state
energetics, TDDFT for the colour-centre transition, broken-symmetry DFT for the
exchange coupling.  Each of those rests on the assumption that one Slater
determinant is a qualitatively correct zeroth order description.  That
assumption is exactly the one that fails in the interesting part of this
project.

Two places where it is genuinely at risk:

*Ni(II) along the degradation coordinate.*  A square-planar d8 centre is a
clean closed shell, and a three-coordinate d8 centre is a clean triplet, but in
between the ligand field collapses through a region where the two states are
near-degenerate and the singlet acquires substantial two-determinant character
(the classic (d_{x2-y2})^2 / (d_{x2-y2})^1(d_{z2})^1 mixing).  A restricted
Kohn-Sham singlet cannot represent that at all, and an unrestricted one fakes
it by contaminating itself with the triplet.

*The colour centre.*  A doublet aryl radical on a PAH is usually well behaved,
but the low-lying excitations of a defect state can carry doubly-excited
character that TDDFT is structurally incapable of describing.

CASSCF is the cheapest honest answer to both worries.  It is not a quantitative
method here -- it has no dynamic correlation, so the absolute spin gaps it
produces are systematically biased -- but it does two things nothing else in
the project can:

1. It gives a *variationally* correct multideterminantal wavefunction for a
   spin state, so the singlet-triplet gap it produces is a genuine cross-check
   on the DFT gaps rather than another instance of the same approximation.
2. It reports how multireference the problem actually is, which is what decides
   whether the DFT numbers are defensible at all.

Diagnostics implemented here
----------------------------
Both of the diagnostics named in the project brief are implemented, because
Psi4 1.11 makes each of them available by a different route:

*Leading CI coefficient / reference weight.*  Psi4 does **not** expose the
converged CI vector to Python in any usable way -- ``CIWavefunction.D_vector()``
and ``CIWavefunction.no_occupations()`` both terminate the process (SIGKILL and
SIGSEGV respectively in 1.11), and ``CIWavefunction.ndet()`` returns an
uninitialised pointer value rather than a determinant count.  What *is*
reliable is detci's own printed table of leading determinants.  This module
therefore redirects the Psi4 output to a per-job file, runs the calculation,
and parses that table back out.  ``reference_weight`` is the square of the
largest coefficient; the rule of thumb is that below ~0.9 a single-reference
treatment is no longer defensible.

*Natural orbital occupations.*  Computed numerically, with no text parsing,
from the CI one-particle density matrix via ``CIWavefunction.get_opdm``.  For a
CASSCF wavefunction ``get_opdm(0, 0, "SUM", False)`` returns the active-space
block; for CISD it returns the full MO-space matrix.  Diagonalising it gives
the natural occupations, and the deviation from the integer 2/0 pattern is the
static-correlation measure.  We report the Head-Gordon number of effectively
unpaired electrons

    N_U = sum_i min(n_i, 2 - n_i)

and, crucially for an open-shell system, its *excess* over the number of
electrons the spin state is required to have unpaired (``multiplicity - 1``).
Without that subtraction every triplet looks maximally multireference, which is
nonsense: a triplet is *supposed* to have two singly-occupied natural orbitals.

Conventions
-----------
* Symmetry is always ``c1``, so ``RESTRICTED_DOCC`` and ``ACTIVE`` are always
  single-element lists.
* detci requires an RHF or ROHF reference.  A UKS/UHF reference is silently
  meaningless here, so :func:`casscf_reference` picks ``rhf`` for a closed
  shell and ``rohf`` otherwise, and this overrides
  :meth:`~nimrod.psi4_driver.JobSpec.resolved_reference`, which would hand back
  ``uhf``.
* Spin gaps follow the same sign convention as :mod:`nimrod.degradation`:
  ``gap = E(high multiplicity) - E(low multiplicity)``, so a positive gap means
  the low-spin state is the ground state.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .config import (
    BASIS_TIERS,
    DEFAULT_COMPUTE,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL,
    OUTPUT_DIR,
)
from .geometry import ATOMIC_NUMBER, Structure
from .psi4_driver import (
    JobResult,
    JobSpec,
    ensure_initialised,
    load_cached,
    run_energy,
)

__all__ = [
    "CASSCF_BASIS",
    "CASSCF_PRESETS",
    "REFERENCE_WEIGHT_THRESHOLD",
    "EXCESS_UNPAIRED_THRESHOLD",
    "cas_space",
    "count_electrons",
    "casscf_reference",
    "Determinant",
    "parse_leading_determinants",
    "natural_occupations",
    "unpaired_metrics",
    "casscf_energy",
    "CASSCFSpinGap",
    "casscf_spin_gap",
    "MultireferenceDiagnostic",
    "multireference_diagnostic",
]


#: Default basis for CASSCF.  Correlated wavefunction methods need a basis with
#: proper polarisation; cc-pVDZ is the cheapest one that is not embarrassing.
CASSCF_BASIS = BASIS_TIERS["correlated"]

#: ``C0**2`` below this and the single-determinant picture is not defensible.
#: 0.90 is the conventional cut; 0.95 is the conservative one.
REFERENCE_WEIGHT_THRESHOLD = 0.90

#: Effectively-unpaired electrons *in excess of* those the spin state demands.
#: Anything above ~0.5 means real static correlation the DFT cannot see.
EXCESS_UNPAIRED_THRESHOLD = 0.50


# --------------------------------------------------------------------------
# Convergence presets
# --------------------------------------------------------------------------
#
# The MCSCF orbital rotation is a much harder optimisation than an SCF: the
# active-space energy is not variationally bounded with respect to which
# orbitals end up in the active space, so a two-step (TS) solver that has
# wandered into the wrong orbital set will happily converge onto a saddle.  The
# ladder below goes TS -> TS with a later DIIS start -> augmented Hessian ->
# augmented Hessian from a core guess, which is the usual escalation.
#
# ``scf_type pk`` and ``mcscf_type conv`` are pinned together: mixing a
# density-fitted SCF with conventional MCSCF integrals produces a reference
# energy that does not match the CASSCF reference energy, which makes the
# printed correlation energy meaningless even though the total energy is fine.

_CAS_COMMON = {
    "scf_type": "pk",
    "mcscf_type": "conv",
    "e_convergence": 1e-9,
    "d_convergence": 1e-8,
    "maxiter": 200,
}

CASSCF_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "label": "mcscf-ts",
        "options": {
            **_CAS_COMMON,
            "guess": "sad",
            "mcscf_algorithm": "ts",
            "mcscf_maxiter": 100,
            "mcscf_e_convergence": 1e-9,
            "mcscf_r_convergence": 1e-6,
        },
    },
    {
        "label": "mcscf-ts-late-diis",
        "options": {
            **_CAS_COMMON,
            "guess": "sad",
            "mcscf_algorithm": "ts",
            "mcscf_diis_start": 10,
            "mcscf_so_start_e": 1e-3,
            "mcscf_maxiter": 200,
            "mcscf_e_convergence": 1e-9,
            "mcscf_r_convergence": 1e-6,
        },
    },
    {
        "label": "mcscf-ah",
        "options": {
            **_CAS_COMMON,
            "guess": "sad",
            "mcscf_algorithm": "ah",
            "mcscf_maxiter": 300,
            "mcscf_e_convergence": 1e-8,
            "mcscf_r_convergence": 1e-5,
        },
    },
    {
        "label": "mcscf-ah-core",
        "options": {
            **_CAS_COMMON,
            "guess": "core",
            "mcscf_algorithm": "ah",
            "mcscf_maxiter": 400,
            "mcscf_e_convergence": 1e-8,
            "mcscf_r_convergence": 1e-5,
        },
    },
)

#: CISD is a plain CI on top of a converged SCF, so it needs no MCSCF ladder --
#: only the ``detci`` module selection (see :func:`_cisd_spec`).
CISD_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "label": "cisd-pk",
        "options": {
            "scf_type": "pk",
            "guess": "sad",
            "e_convergence": 1e-9,
            "d_convergence": 1e-8,
            "maxiter": 200,
        },
    },
    {
        "label": "cisd-pk-soscf",
        "options": {
            "scf_type": "pk",
            "guess": "sad",
            "soscf": True,
            "e_convergence": 1e-9,
            "d_convergence": 1e-8,
            "maxiter": 300,
        },
    },
)


# --------------------------------------------------------------------------
# Active space bookkeeping
# --------------------------------------------------------------------------


def count_electrons(geometry: str, charge: int = 0) -> int:
    """Total electron count of a Cartesian geometry block.

    Parsed from the element symbols, so it needs no Psi4 process; unknown
    elements raise rather than being silently skipped.
    """
    structure = Structure.from_psi4(geometry)
    if len(structure) == 0:
        raise ValueError("geometry block contains no atoms")
    unknown = sorted({s for s in structure.symbols if s not in ATOMIC_NUMBER})
    if unknown:
        raise KeyError(
            f"no atomic number for element(s) {unknown}; extend "
            "nimrod.geometry.ATOMIC_NUMBER"
        )
    return structure.n_electrons - charge


def cas_space(
    n_total_electrons: int,
    n_active_electrons: int,
    n_active_orbitals: int,
) -> dict[str, list[int]]:
    """Psi4 orbital-space options for a CAS(``n_active_electrons``, ``n_active_orbitals``).

    Returns ``{"RESTRICTED_DOCC": [n], "ACTIVE": [m]}``.  Both are single-element
    lists because every calculation in this project runs in ``c1``, where there
    is exactly one irrep.

    The doubly-occupied restricted space holds every electron not promoted into
    the active space::

        restricted_docc = (n_total_electrons - n_active_electrons) // 2

    which only makes sense if those two counts have the same parity -- all
    unpaired electrons must live in the active space, by construction of the
    CAS ansatz.
    """
    for name, value in (
        ("n_total_electrons", n_total_electrons),
        ("n_active_electrons", n_active_electrons),
        ("n_active_orbitals", n_active_orbitals),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {value!r}")

    n_total_electrons = int(n_total_electrons)
    n_active_electrons = int(n_active_electrons)
    n_active_orbitals = int(n_active_orbitals)

    if n_total_electrons < 0:
        raise ValueError(f"n_total_electrons must be non-negative, got {n_total_electrons}")
    if n_active_electrons < 0:
        raise ValueError(f"n_active_electrons must be non-negative, got {n_active_electrons}")
    if n_active_orbitals < 1:
        raise ValueError(f"n_active_orbitals must be at least 1, got {n_active_orbitals}")

    if n_active_electrons > n_total_electrons:
        raise ValueError(
            f"active space asks for {n_active_electrons} electrons but the "
            f"molecule only has {n_total_electrons}"
        )
    if n_active_electrons > 2 * n_active_orbitals:
        raise ValueError(
            f"CAS({n_active_electrons},{n_active_orbitals}) is overfull: "
            f"{n_active_orbitals} orbitals hold at most {2 * n_active_orbitals} "
            "electrons"
        )

    inactive = n_total_electrons - n_active_electrons
    if inactive % 2 != 0:
        raise ValueError(
            f"parity mismatch: {n_total_electrons} total electrons and "
            f"{n_active_electrons} active electrons leave {inactive} inactive "
            "electrons, which cannot fill doubly-occupied orbitals.  The active "
            "space must absorb every unpaired electron, so n_active_electrons "
            "must have the same parity as n_total_electrons."
        )

    restricted_docc = inactive // 2
    if restricted_docc < 0:  # unreachable given the checks above; kept explicit
        raise ValueError(f"restricted_docc would be negative ({restricted_docc})")

    return {"RESTRICTED_DOCC": [restricted_docc], "ACTIVE": [n_active_orbitals]}


def casscf_reference(multiplicity: int) -> str:
    """The only references detci accepts: ``rhf`` closed shell, ``rohf`` open.

    Psi4's detci cannot start from an unrestricted determinant, and
    :meth:`nimrod.psi4_driver.JobSpec.resolved_reference` would hand back
    ``uhf`` for any multiplicity above one, so every CASSCF spec in this module
    sets ``reference`` explicitly.
    """
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    return "rhf" if multiplicity == 1 else "rohf"


def _check_spin_compatibility(n_active_electrons: int, n_active_orbitals: int, multiplicity: int) -> None:
    """The active space must be able to carry the requested spin."""
    unpaired = multiplicity - 1
    if unpaired < 0:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    if unpaired > n_active_electrons:
        raise ValueError(
            f"multiplicity {multiplicity} needs {unpaired} unpaired electrons "
            f"but the active space only holds {n_active_electrons}"
        )
    if (n_active_electrons - unpaired) % 2 != 0:
        raise ValueError(
            f"CAS({n_active_electrons},{n_active_orbitals}) cannot form "
            f"multiplicity {multiplicity}: {n_active_electrons} active "
            f"electrons and {unpaired} unpaired electrons have opposite parity"
        )
    if unpaired > n_active_orbitals:
        raise ValueError(
            f"multiplicity {multiplicity} needs {unpaired} singly-occupied "
            f"orbitals but the active space has only {n_active_orbitals}"
        )


def _space_label(n_active_electrons: int, n_active_orbitals: int) -> str:
    return f"({n_active_electrons},{n_active_orbitals})"


# --------------------------------------------------------------------------
# Leading determinants, scraped from the detci output
# --------------------------------------------------------------------------

_DET_HEADER = "most important determinants"

# e.g.  "    *   1   -0.963838  (    0,    0)  5AX 6AX 7AX"
_DET_LINE = re.compile(
    r"^\s*\*?\s*(?P<rank>\d+)\s+"
    r"(?P<coef>[-+]?\d+\.\d+)\s+"
    r"\(\s*(?P<alpha>\d+)\s*,\s*(?P<beta>\d+)\s*\)\s*"
    r"(?P<occ>.*?)\s*$"
)


@dataclass(frozen=True)
class Determinant:
    """One entry of detci's leading-determinant table."""

    rank: int
    coefficient: float
    alpha_string: int
    beta_string: int
    occupation: str

    @property
    def weight(self) -> float:
        """``|C|**2`` -- this determinant's share of the normalised CI vector."""
        return self.coefficient * self.coefficient

    def as_tuple(self) -> tuple[int, float, int, int, str]:
        """JSON-friendly form, so it survives :class:`JobResult` caching."""
        return (self.rank, self.coefficient, self.alpha_string, self.beta_string, self.occupation)


def parse_leading_determinants(text: str) -> list[Determinant]:
    """Extract the last ``The N most important determinants:`` table from Psi4 output.

    The *last* table is the right one: an MCSCF run prints the table once per
    CI diagonalisation it decides to report, and a failed preset that was
    retried leaves its own tables earlier in the same file.

    Returns an empty list if no table is present -- for example when the run
    used a module other than detci, or when ``NUM_DETS_PRINT`` was zero.
    """
    start = text.rfind(_DET_HEADER)
    if start < 0:
        return []

    determinants: list[Determinant] = []
    blanks = 0
    for line in text[start:].splitlines()[1:]:
        if not line.strip():
            if determinants:
                blanks += 1
                if blanks >= 2:
                    break
            continue
        match = _DET_LINE.match(line)
        if match is None:
            break
        blanks = 0
        determinants.append(
            Determinant(
                rank=int(match.group("rank")),
                coefficient=float(match.group("coef")),
                alpha_string=int(match.group("alpha")),
                beta_string=int(match.group("beta")),
                occupation=match.group("occ").strip(),
            )
        )

    determinants.sort(key=lambda d: abs(d.coefficient), reverse=True)
    return determinants


@contextmanager
def _captured_output(path: Path) -> Iterator[Path]:
    """Route Psi4's output to ``path`` for the duration of the block.

    :meth:`nimrod.config.ComputeConfig.apply` calls ``psi4.core.be_quiet()``,
    which sends the output to ``/dev/null``; without this override there is
    nothing to parse.  The previous quiet state is restored on the way out so
    the rest of the campaign stays silent.
    """
    import psi4

    ensure_initialised()
    path.parent.mkdir(parents=True, exist_ok=True)
    psi4.core.set_output_file(str(path), False)
    try:
        yield path
    finally:
        try:
            psi4.core.flush_outfile()
        except Exception:  # pragma: no cover - psi4 internal state
            pass
        psi4.core.set_output_file(str(OUTPUT_DIR / "psi4.log"), True)
        if DEFAULT_COMPUTE.quiet:
            psi4.core.be_quiet()


# --------------------------------------------------------------------------
# Natural occupations
# --------------------------------------------------------------------------


def natural_occupations(wfn) -> tuple[np.ndarray | None, str]:
    """Natural orbital occupations of a CI wavefunction, largest first.

    Returns ``(occupations, space)`` where ``space`` is ``"active"`` when the
    density Psi4 handed back covers only the CAS active orbitals, ``"full-mo"``
    when it spans the whole MO space, and ``"unavailable"`` when neither call
    worked.

    ``CIWavefunction.no_occupations()`` would be the obvious API for this and
    segfaults in Psi4 1.11, so the occupations are obtained by diagonalising the
    one-particle density matrix instead.  ``get_opdm(0, 0, "SUM", False)`` is
    tried first because for CASSCF it returns exactly the active block, which is
    the physically meaningful set; for CISD the same call returns the full MO
    matrix, which is also what we want there.
    """
    if not hasattr(wfn, "get_opdm"):
        return None, "unavailable"

    try:
        nmo = int(wfn.nmo())
    except Exception:  # pragma: no cover - psi4 internal state
        nmo = -1

    for args in ((0, 0, "SUM", False), (-1, -1, "SUM", True)):
        try:
            matrix = np.asarray(wfn.get_opdm(*args))
        except Exception:
            continue
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            continue
        # The OPDM is symmetric by construction; symmetrise against round-off
        # so eigvalsh cannot return complex noise.
        occupations = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[::-1]
        space = "active" if 0 < matrix.shape[0] < nmo else "full-mo"
        return occupations, space

    return None, "unavailable"


def unpaired_metrics(occupations: Sequence[float], multiplicity: int) -> dict[str, float]:
    """Static-correlation measures from a set of natural occupations.

    ``max_fractional_occupation``
        ``max_i min(n_i, 2 - n_i)``.  Zero for a single determinant, one for a
        perfect diradical.  This is the "deviation from 2/0" figure.

    ``n_effectively_unpaired``
        Head-Gordon's linear index ``N_U = sum_i min(n_i, 2 - n_i)``.

    ``n_effectively_unpaired_nl``
        Head-Gordon's nonlinear index ``sum_i n_i^2 (2 - n_i)^2``, which
        suppresses the long tail of weakly correlated orbitals and so is the
        less alarmist of the two.

    ``excess_unpaired``
        ``N_U - (multiplicity - 1)``.  A triplet is *required* to have two
        singly-occupied natural orbitals, so the raw index says nothing about
        multireference character until that mandatory part is subtracted.  This
        is the number the single-reference verdict is based on.
    """
    occ = np.clip(np.asarray(list(occupations), dtype=float), 0.0, 2.0)
    if occ.size == 0:
        raise ValueError("no natural occupations supplied")
    fractional = np.minimum(occ, 2.0 - occ)
    n_u = float(fractional.sum())
    return {
        "max_fractional_occupation": float(fractional.max()),
        "n_effectively_unpaired": n_u,
        "n_effectively_unpaired_nl": float(np.sum(occ**2 * (2.0 - occ) ** 2)),
        "excess_unpaired": n_u - float(multiplicity - 1),
    }


# --------------------------------------------------------------------------
# The property hook
# --------------------------------------------------------------------------

_PSIVARS = (
    ("CI ROOT 0 TOTAL ENERGY", "ci_total_energy"),
    ("CI CORRELATION ENERGY", "ci_correlation_energy"),
    ("MCSCF TOTAL ENERGY", "mcscf_total_energy"),
    ("CURRENT REFERENCE ENERGY", "reference_energy"),
)


def _make_property_hook(output_path: Path | None, multiplicity: int):
    """Build the ``property_hook`` handed to :func:`~nimrod.psi4_driver.run_energy`.

    Everything harvested here lands in ``JobResult.properties`` and is therefore
    cached with the energy, which matters: the leading-determinant table can
    only be recovered from the Psi4 output of the run that produced it, so if it
    were not cached a cache hit would silently lose the diagnostic.
    """

    def hook(wfn, molecule) -> dict[str, Any]:
        import psi4

        props: dict[str, Any] = {}

        occupations, space = natural_occupations(wfn)
        props["occupation_space"] = space
        if occupations is not None:
            props["natural_occupations"] = [float(x) for x in occupations]
            props.update(unpaired_metrics(occupations, multiplicity))
        else:
            props["occupation_error"] = (
                "CIWavefunction.get_opdm() returned nothing usable; natural "
                "occupations unavailable"
            )

        for variable, key in _PSIVARS:
            try:
                props[key] = float(psi4.variable(variable))
            except Exception:
                pass

        if output_path is not None:
            try:
                psi4.core.flush_outfile()
                text = output_path.read_text(errors="replace")
            except OSError as exc:
                props["ci_vector_error"] = f"could not read Psi4 output: {exc}"
            else:
                determinants = parse_leading_determinants(text)
                if determinants:
                    props["leading_determinants"] = [d.as_tuple() for d in determinants]
                    props["largest_ci_coefficient"] = determinants[0].coefficient
                    props["reference_weight"] = determinants[0].weight
                else:
                    props["ci_vector_error"] = (
                        "no leading-determinant table in the Psi4 output "
                        "(wrong module, or NUM_DETS_PRINT = 0)"
                    )

        return props

    return hook


# --------------------------------------------------------------------------
# CASSCF single points
# --------------------------------------------------------------------------


def casscf_energy(
    geometry: str,
    charge: int,
    multiplicity: int,
    n_active_electrons: int,
    n_active_orbitals: int,
    basis: str = CASSCF_BASIS,
    *,
    n_total_electrons: int | None = None,
    label: str = "",
    options: dict[str, Any] | None = None,
    num_dets_print: int = 12,
    capture_ci_vector: bool = True,
    use_cache: bool = True,
    presets: Sequence[dict[str, Any]] | None = None,
) -> JobResult:
    """A CASSCF single point, run through the standard driver.

    Parameters
    ----------
    geometry
        Bare Cartesian block in angstrom (no charge/multiplicity line); the
        driver adds the header and forces ``c1``.
    charge, multiplicity
        Deliberately required rather than defaulted: which spin state is being
        computed is the entire content of a CASSCF spin-gap study, and a
        silently-defaulted singlet would be a very quiet way to get a wrong
        answer.
    n_total_electrons
        Counted from ``geometry`` and ``charge`` when omitted.  Pass it
        explicitly only for geometries this module cannot parse (ghost atoms,
        multi-fragment blocks).
    capture_ci_vector
        Redirect Psi4's output to ``.psi4_output/casscf_<fingerprint>.out`` for
        this job and parse the leading-determinant table out of it.  This is the
        only way to get CI coefficients out of Psi4 1.11 (see the module
        docstring).  Turn it off to keep the campaign log tidy when the
        diagnostic is not wanted.

    The result carries the natural occupations, the static-correlation indices
    and, when captured, the leading determinants in
    :attr:`~nimrod.psi4_driver.JobResult.properties`.
    """
    if n_total_electrons is None:
        n_total_electrons = count_electrons(geometry, charge)

    _check_spin_compatibility(n_active_electrons, n_active_orbitals, multiplicity)
    spaces = cas_space(n_total_electrons, n_active_electrons, n_active_orbitals)

    job_options: dict[str, Any] = dict(spaces)
    job_options["num_dets_print"] = int(num_dets_print) if capture_ci_vector else 0
    job_options["opdm"] = True
    if options:
        job_options.update(options)

    spec = JobSpec(
        geometry=geometry,
        method="casscf",
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        reference=casscf_reference(multiplicity),
        options=job_options,
        label=label or f"casscf{_space_label(n_active_electrons, n_active_orbitals)}",
    )

    return _run_ci(
        spec,
        multiplicity=multiplicity,
        presets=presets if presets is not None else CASSCF_PRESETS,
        capture=capture_ci_vector,
        use_cache=use_cache,
    )


def _cisd_spec(
    geometry: str,
    charge: int,
    multiplicity: int,
    basis: str,
    label: str,
    options: dict[str, Any] | None,
    num_dets_print: int,
    capture: bool,
) -> JobSpec:
    """A CISD spec pinned to the ``detci`` module.

    Psi4's default CISD procedure for an RHF reference goes through ``fnocc``,
    which is faster but returns a plain :class:`psi4.core.Wavefunction`: no
    ``get_opdm``, no printed determinant table, hence no diagnostic at all.
    Forcing ``qc_module detci`` costs time and buys both.
    """
    job_options: dict[str, Any] = {
        "qc_module": "detci",
        "opdm": True,
        "num_dets_print": int(num_dets_print) if capture else 0,
    }
    if options:
        job_options.update(options)

    return JobSpec(
        geometry=geometry,
        method="cisd",
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        reference=casscf_reference(multiplicity),
        options=job_options,
        label=label or "cisd",
    )


def _run_ci(
    spec: JobSpec,
    *,
    multiplicity: int,
    presets: Sequence[dict[str, Any]],
    capture: bool,
    use_cache: bool,
) -> JobResult:
    """Run a detci job, optionally capturing its output for CI-vector parsing."""
    if use_cache:
        cached = load_cached(spec)
        if cached is not None:
            return cached

    if not capture:
        return run_energy(
            spec,
            use_cache=use_cache,
            property_hook=_make_property_hook(None, multiplicity),
            presets=presets,
        )

    output_path = OUTPUT_DIR / f"{spec.method}_{spec.fingerprint()}.out"
    with _captured_output(output_path):
        return run_energy(
            spec,
            use_cache=use_cache,
            property_hook=_make_property_hook(output_path, multiplicity),
            presets=presets,
        )


# --------------------------------------------------------------------------
# Spin gaps
# --------------------------------------------------------------------------


@dataclass
class CASSCFSpinGap:
    """A CASSCF spin-state splitting, as a cross-check on the DFT gaps.

    Sign convention matches :mod:`nimrod.degradation`::

        gap = E(high multiplicity) - E(low multiplicity)

    so a *positive* gap means the low-spin state is the ground state, and the
    gap changing sign along the degradation coordinate is the spin crossover
    the sensor is built to report.
    """

    label: str
    active_space: str
    basis: str
    charge: int
    low_multiplicity: int
    high_multiplicity: int
    low: JobResult
    high: JobResult

    @property
    def ok(self) -> bool:
        return self.low.ok and self.high.ok

    @property
    def low_energy(self) -> float | None:
        return self.low.energy if self.low.ok else None

    @property
    def high_energy(self) -> float | None:
        return self.high.energy if self.high.ok else None

    @property
    def gap_hartree(self) -> float | None:
        if not self.ok:
            return None
        return float(self.high.energy) - float(self.low.energy)  # type: ignore[arg-type]

    @property
    def gap_ev(self) -> float | None:
        gap = self.gap_hartree
        return None if gap is None else gap * HARTREE_TO_EV

    @property
    def gap_kcal(self) -> float | None:
        gap = self.gap_hartree
        return None if gap is None else gap * HARTREE_TO_KCAL

    @property
    def ground_multiplicity(self) -> int | None:
        gap = self.gap_hartree
        if gap is None:
            return None
        return self.low_multiplicity if gap > 0.0 else self.high_multiplicity

    def summary(self) -> str:
        if not self.ok:
            failed = [
                f"m={r.multiplicity}: {r.error}"
                for r in (self.low, self.high)
                if not r.ok
            ]
            return f"{self.label}: FAILED -- " + "; ".join(failed)
        return (
            f"{self.label}: CASSCF{self.active_space}/{self.basis} "
            f"m={self.low_multiplicity} {self.low.energy:.8f} Eh, "
            f"m={self.high_multiplicity} {self.high.energy:.8f} Eh, "
            f"gap = {self.gap_ev:+.4f} eV ({self.gap_kcal:+.2f} kcal/mol), "
            f"ground state m={self.ground_multiplicity}"
        )


def casscf_spin_gap(
    geometry: str,
    low_multiplicity: int,
    high_multiplicity: int,
    n_active_electrons: int,
    n_active_orbitals: int,
    *,
    charge: int = 0,
    basis: str = CASSCF_BASIS,
    n_total_electrons: int | None = None,
    label: str = "",
    options: dict[str, Any] | None = None,
    num_dets_print: int = 12,
    capture_ci_vector: bool = True,
    use_cache: bool = True,
    presets: Sequence[dict[str, Any]] | None = None,
) -> CASSCFSpinGap:
    """Two CASSCF single points in the *same* active space, and their splitting.

    Using one active space for both spin states is the whole point: it makes the
    two energies comparable in a way that two separately-chosen active spaces
    would not be, and it is what lets this number be held up against the DFT
    singlet-triplet gaps computed elsewhere in the project.
    """
    if high_multiplicity <= low_multiplicity:
        raise ValueError(
            f"high_multiplicity ({high_multiplicity}) must exceed "
            f"low_multiplicity ({low_multiplicity})"
        )
    if n_total_electrons is None:
        n_total_electrons = count_electrons(geometry, charge)

    space = _space_label(n_active_electrons, n_active_orbitals)
    base = label or f"casscf{space}"

    results = {}
    for multiplicity in (low_multiplicity, high_multiplicity):
        results[multiplicity] = casscf_energy(
            geometry,
            charge,
            multiplicity,
            n_active_electrons,
            n_active_orbitals,
            basis,
            n_total_electrons=n_total_electrons,
            label=f"{base}-m{multiplicity}",
            options=options,
            num_dets_print=num_dets_print,
            capture_ci_vector=capture_ci_vector,
            use_cache=use_cache,
            presets=presets,
        )

    return CASSCFSpinGap(
        label=base,
        active_space=space,
        basis=basis,
        charge=charge,
        low_multiplicity=low_multiplicity,
        high_multiplicity=high_multiplicity,
        low=results[low_multiplicity],
        high=results[high_multiplicity],
    )


# --------------------------------------------------------------------------
# The multireference diagnostic
# --------------------------------------------------------------------------


@dataclass
class MultireferenceDiagnostic:
    """How defensible is a single-reference (DFT) treatment of this state?

    Two independent measures, both reported when both are obtainable:

    ``reference_weight``
        ``C0**2``, the weight of the leading determinant in the CI vector.
        Scraped from detci's printed table because Psi4 1.11 exposes no working
        Python accessor for the CI vector.

    ``excess_unpaired``
        Head-Gordon effectively-unpaired electrons in excess of the
        ``multiplicity - 1`` that the spin state mandates.  Computed from the CI
        one-particle density matrix, no text parsing involved.

    :attr:`single_reference_ok` is ``True`` only if every measure that could be
    computed passes; it is ``None`` when none of them could be.
    """

    label: str
    method: str
    basis: str
    charge: int
    multiplicity: int
    active_space: str | None = None
    converged: bool = False
    energy: float | None = None
    largest_ci_coefficient: float | None = None
    reference_weight: float | None = None
    leading_determinants: list[tuple] = field(default_factory=list)
    natural_occupations: list[float] = field(default_factory=list)
    occupation_space: str = "unavailable"
    max_fractional_occupation: float | None = None
    n_effectively_unpaired: float | None = None
    n_effectively_unpaired_nl: float | None = None
    excess_unpaired: float | None = None
    sources: list[str] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def single_reference_ok(self) -> bool | None:
        if not self.converged:
            return None
        verdicts = []
        if self.reference_weight is not None:
            verdicts.append(self.reference_weight >= REFERENCE_WEIGHT_THRESHOLD)
        if self.excess_unpaired is not None:
            verdicts.append(self.excess_unpaired <= EXCESS_UNPAIRED_THRESHOLD)
        if not verdicts:
            return None
        return all(verdicts)

    def summary(self) -> str:
        if not self.converged:
            return f"{self.label}: {self.method.upper()} did not converge -- {self.error}"
        bits = [f"{self.label}: {self.method.upper()}"]
        if self.active_space:
            bits[-1] += self.active_space
        bits[-1] += f"/{self.basis} m={self.multiplicity} E={self.energy:.8f} Eh"
        if self.reference_weight is not None:
            bits.append(
                f"C0 = {self.largest_ci_coefficient:+.6f}, "
                f"weight C0^2 = {self.reference_weight:.4f}"
            )
        if self.n_effectively_unpaired is not None:
            bits.append(
                f"N_U = {self.n_effectively_unpaired:.3f} "
                f"(excess {self.excess_unpaired:+.3f}), "
                f"max fractional occupation = {self.max_fractional_occupation:.3f} "
                f"[{self.occupation_space}]"
            )
        verdict = self.single_reference_ok
        bits.append(
            "single-reference DFT defensible"
            if verdict
            else ("VERDICT UNAVAILABLE" if verdict is None else "MULTIREFERENCE -- DFT suspect")
        )
        return " | ".join(bits)


def _diagnostic_from_result(
    result: JobResult,
    *,
    label: str,
    active_space: str | None,
) -> MultireferenceDiagnostic:
    props = result.properties or {}
    sources: list[str] = []
    notes: list[str] = []

    if "reference_weight" in props:
        sources.append("ci-coefficients")
    elif "ci_vector_error" in props:
        notes.append(str(props["ci_vector_error"]))

    if "n_effectively_unpaired" in props:
        sources.append("natural-occupations")
    elif "occupation_error" in props:
        notes.append(str(props["occupation_error"]))

    if "property_error" in props:
        notes.append(str(props["property_error"]))

    return MultireferenceDiagnostic(
        label=label,
        method=result.method,
        basis=result.basis,
        charge=result.charge,
        multiplicity=result.multiplicity,
        active_space=active_space,
        converged=result.ok,
        energy=result.energy,
        largest_ci_coefficient=props.get("largest_ci_coefficient"),
        reference_weight=props.get("reference_weight"),
        leading_determinants=[tuple(d) for d in props.get("leading_determinants", [])],
        natural_occupations=list(props.get("natural_occupations", [])),
        occupation_space=props.get("occupation_space", "unavailable"),
        max_fractional_occupation=props.get("max_fractional_occupation"),
        n_effectively_unpaired=props.get("n_effectively_unpaired"),
        n_effectively_unpaired_nl=props.get("n_effectively_unpaired_nl"),
        excess_unpaired=props.get("excess_unpaired"),
        sources=sources,
        error=result.error,
        notes=notes,
    )


def multireference_diagnostic(
    geometry: str,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "casscf",
    n_active_electrons: int | None = None,
    n_active_orbitals: int | None = None,
    basis: str = CASSCF_BASIS,
    n_total_electrons: int | None = None,
    label: str = "",
    options: dict[str, Any] | None = None,
    num_dets_print: int = 12,
    use_cache: bool = True,
    presets: Sequence[dict[str, Any]] | None = None,
) -> MultireferenceDiagnostic:
    """Run a CI calculation and report how multireference the state is.

    ``method='casscf'`` needs an active space and gives the diagnostic *within*
    that space, which is the right question when the suspected static
    correlation is a known set of frontier orbitals (the Ni d manifold, the
    defect SOMO).  ``method='cisd'`` needs no active space and gives the
    diagnostic over the whole occupied-virtual space, which is the right
    question when you do not yet know where the trouble is -- but it is only
    meaningful for a state that is *nearly* single-reference to begin with,
    since CISD is itself built on one determinant.
    """
    method = method.lower()
    active_space: str | None = None

    if method == "casscf":
        if n_active_electrons is None or n_active_orbitals is None:
            raise ValueError(
                "method='casscf' requires n_active_electrons and n_active_orbitals"
            )
        active_space = _space_label(n_active_electrons, n_active_orbitals)
        result = casscf_energy(
            geometry,
            charge,
            multiplicity,
            n_active_electrons,
            n_active_orbitals,
            basis,
            n_total_electrons=n_total_electrons,
            label=label or f"mrdiag-casscf{active_space}",
            options=options,
            num_dets_print=num_dets_print,
            capture_ci_vector=True,
            use_cache=use_cache,
            presets=presets,
        )
    elif method == "cisd":
        if n_active_electrons is not None or n_active_orbitals is not None:
            raise ValueError("method='cisd' does not take an active space")
        spec = _cisd_spec(
            geometry,
            charge,
            multiplicity,
            basis,
            label or "mrdiag-cisd",
            options,
            num_dets_print,
            capture=True,
        )
        result = _run_ci(
            spec,
            multiplicity=multiplicity,
            presets=presets if presets is not None else CISD_PRESETS,
            capture=True,
            use_cache=use_cache,
        )
    else:
        raise ValueError(
            f"unsupported diagnostic method {method!r}; use 'casscf' or 'cisd'"
        )

    return _diagnostic_from_result(
        result,
        label=label or result.label,
        active_space=active_space,
    )
