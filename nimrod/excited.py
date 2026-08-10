"""Excited states and the optical readout of the colour centre.

The sensor reports on catalyst degradation through two channels.  This module
owns the optical one.

**The physics.**  Pristine pyrene is a closed-shell alternant PAH: its lowest
transitions are the familiar ``p``, ``alpha`` and ``beta`` bands built from the
delocalised pi system, and the lowest of them is symmetry-forbidden.  Bonding a
single aryl group to one pyrene CH carbon turns that carbon sp3.  Two things
happen at once:

* one carbon leaves the pi system, so the remaining 15 pi electrons are odd —
  the ground state is a **doublet** with an unpaired spin localised on the
  defect; and
* the conjugation path is cut, so the pi system is no longer a clean pyrene.
  The frontier orbital becomes a singly-occupied, defect-localised state that
  sits *inside* the pristine HOMO-LUMO gap.

The consequence is a new, strongly **red-shifted** and much brighter transition
that does not exist in the parent PAH.  That is the "colour centre": exactly the
same mechanism that makes an sp3 defect in a carbon nanotube emit ~200 meV below
the E11 exciton.  :func:`colour_centre_shift` is the function that measures it,
and it is the primary optical observable of the whole project.

**How the excited states are obtained.**  Linear-response TDDFT
(:func:`tddft_states`), with EOM-CCSD (:func:`eom_ccsd_states`) available as a
wavefunction-theory benchmark on systems small enough to afford it.

Psi4-specific realities that shaped this module — all four were verified
empirically against Psi4 1.11, not assumed:

1. ``TDSCF_STATES`` is *per irrep*.  Every molecule in this project is built
   with ``symmetry c1`` (see :func:`nimrod.psi4_driver.build_molecule`), so
   there is exactly one irrep and the option must be a **one-element list**,
   ``[n]``.  A bare int happens to work, a longer list raises
   ``ValidationError: TDSCF: States requested (...) do not match number of
   irreps (1)``.  Because Psi4 options are global and sticky, a stale value
   from a previous job produces that error on a job that is itself correct;
   :func:`nimrod.psi4_driver.clean_context` is what prevents this.

2. ``psi4.energy("td-<functional>")`` **silently computes the wrong functional**
   for some names.  ``run_tdscf_energy`` in ``proc.py`` does
   ``run_scf(name.strip("td-"))``, and :meth:`str.strip` removes *characters*,
   not a prefix.  So ``"td-wb97x-d"`` becomes ``"wb97x"`` — a real functional,
   so nothing raises, and you quietly get wB97X with no dispersion.  Verified:
   ``td-wb97x-d`` on H2O/6-31G reproduces plain ``wb97x`` to 1e-10 Eh and
   differs from ``wb97x-d`` by 0.88 mEh.  ``"td-tpssh"`` becomes ``"pssh"`` and
   dies with a bare ``KeyError: 'pssh'``.  This module therefore drives TDSCF
   through :func:`psi4.driver.procrouting.response.scf_response.tdscf_excitations`
   by default — the identical computation the ``td-`` string dispatches to,
   minus the name mangling — and :func:`tdscf_capability` refuses in advance
   any functional the string route would corrupt.

3. The ``td-`` string route also *throws away information*.  ``run_tdscf_energy``
   discards the return value of ``tdscf_excitations`` and keeps only the
   QCVariables, which do not record the spin symmetry of each root.  Calling the
   function directly returns a list of dicts carrying ``"SPIN"``,
   ``"SYMMETRY"`` and the transition eigenvectors, which is how this module
   labels singlets vs triplets and identifies the dominant orbital pair.

4. Psi4's TDSCF Vx kernel **does not support meta-GGAs or VV10**.  That rules
   out ``tpss``, ``tpssh`` and ``m06`` from :data:`nimrod.config.FUNCTIONAL_LADDER`
   — including ``tpssh``, the project's :data:`~nimrod.config.REFERENCE_FUNCTIONAL`.
   Optical work therefore runs on :data:`TDDFT_LADDER`, the non-meta subset,
   and that substitution is recorded rather than hidden.

**The requested roots are not guaranteed to be the lowest roots.**  This one is
not a Psi4 bug, it is how a Davidson solver with a
``guess = DENOMINATORS`` starting space behaves near degeneracies, and it bites
hard on symmetric aromatics.  Benzene at TD-B3LYP/6-31G, verified here:
asking for 8 roots returns 5.40, 6.26, 7.86, 7.96, 7.96, 7.99, 8.21, 8.21 eV,
all of them dark.  Asking for 14 roots on the identical job returns the same
list **with a doubly degenerate pair at 7.272 eV, f = 0.568, inserted at
positions 3 and 4** — the E1u band, the only transition in benzene anybody
actually sees.  A short root window silently dropped the entire absorption
spectrum.  :func:`verify_root_stability` exists to catch exactly this, and any
production optical number in this project should be taken from a window that it
certifies.

**Honesty about the open-shell case.**  The colour centre is a doublet radical,
so its absorption is a UKS TDDFT calculation, and spin-contaminated UKS TDDFT on
radicals is a known weak point of the method: the response space contains
spin-adapted doublets mixed with quartet character, which pushes low-lying
states down and can produce spurious roots.  :func:`occ_absorption` therefore
always returns the reference ``<S^2>`` alongside the transition and flags the
result when the contamination exceeds a threshold.  A reader who does not know
how far to trust the number is given the number that tells them.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    BASIS_TIERS,
    FUNCTIONAL_LADDER,
    HARTREE_TO_CM,
    HARTREE_TO_EV,
    HARTREE_TO_NM,
    SCF_PRESETS,
)
from .psi4_driver import JobResult, JobSpec, clean_context, run_energy

__all__ = [
    "ExcitedState",
    "TDDFTResult",
    "OCCAbsorption",
    "RootStability",
    "TDSCFCapability",
    "TDDFTError",
    "TDDFT_LADDER",
    "tdscf_capability",
    "tddft_states",
    "run_tddft",
    "verify_root_stability",
    "occ_absorption",
    "eom_ccsd_states",
    "run_eom_ccsd",
    "colour_centre_shift",
    "lowest_bright",
    "brightest",
    "absorption_spectrum",
    "reference_s_squared",
    "probe_tdscf_variables",
]


# --------------------------------------------------------------------------
# Which functionals TDSCF can actually run
# --------------------------------------------------------------------------

#: Multiplicity -> spin name, for labelling roots of an open-shell reference.
_MULTIPLICITY_NAME = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}

#: Oscillator strength below which a transition counts as dark.
DARK_THRESHOLD = 1.0e-3

#: The subset of :data:`nimrod.config.FUNCTIONAL_LADDER` that Psi4's TDSCF
#: kernel accepts.  The excluded members (``tpss``, ``tpssh``, ``m06``) are
#: meta-GGAs; see :func:`tdscf_capability`.  Populated lazily on first use so
#: that importing this module never requires Psi4.
TDDFT_LADDER: tuple[str, ...] = ("bp86", "pbe", "b3lyp", "pbe0", "wb97x-d")

#: Default functional for optical work.  PBE0's 25% exact exchange is the usual
#: compromise for valence pi->pi* transitions of aromatics; it is *not*
#: :data:`nimrod.config.REFERENCE_FUNCTIONAL` because that one (TPSSh) is a
#: meta-GGA and TDSCF cannot run it.
DEFAULT_TDDFT_FUNCTIONAL = "pbe0"


class TDDFTError(RuntimeError):
    """Raised when an excited-state calculation cannot be run or did not run."""


@dataclass(frozen=True)
class TDSCFCapability:
    """Whether — and how — Psi4's TDSCF can run a given functional.

    ``driver_string_safe`` is the guard against the ``str.strip("td-")`` bug
    described in the module docstring: it is False whenever
    ``("td-" + name).strip("td-") != name``, i.e. whenever the ``td-`` method
    string would silently dispatch to a *different* functional.
    """

    functional: str
    supported: bool
    reason: str = ""
    is_meta: bool = False
    needs_vv10: bool = False
    driver_string_safe: bool = True
    mangled_to: str | None = None
    hf_exchange: float | None = None
    range_separated: bool = False


def _strip_bug_target(functional: str) -> str:
    """What ``psi4.energy('td-<f>')`` will *actually* compute.

    Reproduces ``proc.run_tdscf_energy``'s ``name.strip('td-')`` exactly.
    """
    return f"td-{functional}".strip("td-")


def tdscf_capability(functional: str) -> TDSCFCapability:
    """Ask Psi4 whether its TDSCF kernel can handle ``functional``.

    Uses the real superfunctional object rather than a hardcoded list, so the
    answer stays correct if the Psi4 version changes.
    """
    import psi4

    from .psi4_driver import ensure_initialised

    ensure_initialised()
    name = functional.lower()
    mangled = _strip_bug_target(name)
    safe = mangled == name

    try:
        sup, _ = psi4.driver.dft.build_superfunctional(name, True)
    except Exception as exc:  # not a functional Psi4 knows at all
        return TDSCFCapability(
            functional=name,
            supported=False,
            reason=f"Psi4 does not recognise the functional: {type(exc).__name__}: {exc}",
            driver_string_safe=safe,
            mangled_to=None if safe else mangled,
        )

    is_meta = bool(sup.is_meta())
    vv10 = bool(sup.needs_vv10())
    reason = ""
    if is_meta:
        reason = "Psi4's TDSCF Vx kernel does not support meta-GGA functionals"
    elif vv10:
        reason = "Psi4's TDSCF Vx kernel does not support VV10 non-local correlation"

    return TDSCFCapability(
        functional=name,
        supported=not (is_meta or vv10),
        reason=reason,
        is_meta=is_meta,
        needs_vv10=vv10,
        driver_string_safe=safe,
        mangled_to=None if safe else mangled,
        hf_exchange=float(sup.x_alpha()),
        range_separated=bool(sup.is_x_lrc()),
    )


def tddft_ladder() -> tuple[str, ...]:
    """The members of :data:`nimrod.config.FUNCTIONAL_LADDER` TDSCF can run.

    Queries Psi4 rather than trusting :data:`TDDFT_LADDER`, which is only a
    cached answer for the version this module was written against.
    """
    return tuple(f.name for f in FUNCTIONAL_LADDER if tdscf_capability(f.name).supported)


# --------------------------------------------------------------------------
# A single excited state
# --------------------------------------------------------------------------


@dataclass
class ExcitedState:
    """One root of a linear-response or EOM calculation.

    Energies are carried in all four units the rest of the project needs, so no
    call site ever has to remember a conversion factor.  ``index`` is 1-based:
    root 1 is the lowest excited state, matching Psi4's own numbering.
    """

    index: int
    energy_ev: float
    energy_nm: float
    energy_cm: float
    oscillator_strength: float | None
    spin_label: str                       #: 'singlet' | 'triplet' | 'doublet' | ...
    energy_hartree: float = 0.0
    symmetry: str = "A"
    #: Largest single orbital contribution, e.g. ``"HOMO -> LUMO+1 (alpha)"``.
    dominant_pair: str | None = None
    #: Fraction of the right-eigenvector norm carried by ``dominant_pair``.
    dominant_weight: float | None = None
    #: Electric transition dipole, length gauge, atomic units.
    transition_dipole: list[float] | None = None
    #: True when the oscillator strength could not be computed (EOM-CCSD).
    oscillator_strength_available: bool = True

    @property
    def bright(self) -> bool:
        f = self.oscillator_strength
        return f is not None and f >= DARK_THRESHOLD

    def __str__(self) -> str:
        f = "n/a" if self.oscillator_strength is None else f"{self.oscillator_strength:.4f}"
        pair = f"  {self.dominant_pair}" if self.dominant_pair else ""
        return (
            f"S{self.index} {self.spin_label:8s} "
            f"{self.energy_ev:7.3f} eV  {self.energy_nm:7.1f} nm  f={f}{pair}"
        )


def _make_state(
    index: int,
    omega_hartree: float,
    oscillator_strength: float | None,
    spin_label: str,
    **extra: Any,
) -> ExcitedState:
    """Build an :class:`ExcitedState` from an excitation energy in hartree."""
    omega = float(omega_hartree)
    # A zero or negative root means the reference is unstable (triplet
    # instability); do not silently divide by it.
    nm = HARTREE_TO_NM / omega if omega > 1e-12 else math.inf
    return ExcitedState(
        index=index,
        energy_ev=omega * HARTREE_TO_EV,
        energy_nm=nm,
        energy_cm=omega * HARTREE_TO_CM,
        oscillator_strength=None if oscillator_strength is None else float(oscillator_strength),
        spin_label=spin_label,
        energy_hartree=omega,
        **extra,
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class TDDFTResult:
    """A complete TDDFT job: the roots plus everything needed to judge them."""

    states: list[ExcitedState]
    method: str
    basis: str
    reference: str
    multiplicity: int
    converged: bool
    ground_energy: float | None = None
    #: ``<S^2>`` of the SCF reference (None for a restricted reference).
    s_squared: float | None = None
    #: ``S(S+1)`` for the requested multiplicity.
    s_squared_ideal: float | None = None
    #: ``<S^2> - S(S+1)``; the honest measure of how far the reference is from
    #: a spin eigenfunction, and therefore of how far to trust the roots.
    spin_contamination: float | None = None
    preset: str | None = None
    wall_seconds: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.converged and bool(self.states) and self.error is None

    def bright_states(self, threshold: float = DARK_THRESHOLD) -> list[ExcitedState]:
        return [s for s in self.states if (s.oscillator_strength or 0.0) >= threshold]

    def summary(self) -> str:
        head = [
            f"{self.method}/{self.basis}  ref={self.reference}  "
            f"mult={self.multiplicity}  converged={self.converged}"
        ]
        if self.spin_contamination is not None:
            head.append(
                f"  <S^2> = {self.s_squared:.4f} (ideal {self.s_squared_ideal:.4f}, "
                f"contamination {self.spin_contamination:+.4f})"
            )
        for w in self.warnings:
            head.append(f"  ! {w}")
        if self.error:
            head.append(f"  ERROR: {self.error}")
        head.extend("  " + str(s) for s in self.states)
        return "\n".join(head)


@dataclass
class OCCAbsorption:
    """The colour-centre absorption of an open-shell defect.

    Two different states are reported because they answer two different
    questions and on a PAH radical they are routinely *not* the same root:

    ``lowest_bright``
        The lowest-energy root clearing the brightness threshold.  This is the
        colour-centre band — the absorption edge, the thing a photoluminescence
        experiment sees, and the number a degradation scan should track.
    ``strongest``
        The root with the largest oscillator strength anywhere in the window.
        On a large PAH this is usually a high-lying delocalised pi->pi* band
        that has nothing to do with the defect.

    ``lowest_bright`` falls back to ``strongest`` only when *no* root clears the
    threshold, and says so in :attr:`warnings` when it does.
    """

    lowest_bright: ExcitedState | None
    states: list[ExcitedState]
    s_squared: float | None
    s_squared_ideal: float | None
    spin_contamination: float | None
    reliable: bool
    method: str
    basis: str
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    strongest: ExcitedState | None = None

    @property
    def ok(self) -> bool:
        return self.lowest_bright is not None and self.error is None


# --------------------------------------------------------------------------
# Reference spin contamination
# --------------------------------------------------------------------------


def reference_s_squared(wfn) -> tuple[float, float] | None:
    """``(<S^2>, S(S+1))`` for an unrestricted reference, else ``None``.

    Uses the standard determinantal expression

    .. math::
        \\langle S^2 \\rangle = S_z(S_z+1) + N_\\beta
                                - \\sum_{ij} |\\langle \\phi_i^\\alpha |
                                  \\phi_j^\\beta \\rangle|^2

    evaluated from the occupied MO coefficients and the AO overlap.  This is
    the single scalar the optical module needs; the full spin analysis lives in
    :mod:`nimrod.spin`, and if that module exposes a compatible ``s_squared``
    helper it is preferred so the two modules cannot drift apart.
    """
    if wfn.same_a_b_orbs():
        return None

    na, nb = int(wfn.nalpha()), int(wfn.nbeta())
    sz = 0.5 * (na - nb)
    exact = sz * (sz + 1.0)

    external = _external_s_squared(wfn)
    if external is not None:
        return external, exact

    ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
    cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
    overlap = np.asarray(wfn.S())
    mixed = ca.T @ overlap @ cb          # <phi_i^alpha | phi_j^beta>
    return float(exact + nb - np.sum(mixed**2)), exact


#: Names in :mod:`nimrod.spin` that unambiguously mean ``<S^2>`` itself.  A name
#: like ``spin_contamination`` is deliberately *not* here: it is ambiguous
#: between ``<S^2>`` and ``<S^2> - S(S+1)``, and guessing wrong would corrupt
#: every reliability judgement downstream.  ``nimrod.spin.spin_squared`` was
#: checked against the expression below on CH3/UB3LYP/6-31G and agrees to
#: 4e-16, so the two modules cannot drift apart silently.
_S_SQUARED_NAMES = ("spin_squared", "s_squared")


def _external_s_squared(wfn) -> float | None:
    """Use :mod:`nimrod.spin`'s ``<S^2>`` if that module exposes one."""
    try:
        from . import spin as _spin  # type: ignore[attr-defined]
    except Exception:
        return None
    fn = None
    for name in _S_SQUARED_NAMES:
        candidate = getattr(_spin, name, None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        return None
    try:
        value = float(fn(wfn))
    except Exception:
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------
# Harvesting the TDSCF result
# --------------------------------------------------------------------------


def _as_array(matrix) -> np.ndarray:
    """psi4 ``Matrix`` / ``Vector`` / ndarray -> ndarray, psi4numpy style."""
    if hasattr(matrix, "to_array"):
        return np.asarray(matrix.to_array())
    return np.asarray(matrix)


def _orbital_label(index: int, n_occupied: int) -> str:
    """MO index (0-based, ascending) -> ``HOMO-2`` / ``LUMO+1`` style label."""
    if index < n_occupied:
        gap = n_occupied - 1 - index
        return "HOMO" if gap == 0 else f"HOMO-{gap}"
    gap = index - n_occupied
    return "LUMO" if gap == 0 else f"LUMO+{gap}"


def _dominant_pair(
    entry: dict[str, Any],
    n_alpha: int,
    n_beta: int,
    restricted: bool,
) -> tuple[str | None, float | None]:
    """Largest single orbital contribution to a transition.

    The "right eigenvector" Psi4 hands back is indexed ``[i, a]`` over occupied
    ``i`` and virtual ``a`` of one spin channel, but it is **not** the
    excitation amplitude ``X``: ``scf_response._solve_loop`` solves the reduced
    non-Hermitian problem and returns ``X + Y`` as the right vector and
    ``X - Y`` as the left one (for TDA, where ``Y = 0``, the right vector *is*
    ``X``).  We report the pair carrying the largest ``|X+Y|`` and its share of
    that vector's norm, so the weight describes the dominant character of the
    transition; it is not a normalised CI coefficient.

    ``restricted`` must be the reference's own ``same_a_b_orbs()`` flag, not a
    guess from comparing the two channels: for a restricted reference Psi4
    stores the *same* array under both the ALPHA and BETA keys, so the single
    excitation is counted twice and its share has to be doubled.  An
    unrestricted reference with equal alpha/beta occupations (a broken-symmetry
    singlet) can have near-identical channels without that being true.
    """
    channels: list[tuple[str, np.ndarray, int]] = []
    for key, tag, nocc in (
        ("RIGHT EIGENVECTOR ALPHA", "alpha", n_alpha),
        ("RIGHT EIGENVECTOR BETA", "beta", n_beta),
    ):
        if key in entry and entry[key] is not None:
            arr = _as_array(entry[key])
            if arr.ndim == 2 and arr.size:
                channels.append((tag, arr, nocc))
    if not channels:
        return None, None

    same_orbitals = restricted and len(channels) == 2

    total_norm = sum(float(np.sum(arr**2)) for _, arr, _ in channels)
    if total_norm <= 0.0:
        return None, None

    tag, arr, nocc = max(channels, key=lambda c: np.abs(c[1]).max())
    i, a = np.unravel_index(int(np.argmax(np.abs(arr))), arr.shape)
    amplitude = float(arr[i, a])
    weight = amplitude**2 / total_norm
    if same_orbitals:
        # Restricted reference: alpha and beta are the same excitation counted
        # twice, so the single pair really carries twice this share.
        weight *= 2.0

    label = f"{_orbital_label(int(i), nocc)} -> {_orbital_label(nocc + int(a), nocc)}"
    if not same_orbitals:
        label += f" ({tag})"
    return label, weight


def _spin_label_for(api_spin: str, restricted: bool, multiplicity: int) -> str:
    """Trustworthy spin label for a root.

    ``tdscf_excitations`` tags every root with the name of the solver branch it
    came from — ``"singlet"`` or ``"triplet"``.  For a *restricted* reference
    that is correct.  For an unrestricted reference there is only one branch and
    it is still labelled ``"singlet"``, which is simply wrong: excitations of a
    doublet reference are doublets (with quartet contamination).  We therefore
    override the tag whenever the reference is unrestricted.
    """
    if restricted:
        return str(api_spin).lower()
    return _MULTIPLICITY_NAME.get(multiplicity, f"M={multiplicity}")


def _states_from_api(
    raw: Sequence[dict[str, Any]],
    *,
    restricted: bool,
    multiplicity: int,
    n_alpha: int,
    n_beta: int,
) -> list[ExcitedState]:
    """Convert the ``tdscf_excitations`` return value into :class:`ExcitedState`."""
    states: list[ExcitedState] = []
    for i, entry in enumerate(raw, start=1):
        pair, weight = _dominant_pair(entry, n_alpha, n_beta, restricted)
        dipole = entry.get("ELECTRIC DIPOLE TRANSITION MOMENT (LEN)")
        states.append(
            _make_state(
                i,
                entry["EXCITATION ENERGY"],
                entry.get("OSCILLATOR STRENGTH (LEN)"),
                _spin_label_for(entry.get("SPIN", "singlet"), restricted, multiplicity),
                symmetry=str(entry.get("SYMMETRY", "A")),
                dominant_pair=pair,
                dominant_weight=weight,
                transition_dipole=None if dipole is None else [float(x) for x in np.ravel(_as_array(dipole))],
            )
        )
    return states


#: ``<prefix> ROOT 0 -> ROOT <n> <property>``.  The irrep-resolved variants of
#: these keys read ``ROOT 0 (A) -> ROOT 1 (A) ...`` and so fail to match, which
#: is exactly what we want: one key per root per property.
_ROOT_RE = re.compile(r"^(?P<prefix>[A-Z0-9()\-+*' ]+?) ROOT 0 -> ROOT (?P<n>\d+) (?P<prop>.+)$")

_WANTED_PROPS = {
    "EXCITATION ENERGY": "energy",
    "OSCILLATOR STRENGTH (LEN)": "f_len",
    "OSCILLATOR STRENGTH (VEL)": "f_vel",
}


def _collect_root_variables(variables: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    """Group ``ROOT 0 -> ROOT n`` QCVariables by variable-name prefix and root.

    Psi4 publishes the same numbers under several prefixes — the method-specific
    one (``TD-B3LYP``) and a generic one (``TD-DFT``) — plus ``- A TRANSITION``
    duplicates.  We keep only the plain scalar forms and let the caller choose
    the prefix.
    """
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    for key, value in variables.items():
        match = _ROOT_RE.match(key)
        if match is None:
            continue
        prop = match.group("prop")
        if prop not in _WANTED_PROPS:
            continue
        if not isinstance(value, (int, float, np.floating)):
            continue
        prefix = match.group("prefix").strip()
        root = int(match.group("n"))
        grouped.setdefault(prefix, {}).setdefault(root, {})[_WANTED_PROPS[prop]] = float(value)
    return grouped


def _pick_prefix(grouped: dict[str, dict[int, dict[str, float]]], functional: str) -> str | None:
    """Choose which QCVariable prefix to read the roots from."""
    if not grouped:
        return None
    preferred = f"TD-{functional.upper()}"
    for candidate in (preferred, "TD-DFT", "TD-HF", "TD-SCF"):
        if candidate in grouped:
            return candidate
    return sorted(grouped)[0]


def _states_from_variables(
    variables: dict[str, Any],
    functional: str,
    *,
    restricted: bool,
    multiplicity: int,
    triplets: str,
) -> tuple[list[ExcitedState], list[str]]:
    """Parse roots out of the QCVariable namespace (the ``td-`` string route).

    This path cannot recover the spin symmetry of a root: the variables do not
    carry it.  When triplets were requested we fall back on the fact that Psi4
    writes an *exactly* zero length-gauge oscillator strength for a spin-
    forbidden root while singlets come out at ~1e-26 numerical noise.  That is a
    sentinel, not a quantum number, so the caller gets a warning saying so.
    """
    warnings: list[str] = []
    grouped = _collect_root_variables(variables)
    prefix = _pick_prefix(grouped, functional)
    if prefix is None:
        return [], ["no 'ROOT 0 -> ROOT n' QCVariables were published by the TDSCF module"]
    if prefix != f"TD-{functional.upper()}":
        warnings.append(
            f"harvested QCVariables under prefix {prefix!r} rather than "
            f"'TD-{functional.upper()}'"
        )

    heuristic_spin = triplets.upper() == "ALSO" and restricted
    if heuristic_spin:
        warnings.append(
            "singlet/triplet labels from the QCVariable route are inferred from "
            "an exactly-zero oscillator strength, not read from Psi4; use "
            "harvest='api' for authoritative spin labels"
        )

    states: list[ExcitedState] = []
    for root in sorted(grouped[prefix]):
        entry = grouped[prefix][root]
        if "energy" not in entry:
            continue
        f_len = entry.get("f_len")
        if not restricted:
            spin = _MULTIPLICITY_NAME.get(multiplicity, f"M={multiplicity}")
        elif triplets.upper() == "ONLY":
            spin = "triplet"
        elif heuristic_spin:
            spin = "triplet" if f_len == 0.0 else "singlet"
        else:
            spin = "singlet"
        states.append(_make_state(root, entry["energy"], f_len, spin))
    return states, warnings


# --------------------------------------------------------------------------
# TDDFT
# --------------------------------------------------------------------------


def _tdscf_hook(
    *,
    n_states: int,
    triplets: str,
    tda: bool,
    r_convergence: float | None,
    maxiter: int,
    multiplicity: int,
):
    """Property hook that runs TDSCF on the converged SCF wavefunction.

    Returns only JSON-serialisable data, because
    :func:`nimrod.psi4_driver.run_energy` caches ``JobResult.properties`` to
    disk.  Psi4 ``Matrix`` objects are reduced to the scalars we want before
    they ever leave this function.
    """

    def hook(wfn, molecule) -> dict[str, Any]:
        from psi4.driver.procrouting.response.scf_response import tdscf_excitations

        kwargs: dict[str, Any] = {
            # PER IRREP: c1 means exactly one irrep, so exactly one element.
            "states": [int(n_states)],
            "triplets": triplets,
            "tda": bool(tda),
            "maxiter": int(maxiter),
        }
        if r_convergence is not None:
            kwargs["r_convergence"] = float(r_convergence)

        raw = tdscf_excitations(wfn, **kwargs)
        restricted = bool(wfn.same_a_b_orbs())
        states = _states_from_api(
            raw,
            restricted=restricted,
            multiplicity=multiplicity,
            n_alpha=int(wfn.nalpha()),
            n_beta=int(wfn.nbeta()),
        )

        props: dict[str, Any] = {
            "states": [asdict(s) for s in states],
            "restricted": restricted,
        }
        spin = reference_s_squared(wfn)
        if spin is not None:
            props["s_squared"], props["s_squared_ideal"] = spin
        return props

    return hook


def _variables_hook(functional: str, *, triplets: str, multiplicity: int):
    """Property hook for the ``td-<functional>`` driver-string route."""

    def hook(wfn, molecule) -> dict[str, Any]:
        import psi4

        variables = dict(wfn.variables())
        if not variables:
            variables = dict(psi4.core.variables())
        restricted = bool(wfn.same_a_b_orbs())
        states, warnings = _states_from_variables(
            variables,
            functional,
            restricted=restricted,
            multiplicity=multiplicity,
            triplets=triplets,
        )
        props: dict[str, Any] = {
            "states": [asdict(s) for s in states],
            "restricted": restricted,
            "harvest_warnings": warnings,
        }
        spin = reference_s_squared(wfn)
        if spin is not None:
            props["s_squared"], props["s_squared_ideal"] = spin
        return props

    return hook


def run_tddft(
    geometry: str,
    *,
    functional: str = DEFAULT_TDDFT_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 6,
    tda: bool = False,
    triplets: str = "NONE",
    reference: str | None = None,
    r_convergence: float | None = None,
    maxiter: int = 120,
    harvest: str = "api",
    options: dict[str, Any] | None = None,
    label: str = "",
    use_cache: bool = True,
) -> TDDFTResult:
    """Run linear-response TDDFT and return every root with its provenance.

    Parameters
    ----------
    harvest
        ``"api"`` (default) calls ``tdscf_excitations`` directly on the
        converged SCF wavefunction.  ``"driver"`` uses the documented
        ``psi4.energy("td-<functional>")`` method string instead.  Both run the
        identical solver — ``run_tdscf_energy`` is a thin wrapper around the
        same function — but the driver route mangles some functional names and
        discards the spin labels (see the module docstring), so it is refused
        outright for an unsafe name and warns about what it cannot report.
    triplets
        ``"NONE"`` / ``"ALSO"`` / ``"ONLY"``.  Only meaningful for a restricted
        reference; Psi4 raises ``ValidationError`` if asked for triplets from an
        unrestricted one, so this function rejects that combination up front
        with a clearer message.
    n_states
        Total number of roots.  With ``triplets="ALSO"`` Psi4 splits them
        roughly 50-50 between singlets and triplets, preferring singlets.

    Never raises for a physics failure: an unconverged SCF or a rejected
    functional comes back as a :class:`TDDFTResult` with ``ok == False`` and a
    populated ``error``.  :func:`tddft_states` is the raising convenience
    wrapper.
    """
    functional = functional.lower()
    triplets = triplets.upper()
    warnings: list[str] = []

    def failed(message: str) -> TDDFTResult:
        return TDDFTResult(
            states=[],
            method=functional,
            basis=basis,
            reference=reference or ("uks" if multiplicity > 1 else "rks"),
            multiplicity=multiplicity,
            converged=False,
            error=message,
            warnings=warnings,
        )

    if n_states < 1:
        return failed("n_states must be at least 1")
    if triplets not in {"NONE", "ALSO", "ONLY"}:
        return failed(f"triplets must be NONE/ALSO/ONLY, got {triplets!r}")

    capability = tdscf_capability(functional)
    if not capability.supported:
        return failed(f"{functional}: {capability.reason}")

    resolved_reference = (reference or ("uks" if multiplicity > 1 else "rks")).lower()
    unrestricted = resolved_reference.startswith("u")
    if triplets != "NONE" and unrestricted:
        return failed(
            "Psi4 cannot compute triplet responses from an unrestricted "
            f"reference ({resolved_reference}); the roots of a multiplicity-"
            f"{multiplicity} reference are already open-shell excitations"
        )

    if harvest == "driver" and not capability.driver_string_safe:
        return failed(
            f"psi4.energy('td-{functional}') would silently compute "
            f"'{capability.mangled_to}' instead: proc.run_tdscf_energy uses "
            "name.strip('td-'), which removes characters rather than a prefix. "
            "Use harvest='api'."
        )
    if harvest not in {"api", "driver"}:
        return failed(f"harvest must be 'api' or 'driver', got {harvest!r}")

    # These options are inert on the "api" path (the solver takes its settings
    # as arguments) but they are still set, because JobSpec.fingerprint() hashes
    # the option dict and the disk cache must distinguish a 4-root job from a
    # 10-root one.
    job_options: dict[str, Any] = {
        "tdscf_states": [int(n_states)],   # PER IRREP; c1 -> exactly one entry
        "tdscf_tda": bool(tda),
        "tdscf_triplets": triplets,
        "tdscf_maxiter": int(maxiter),
        "save_jk": True,                   # the response solver needs wfn.jk()
    }
    if r_convergence is not None:
        job_options["tdscf_r_convergence"] = float(r_convergence)
    job_options.update(options or {})

    method = functional if harvest == "api" else f"td-{functional}"
    spec = JobSpec(
        geometry=geometry,
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        reference=resolved_reference,
        options=job_options,
        label=label or f"tddft-{functional}-{basis}",
    )

    if harvest == "api":
        hook = _tdscf_hook(
            n_states=n_states,
            triplets=triplets,
            tda=tda,
            r_convergence=r_convergence,
            maxiter=maxiter,
            multiplicity=multiplicity,
        )
    else:
        hook = _variables_hook(functional, triplets=triplets, multiplicity=multiplicity)
        warnings.append(
            "harvest='driver': Psi4 discards the structured TDSCF result, so "
            "orbital characters and authoritative spin labels are unavailable"
        )

    job: JobResult = run_energy(spec, use_cache=use_cache, property_hook=hook)
    return _result_from_job(job, functional, basis, multiplicity, warnings)


def _result_from_job(
    job: JobResult,
    functional: str,
    basis: str,
    multiplicity: int,
    warnings: list[str],
) -> TDDFTResult:
    """Assemble a :class:`TDDFTResult`, including from a cached job."""
    props = job.properties or {}
    warnings = list(warnings) + list(props.get("harvest_warnings", []))

    states = [ExcitedState(**d) for d in props.get("states", [])]
    s2 = props.get("s_squared")
    s2_ideal = props.get("s_squared_ideal")
    contamination = None if s2 is None or s2_ideal is None else float(s2) - float(s2_ideal)

    error = job.error or props.get("property_error")
    if job.converged and not states and error is None:
        error = "SCF converged but the TDSCF solver returned no roots"

    return TDDFTResult(
        states=states,
        method=functional,
        basis=basis,
        reference=job.reference,
        multiplicity=multiplicity,
        converged=job.converged and bool(states) and error is None,
        ground_energy=job.energy,
        s_squared=None if s2 is None else float(s2),
        s_squared_ideal=None if s2_ideal is None else float(s2_ideal),
        spin_contamination=contamination,
        preset=job.preset,
        wall_seconds=job.wall_seconds,
        error=error,
        warnings=warnings,
    )


def tddft_states(
    geometry: str,
    functional: str = DEFAULT_TDDFT_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 6,
    **kwargs: Any,
) -> list[ExcitedState]:
    """Excited states of one structure, as a plain list.

    Thin wrapper over :func:`run_tddft` for call sites that only want the roots.
    Raises :class:`TDDFTError` if the calculation did not produce any, so a
    failure can never be mistaken for "no excited states".
    """
    result = run_tddft(
        geometry,
        functional=functional,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        n_states=n_states,
        **kwargs,
    )
    if not result.ok:
        raise TDDFTError(result.error or "TDDFT produced no states")
    return result.states


# --------------------------------------------------------------------------
# Did the solver actually find the lowest roots?
# --------------------------------------------------------------------------


@dataclass
class RootStability:
    """Whether a short root window contains everything a longer one finds.

    See the module docstring: Psi4's Davidson solver can converge ``n`` roots
    that are *not* the ``n`` lowest, and the states it skips are sometimes the
    only bright ones.  This is the check.
    """

    n_requested: int
    n_reference: int
    window_ev: float                      #: top of the short run's energy window
    missing: list[ExcitedState] = field(default_factory=list)
    brightest_missing: ExcitedState | None = None
    error: str | None = None

    @property
    def stable(self) -> bool:
        return self.error is None and not self.missing

    def summary(self) -> str:
        if self.error:
            return f"root stability: could not be checked ({self.error})"
        if not self.missing:
            return (
                f"root stability: OK — {self.n_requested} roots reproduce every "
                f"state the {self.n_reference}-root run finds below "
                f"{self.window_ev:.3f} eV"
            )
        worst = self.brightest_missing
        lines = [
            f"root stability: FAILED — the {self.n_requested}-root run MISSED "
            f"{len(self.missing)} state(s) below its own ceiling of "
            f"{self.window_ev:.3f} eV:"
        ]
        lines.extend(f"    missed {s}" for s in self.missing)
        if worst is not None and worst.bright:
            lines.append(
                f"    the missed set includes a BRIGHT state "
                f"(f = {worst.oscillator_strength:.4f} at {worst.energy_ev:.3f} eV); "
                "any spectrum from the short run is wrong"
            )
        return "\n".join(lines)


def verify_root_stability(
    geometry: str,
    *,
    n_states: int,
    n_reference: int | None = None,
    tolerance_ev: float = 1.0e-3,
    **kwargs: Any,
) -> RootStability:
    """Re-run with more roots and report anything the short window missed.

    Runs the same TDDFT job twice — once with ``n_states`` roots, once with
    ``n_reference`` (default ``2 * n_states``) — and lists every state of the
    long run that lies *below the short run's own highest root* yet does not
    appear in it.  Such a state cannot be explained by the window being too
    small; it was skipped.

    Both jobs go through the ordinary disk cache, so calling this after a
    production run costs only the longer of the two calculations.
    """
    n_reference = n_reference or 2 * n_states
    if n_reference <= n_states:
        return RootStability(n_states, n_reference, 0.0, error="n_reference must exceed n_states")

    short = run_tddft(geometry, n_states=n_states, **kwargs)
    long = run_tddft(geometry, n_states=n_reference, **kwargs)
    if not short.ok or not long.ok:
        return RootStability(
            n_states,
            n_reference,
            0.0,
            error=short.error or long.error or "one of the two runs produced no roots",
        )

    ceiling = max(s.energy_ev for s in short.states)

    # Match each long-run state to *at most one* short-run state and consume it.
    # A plain "is there any short root at this energy?" test is blind to
    # degeneracy: if the short window found one member of a degenerate pair and
    # the long window finds two, both long members match the same short root and
    # the check wrongly reports OK.  Benzene's bright E1u band *is* a degenerate
    # pair, so that is precisely the failure this function exists to catch.
    unmatched = sorted(s.energy_ev for s in short.states)
    missing: list[ExcitedState] = []
    for state in sorted(long.states, key=lambda s: s.energy_ev):
        if state.energy_ev > ceiling + tolerance_ev:
            continue
        hit = next(
            (i for i, e in enumerate(unmatched) if abs(state.energy_ev - e) <= tolerance_ev),
            None,
        )
        if hit is None:
            missing.append(state)
        else:
            unmatched.pop(hit)
    return RootStability(
        n_requested=n_states,
        n_reference=n_reference,
        window_ev=ceiling,
        missing=missing,
        brightest_missing=brightest(missing),
    )


# --------------------------------------------------------------------------
# The open-shell colour centre
# --------------------------------------------------------------------------

#: ``<S^2> - S(S+1)`` above which UKS TDDFT on a radical is not to be trusted
#: quantitatively.  0.1 corresponds to roughly 10% quartet admixture in the
#: doublet reference, which is where the literature stops treating linear
#: response from a broken-symmetry determinant as quantitative.
SPIN_CONTAMINATION_LIMIT = 0.10


def occ_absorption(
    geometry: str,
    *,
    functional: str = DEFAULT_TDDFT_FUNCTIONAL,
    basis: str = BASIS_TIERS["screen"],
    charge: int = 0,
    multiplicity: int = 2,
    n_states: int = 8,
    search_lowest: int | None = None,
    threshold: float = DARK_THRESHOLD,
    contamination_limit: float = SPIN_CONTAMINATION_LIMIT,
    **kwargs: Any,
) -> OCCAbsorption:
    """Absorption of the open-shell colour centre: the lowest bright root.

    The defect state is a doublet, so this is a UKS calculation and the roots
    are doublet->doublet excitations.  Rather than take root 1 blindly — the
    lowest root of a radical is very often dark, being an excitation localised
    on the untouched part of the pi system — this scans the lowest
    ``search_lowest`` roots and returns the lowest one that clears
    ``threshold``.  That is the absorption edge: the band that carries the
    colour of the defect.  The globally strongest root of the window is
    reported alongside as :attr:`OCCAbsorption.strongest`, because on a large
    PAH that one is usually a delocalised pi->pi* band belonging to the intact
    part of the host rather than to the defect, and tracking it across a
    degradation scan would follow a different state at every point.

    The reference ``<S^2>`` comes back with the result and sets ``reliable``.
    UKS linear response inherits the spin contamination of its reference:
    a doublet determinant carrying quartet character produces roots that are
    not spin eigenstates, and their energies are systematically too low.  This
    function will still return a number when the reference is contaminated;
    it just refuses to let the caller not know.
    """
    # Triplet responses are meaningless from — and rejected for — an
    # unrestricted reference; silently drop the argument rather than raising a
    # duplicate-keyword TypeError at a caller that passed it out of habit.
    kwargs.pop("triplets", None)
    result = run_tddft(
        geometry,
        functional=functional,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        n_states=n_states,
        triplets="NONE",
        **kwargs,
    )

    warnings = list(result.warnings)
    if not result.ok:
        return OCCAbsorption(
            lowest_bright=None,
            states=result.states,
            s_squared=result.s_squared,
            s_squared_ideal=result.s_squared_ideal,
            spin_contamination=result.spin_contamination,
            reliable=False,
            method=functional,
            basis=basis,
            warnings=warnings,
            error=result.error,
        )

    window = result.states[: search_lowest or len(result.states)]
    strongest = brightest(window)
    state = lowest_bright(window, threshold)
    if state is None and strongest is not None:
        # Nothing clears the bar.  Hand back the best available rather than
        # None, but make it impossible to mistake for a real absorption band.
        state = strongest
        warnings.append(
            f"no root in the lowest {len(window)} exceeds f = {threshold:g}; "
            f"returning the brightest available (f = {state.oscillator_strength:.2e}). "
            "The colour-centre transition may lie above the requested window."
        )

    contamination = result.spin_contamination
    reliable = contamination is not None and abs(contamination) <= contamination_limit
    if contamination is None:
        warnings.append(
            "reference is restricted, so no spin contamination was measured; "
            "expected an unrestricted reference for a doublet colour centre"
        )
    elif not reliable:
        warnings.append(
            f"reference <S^2> = {result.s_squared:.4f} vs ideal "
            f"{result.s_squared_ideal:.4f} (contamination {contamination:+.4f} > "
            f"{contamination_limit:g}): UKS TDDFT excitation energies on a "
            "spin-contaminated radical are biased low and should be treated as "
            "qualitative"
        )

    return OCCAbsorption(
        lowest_bright=state,
        strongest=strongest,
        states=result.states,
        s_squared=result.s_squared,
        s_squared_ideal=result.s_squared_ideal,
        spin_contamination=contamination,
        reliable=reliable,
        method=functional,
        basis=basis,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# EOM-CCSD benchmark
# --------------------------------------------------------------------------

#: EOM-CCSD needs conventional (or CD) integrals; the density-fitted SCF presets
#: in :data:`nimrod.config.SCF_PRESETS` would leave the CC and SCF integrals
#: inconsistent, so the CC path gets its own short ladder.
_CC_PRESETS = (
    {
        "label": "cc-pk",
        "options": {
            "scf_type": "pk",
            "guess": "sad",
            "e_convergence": 1e-9,
            "d_convergence": 1e-8,
            "maxiter": 200,
        },
    },
    {
        "label": "cc-pk-soscf",
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

_CC_ROOT_RE = re.compile(r"^(?P<prefix>CCSD|CC|EOM-CCSD|CC2|CC3) ROOT (?P<n>\d+) TOTAL ENERGY$")


def _eom_hook(multiplicity: int):
    """Harvest EOM roots.

    Psi4's CC modules publish their ``ROOT n`` variables into the **global**
    QCVariable namespace and *not* onto the returned wavefunction — the opposite
    of TDSCF, which populates both.  Verified on Psi4 1.11: for
    ``energy('eom-ccsd')`` the returned ``wfn.variables()`` contains no ``ROOT``
    keys at all, so this hook reads ``psi4.core.variables()`` first.
    """

    def hook(wfn, molecule) -> dict[str, Any]:
        import psi4

        variables = dict(psi4.core.variables())
        variables.update({k: v for k, v in wfn.variables().items() if "ROOT" in k})

        energies: dict[int, float] = {}
        for key, value in variables.items():
            match = _CC_ROOT_RE.match(key)
            if match is None or not isinstance(value, (int, float, np.floating)):
                continue
            root = int(match.group("n"))
            # 'CCSD' and 'CC' carry identical values; last write wins harmlessly.
            energies[root] = float(value)

        if 0 not in energies:
            return {"states": [], "property_error": "no 'CC ROOT 0 TOTAL ENERGY' variable found"}

        ground = energies[0]
        spin = _MULTIPLICITY_NAME.get(multiplicity, f"M={multiplicity}")
        states: list[ExcitedState] = []
        for root in sorted(k for k in energies if k > 0):
            # Oscillator strengths require the EOM transition densities, which
            # Psi4 only forms for a properties() run; report their absence
            # rather than inventing a zero.
            f_key = f"CC ROOT 0 -> ROOT {root} OSCILLATOR STRENGTH (LEN)"
            f_val = variables.get(f_key)
            states.append(
                _make_state(
                    root,
                    energies[root] - ground,
                    None if f_val is None else float(f_val),
                    spin,
                    oscillator_strength_available=f_val is not None,
                )
            )
        return {"states": [asdict(s) for s in states], "cc_ground_energy": ground}

    return hook


def run_eom_ccsd(
    geometry: str,
    *,
    basis: str = BASIS_TIERS["correlated"],
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 3,
    reference: str | None = None,
    method: str = "eom-ccsd",
    options: dict[str, Any] | None = None,
    label: str = "",
    use_cache: bool = True,
) -> TDDFTResult:
    """EOM-CCSD excitation energies — the wavefunction-theory reference.

    Cost is O(N^6) and grows brutally with basis size: this is for the small
    validation systems (water, formaldehyde, ethylene), never for the sensor.
    ``roots_per_irrep`` is the CC analogue of ``tdscf_states`` and is subject to
    the same per-irrep rule, so with ``symmetry c1`` it is a one-element list.

    Oscillator strengths are **not** produced: Psi4 forms EOM transition
    densities only in a ``properties()`` run.  Roots come back with
    ``oscillator_strength is None`` and ``oscillator_strength_available`` False
    rather than a fabricated zero.
    """
    resolved_reference = (reference or ("rhf" if multiplicity == 1 else "uhf")).lower()
    job_options: dict[str, Any] = {
        "roots_per_irrep": [int(n_states)],   # PER IRREP; c1 -> one element
        "r_convergence": 1e-6,
        "freeze_core": False,
    }
    job_options.update(options or {})

    spec = JobSpec(
        geometry=geometry,
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        reference=resolved_reference,
        options=job_options,
        label=label or f"{method}-{basis}",
    )
    job = run_energy(
        spec,
        use_cache=use_cache,
        property_hook=_eom_hook(multiplicity),
        presets=_CC_PRESETS,
    )
    result = _result_from_job(job, method, basis, multiplicity, [])
    if result.states and all(not s.oscillator_strength_available for s in result.states):
        result.warnings.append(
            "EOM-CCSD oscillator strengths are unavailable from energy(); "
            "use TDDFT for intensities or a properties() run for EOM densities"
        )
    return result


def eom_ccsd_states(
    geometry: str,
    basis: str = BASIS_TIERS["correlated"],
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 3,
    **kwargs: Any,
) -> list[ExcitedState]:
    """EOM-CCSD roots as a plain list; raises :class:`TDDFTError` on failure."""
    result = run_eom_ccsd(
        geometry,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        n_states=n_states,
        **kwargs,
    )
    if not result.ok:
        raise TDDFTError(result.error or "EOM-CCSD produced no states")
    return result.states


# --------------------------------------------------------------------------
# The optical readout
# --------------------------------------------------------------------------


def lowest_bright(
    states: Iterable[ExcitedState],
    threshold: float = DARK_THRESHOLD,
) -> ExcitedState | None:
    """Lowest-energy state with ``f >= threshold``.

    This, not root 1, is what a UV-vis spectrum shows you.
    """
    candidates = [s for s in states if (s.oscillator_strength or 0.0) >= threshold]
    return min(candidates, key=lambda s: s.energy_hartree) if candidates else None


def brightest(states: Iterable[ExcitedState]) -> ExcitedState | None:
    """State with the largest oscillator strength."""
    candidates = [s for s in states if s.oscillator_strength is not None]
    return max(candidates, key=lambda s: s.oscillator_strength) if candidates else None


def colour_centre_shift(
    pristine_states: Sequence[ExcitedState],
    defect_states: Sequence[ExcitedState],
    threshold: float = DARK_THRESHOLD,
) -> dict[str, Any]:
    """Spectral shift induced by the sp3 aryl defect — **the optical readout**.

    Compares the lowest bright transition of the pristine host with that of the
    defect structure.  Sign convention: ``red_shift_ev`` is *positive* for a red
    shift (the defect transition lies lower in energy / longer in wavelength),
    which is the expected direction — the defect state sits inside the parent
    HOMO-LUMO gap.

    Both bands are located by oscillator strength rather than by root index,
    because the lowest root of an alternant PAH is symmetry-forbidden and the
    lowest root of the radical is frequently dark as well; comparing root 1 to
    root 1 would compare two states that are not the same band.

    The shift of the *strongest* band in each set is reported alongside, under
    the ``strongest_*`` keys.  The two answers differ whenever the lowest
    detectable band is not the main absorption — which is the normal situation
    for a PAH — and a reader deserves to see both rather than one number chosen
    for them.

    Returns a flat dict of JSON-serialisable scalars (plus ``error`` when a band
    could not be located), suitable for direct inclusion in a results table.

    .. warning::
       Garbage in, garbage out: if either state list came from a root window
       that skipped a bright state (see :func:`verify_root_stability`) this
       function will happily report a shift between the wrong two bands.
    """
    pristine = lowest_bright(pristine_states, threshold)
    defect = lowest_bright(defect_states, threshold)

    out: dict[str, Any] = {
        "threshold": threshold,
        "n_pristine_states": len(pristine_states),
        "n_defect_states": len(defect_states),
    }
    if pristine is None or defect is None:
        missing = []
        if pristine is None:
            missing.append("pristine")
        if defect is None:
            missing.append("defect")
        out["error"] = (
            f"no transition with f >= {threshold:g} among the "
            f"{' and '.join(missing)} states"
        )
        return out

    delta_h = pristine.energy_hartree - defect.energy_hartree   # >0 == red shift
    out.update(
        {
            "pristine_index": pristine.index,
            "pristine_ev": pristine.energy_ev,
            "pristine_nm": pristine.energy_nm,
            "pristine_f": pristine.oscillator_strength,
            "pristine_character": pristine.dominant_pair,
            "defect_index": defect.index,
            "defect_ev": defect.energy_ev,
            "defect_nm": defect.energy_nm,
            "defect_f": defect.oscillator_strength,
            "defect_character": defect.dominant_pair,
            "red_shift_ev": delta_h * HARTREE_TO_EV,
            "red_shift_nm": defect.energy_nm - pristine.energy_nm,
            "red_shift_cm": delta_h * HARTREE_TO_CM,
            "shift_ev": -delta_h * HARTREE_TO_EV,   # signed defect - pristine
            "direction": "red" if delta_h > 0 else ("blue" if delta_h < 0 else "none"),
            "intensity_ratio": (
                (defect.oscillator_strength / pristine.oscillator_strength)
                if pristine.oscillator_strength
                else None
            ),
            "error": None,
        }
    )

    # The main absorption band of each set, which need not be the lowest one.
    p_max, d_max = brightest(pristine_states), brightest(defect_states)
    if p_max is not None and d_max is not None:
        out.update(
            {
                "pristine_strongest_ev": p_max.energy_ev,
                "pristine_strongest_nm": p_max.energy_nm,
                "pristine_strongest_f": p_max.oscillator_strength,
                "defect_strongest_ev": d_max.energy_ev,
                "defect_strongest_nm": d_max.energy_nm,
                "defect_strongest_f": d_max.oscillator_strength,
                "strongest_red_shift_ev": p_max.energy_ev - d_max.energy_ev,
                "strongest_red_shift_nm": d_max.energy_nm - p_max.energy_nm,
            }
        )
    return out


def absorption_spectrum(
    states: Sequence[ExcitedState],
    *,
    fwhm_ev: float = 0.3,
    grid_ev: tuple[float, float, int] = (0.5, 8.0, 751),
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-broadened stick spectrum, ``(energies_ev, intensity)``.

    Each root contributes a normalised Gaussian of area equal to its oscillator
    strength, so the integrated intensity is conserved and two spectra computed
    on the same grid are directly comparable.  Dark and intensity-less
    (EOM-CCSD) roots contribute nothing.  Intensity is in arbitrary units — this
    is for figures and band-maximum tracking, not for extinction coefficients.
    """
    lo, hi, n = grid_ev
    energies = np.linspace(lo, hi, int(n))
    intensity = np.zeros_like(energies)
    sigma = fwhm_ev / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    for state in states:
        f = state.oscillator_strength
        if not f:
            continue
        intensity += f * norm * np.exp(-0.5 * ((energies - state.energy_ev) / sigma) ** 2)
    return energies, intensity


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def probe_tdscf_variables(
    geometry: str,
    *,
    functional: str = "b3lyp",
    basis: str = "6-31G",
    charge: int = 0,
    multiplicity: int = 1,
    n_states: int = 2,
    triplets: str = "NONE",
) -> dict[str, Any]:
    """Run a tiny TDDFT job and report exactly which QCVariables Psi4 published.

    Kept in the shipped module rather than a throwaway script because Psi4's
    QCVariable names are version-dependent and the parser in
    :func:`_states_from_variables` is only as good as this probe says it is.
    Run it after any Psi4 upgrade.

    Returns ``{"root_variables": {...}, "prefixes": [...], "api_keys": [...]}``.
    """
    import psi4

    from psi4.driver.procrouting.response.scf_response import tdscf_excitations

    from .psi4_driver import build_molecule

    spec = JobSpec(
        geometry=geometry,
        method=functional,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        options={},
    )
    with clean_context():
        mol = build_molecule(spec)
        psi4.set_options(
            {
                "basis": basis,
                "reference": spec.resolved_reference(),
                "scf_type": "df",
                "e_convergence": 1e-8,
                "d_convergence": 1e-7,
                "save_jk": True,
            }
        )
        _, wfn = psi4.energy(functional, return_wfn=True, molecule=mol)
        raw = tdscf_excitations(wfn, states=[int(n_states)], triplets=triplets.upper())

        wfn_vars = {k: v for k, v in wfn.variables().items() if "ROOT" in k.upper()}
        global_vars = {k: v for k, v in psi4.core.variables().items() if "ROOT" in k.upper()}
        grouped = _collect_root_variables({**global_vars, **wfn_vars})

        return {
            "psi4_version": psi4.__version__,
            "n_root_variables_on_wfn": len(wfn_vars),
            "n_root_variables_global": len(global_vars),
            "prefixes": sorted(grouped),
            "root_variables": {
                prefix: {root: dict(props) for root, props in sorted(roots.items())}
                for prefix, roots in sorted(grouped.items())
            },
            "api_keys": sorted(raw[0]) if raw else [],
            "api_spins": [entry.get("SPIN") for entry in raw],
            "api_energies": [float(entry["EXCITATION ENERGY"]) for entry in raw],
        }
