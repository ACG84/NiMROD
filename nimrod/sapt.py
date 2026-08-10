"""Interaction-energy decomposition for solvent capture at the vacated site.

Scope, stated up front because it is easy to get wrong
------------------------------------------------------
SAPT decomposes a *non-covalent* interaction between two closed-shell
monomers.  It therefore cannot say anything about the tether in this project:
the colour centre is bonded to the catalyst through a real C-C single bond, and
there is no sensible way to cut that bond into two SAPT monomers.  Any attempt
to "SAPT the tether" would be measuring the cost of homolysis, not an
interaction energy.

What SAPT *is* the right tool for here is the chemistry that ligand loss
creates.  When one Ni-N chelate bond stretches past ~2.5 A the metal is no
longer four-coordinate, and a coordination site opens on the nickel.  In any
real solvent that site does not stay empty: a solvent molecule binds it.  This
module puts a water molecule there and asks what kind of site it actually is:

    electrostatics   a hard, charge-controlled site -- the water is being
                     polarised by an exposed, positively charged metal
    induction        genuine dative bonding -- charge transfer from the water
                     lone pair into the empty metal d orbital.  This is the
                     term that grows as the site opens, and it is the
                     signature of a real open coordination site rather than a
                     mere dent in the surface
    dispersion       a soft, unselective site: the water is only sticking to
                     the ligand periphery, not to the metal
    exchange         Pauli repulsion, always positive; if it dominates, the
                     probe has been placed too close or into a wall

Reading those four numbers along the degradation coordinate turns "the Ni-N
bond got longer" into a chemical statement about what the degraded catalyst
would do to a solvent.

Both fragments here are closed-shell -- the bare bis(iminoenolato)Ni(II)
catalyst is a square-planar d8 S = 0 complex, and water is S = 0 -- so ordinary
closed-shell SAPT0 applies.  Note that the *full* 54-atom sensor assembly is a
doublet and is therefore **not** a valid SAPT monomer; the profiles here are run
on the catalyst alone at the same degradation geometry.
:func:`assert_closed_shell_fragments` enforces that, because Psi4 will not.

Psi4 notes verified against 1.11 in this environment
----------------------------------------------------
* The component psivars really are the ones in :data:`SAPT_VARIABLES`.  Both an
  unprefixed family (``'SAPT ELST ENERGY'``) and a level-prefixed family
  (``'SAPT0 ELST ENERGY'``) are set; they carry identical values for SAPT0.
  Beware ``'SAPT ENERGY'``, which is a *different* variable and is 0.0.
* ``psi4.energy('sapt0', ...)`` does **not** refuse open-shell monomers.  Psi4
  1.11 ships Gonthier's open-shell SAPT0 and will quietly run a UHF-monomer
  calculation instead.  That is a different method with different error
  characteristics, so this module refuses it by default rather than letting a
  mis-specified fragment silently change the theory being reported.
* ``-d3`` / ``-d3bj`` functionals are **not usable in this environment**: the
  ``dftd3`` Python module is absent and QCEngine raises
  ``ResourceError: Program s-dftd3 is registered with QCEngine, but cannot be
  found``.  Natively-implemented dispersion (``wb97x-d``, and the VV10
  functionals ``wb97x-v`` / ``wb97m-v`` / ``b97m-v``) does work, so the
  counterpoise cross-check defaults to ``wb97x-d``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import HARTREE_TO_KCAL, SCF_PRESETS
from .geometry import ATOMIC_NUMBER, Structure, plane_normal
from .psi4_driver import (
    JobResult,
    JobSpec,
    clean_context,
    load_cached,
    run_energy,
    store_cached,
)

__all__ = [
    "SaptScopeError",
    "WATER_OH",
    "WATER_HOH_DEG",
    "DEFAULT_NI_O_DISTANCE",
    "DEFAULT_SAPT_BASIS",
    "DEFAULT_CP_METHOD",
    "SAPT_VARIABLES",
    "SAPT_EXTRA_VARIABLES",
    "water_probe",
    "coordination_plane_normal",
    "vacated_site_direction",
    "Fragment",
    "parse_fragments",
    "assert_closed_shell_fragments",
    "two_fragment_geometry",
    "DimerGeometry",
    "catalyst_water_dimer",
    "SaptComponents",
    "sapt0_interaction",
    "CounterpoiseBinding",
    "counterpoise_binding",
    "CapturePoint",
    "CaptureProfile",
    "capture_profile",
]


class SaptScopeError(ValueError):
    """The requested calculation is outside the closed-shell SAPT scope."""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Experimental gas-phase water geometry (Benedict 1956).  Used rigidly: the
#: probe is a spectator whose internal relaxation is far smaller than the
#: quantities being decomposed.
WATER_OH = 0.9572          #: angstrom
WATER_HOH_DEG = 104.52     #: degrees

#: A representative Ni(II)-OH2 bond length.  Terminal aqua ligands on
#: divalent first-row metals sit at 2.0-2.1 A; 2.05 is the usual starting guess.
DEFAULT_NI_O_DISTANCE = 2.05

#: SAPT0 needs diffuse functions to describe dispersion and induction tails.
#: jun-cc-pvdz (aug-cc-pvdz with diffuse functions stripped from hydrogen and
#: the highest angular momentum removed) is the standard cost/accuracy
#: compromise for SAPT0 and is what the SAPT0 error statistics were fitted to.
#: It carries no Ni parameters, so metal-containing dimers fall back to a def2
#: basis; see :func:`sapt0_interaction`.
DEFAULT_SAPT_BASIS = "jun-cc-pvdz"

#: Basis used when any fragment contains an element jun-cc-pvdz does not cover.
DEFAULT_SAPT_METAL_BASIS = "def2-svpd"

#: Dispersion-corrected functional for the supermolecular cross-check.  ``-d3``
#: variants cannot be used in this environment (see the module docstring);
#: wb97x-d carries its dispersion natively and does work.
DEFAULT_CP_METHOD = "wb97x-d"

#: The four components plus the total, as *actually published* by Psi4 1.11.
#: Probed, not assumed.  ``'SAPT ENERGY'`` is deliberately absent: it exists and
#: is always 0.0, and is a trap.
SAPT_VARIABLES = {
    "electrostatics": "SAPT ELST ENERGY",
    "exchange": "SAPT EXCH ENERGY",
    "induction": "SAPT IND ENERGY",
    "dispersion": "SAPT DISP ENERGY",
    "total": "SAPT TOTAL ENERGY",
}

#: Diagnostic terms harvested alongside the four components.  Free, because the
#: SAPT0 run has already computed them.
SAPT_EXTRA_VARIABLES = {
    "elst10r": "SAPT ELST10,R ENERGY",
    "exch10": "SAPT EXCH10 ENERGY",
    "ind20r": "SAPT IND20,R ENERGY",
    "exch_ind20r": "SAPT EXCH-IND20,R ENERGY",
    "disp20": "SAPT DISP20 ENERGY",
    "exch_disp20": "SAPT EXCH-DISP20 ENERGY",
    "charge_transfer": "SAPT CT ENERGY",
    "delta_hf": "SAPT HF(2) ENERGY",
    "sapt_hf_total": "SAPT HF TOTAL ENERGY",
}

#: Elements covered by the jun-cc-pVXZ family.  Anything outside this triggers
#: the def2 fallback.
_JUN_CC_ELEMENTS = frozenset("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar".split())

#: Atomic numbers, H to Rn.  :data:`nimrod.geometry.ATOMIC_NUMBER` covers only
#: the elements the structure builders use, and a fragment parser that silently
#: skips an element it does not recognise would miscount electrons and let a
#: genuinely open-shell fragment through the guard.  So the table here is
#: complete, and an unrecognised symbol is an error rather than a no-op.
_ELEMENT_Z: dict[str, int] = {
    symbol: z
    for z, symbol in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
        "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
        "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
        "Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split(),
        start=1,
    )
}
assert all(_ELEMENT_Z[s] == z for s, z in ATOMIC_NUMBER.items()), (
    "sapt._ELEMENT_Z disagrees with geometry.ATOMIC_NUMBER"
)

#: Method-name suffixes that route through the missing ``s-dftd3`` program.
_UNAVAILABLE_DISPERSION = ("-d3", "-d3bj", "-d3zero", "-d3m", "-d3mbj", "-d2")


# --------------------------------------------------------------------------
# The probe molecule and where to put it
# --------------------------------------------------------------------------


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise ValueError("cannot normalise a zero-length vector")
    return v / n


def _perpendicular_to(axis: np.ndarray, hint: Sequence[float] | None = None) -> np.ndarray:
    """A unit vector perpendicular to ``axis``, as close to ``hint`` as possible."""
    axis = _unit(axis)
    if hint is not None:
        hint = np.asarray(hint, dtype=float)
        residual = hint - float(np.dot(hint, axis)) * axis
        if float(np.linalg.norm(residual)) > 1e-6:
            return _unit(residual)
    # No usable hint: any perpendicular will do.
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, axis))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    return _unit(trial - float(np.dot(trial, axis)) * axis)


def water_probe(
    origin: Sequence[float],
    lone_pair_axis: Sequence[float],
    *,
    plane_hint: Sequence[float] | None = None,
    r_oh: float = WATER_OH,
    hoh_deg: float = WATER_HOH_DEG,
    name: str = "water-probe",
) -> Structure:
    """A rigid water with its oxygen at ``origin``, lone pairs along an axis.

    ``lone_pair_axis`` is the direction the oxygen lone-pair density points --
    i.e. the direction the water should be "aimed".  The two hydrogens are
    placed symmetrically on the *opposite* side, each subtending half of
    ``hoh_deg`` from the reversed axis, which puts the O-H bonds at
    ``180 - hoh_deg/2`` = 127.7 degrees to the aiming direction.  That is the
    C2v ("lone-pair-first") arrangement adopted by terminal aqua ligands.

    ``plane_hint`` selects which way the HOH plane faces: the hydrogens are
    displaced along the component of ``plane_hint`` perpendicular to the axis.
    For a square-planar metal the right hint is the coordination-plane normal,
    which puts the hydrogens above and below the plane instead of into the two
    cis donors.
    """
    origin = np.asarray(origin, dtype=float)
    axis = _unit(lone_pair_axis)
    perp = _perpendicular_to(axis, plane_hint)

    half = math.radians(hoh_deg) / 2.0
    back = -axis  # the H-H bisector
    offsets = [
        r_oh * (math.cos(half) * back + math.sin(half) * sign * perp)
        for sign in (1.0, -1.0)
    ]
    coords = np.vstack([origin, origin + offsets[0], origin + offsets[1]])
    return Structure(["O", "H", "H"], coords, name)


def coordination_plane_normal(struct: Structure, metal_index: int) -> np.ndarray | None:
    """Normal of the plane through the metal and its donor atoms.

    Returns ``None`` if the metal has fewer than two donors, in which case no
    plane is defined and the caller should fall back to an arbitrary
    orientation.
    """
    donors = [
        j for j in struct.neighbours(metal_index)
        if struct.symbols[j] in ("N", "O", "S", "P", "Cl", "F")
    ]
    if len(donors) < 2:
        return None
    points = np.vstack([struct.coords[metal_index], struct.coords[donors]])
    if len(points) < 3:
        return None
    try:
        return plane_normal(points)
    except ValueError:
        return None


def vacated_site_direction(
    struct: Structure,
    metal_index: int,
    *,
    leaving_index: int | None = None,
) -> np.ndarray:
    """Unit vector from the metal towards the open coordination site.

    With ``leaving_index`` (the dissociating donor) the answer is unambiguous:
    the vacancy is exactly where that donor used to be, so the direction is
    metal -> leaving atom.  This is the right call along a ligand-loss
    coordinate, where the departing nitrogen is still present in the structure
    but too far away to be bonded.

    Without it the direction is taken as the least-blocked one, ``-sum_i
    u(metal -> donor_i)``.  For an intact square-planar complex that sum
    cancels and there is no open site; the function then refuses rather than
    returning a numerically arbitrary axis.
    """
    if leaving_index is not None:
        return _unit(struct.coords[leaving_index] - struct.coords[metal_index])

    donors = [
        j for j in struct.neighbours(metal_index)
        if struct.symbols[j] != "H" and j != metal_index
    ]
    if not donors:
        raise ValueError(
            f"atom {metal_index} has no bonded donors; cannot infer a vacated site"
        )
    total = np.zeros(3)
    for j in donors:
        total += _unit(struct.coords[j] - struct.coords[metal_index])
    if float(np.linalg.norm(total)) < 0.2:
        raise ValueError(
            "the donor set around the metal is (near-)symmetric, so no open "
            "coordination site is defined; pass leaving_index to say which "
            "ligand has dissociated"
        )
    return _unit(-total)


# --------------------------------------------------------------------------
# Two-fragment geometries and the closed-shell guard
# --------------------------------------------------------------------------


@dataclass
class Fragment:
    """One monomer parsed out of a multi-fragment Psi4 geometry block."""

    symbols: list[str]
    charge: int
    multiplicity: int
    explicit_header: bool  #: was a "charge mult" line actually present?
    #: Ghost (basis-set-only) centres carry no electrons but do occupy lines.
    n_ghosts: int = 0

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def n_electrons(self) -> int:
        return sum(_ELEMENT_Z[s] for s in self.symbols) - self.charge

    @property
    def parity_consistent(self) -> bool:
        """Is the stated multiplicity reachable with this electron count?"""
        unpaired = self.multiplicity - 1
        return (self.n_electrons - unpaired) % 2 == 0 and self.n_electrons >= unpaired


def parse_fragments(geometry: str) -> list[Fragment]:
    """Split a Psi4 geometry block on ``--`` and read each fragment header.

    Deliberately does not go through Psi4: the guard has to be able to explain
    what is wrong with a geometry *before* anything expensive is launched, and
    a pure-Python parse gives a much better error message than a libmints
    exception.  A fragment with no leading ``charge multiplicity`` line is
    recorded as Psi4 would default it, ``0 1``, with ``explicit_header`` False.
    """
    fragments: list[Fragment] = []
    for index, chunk in enumerate(geometry.split("--"), start=1):
        symbols: list[str] = []
        ghosts = 0
        charge, multiplicity, explicit = 0, 1, False
        for raw in chunk.strip().splitlines():
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            lowered = parts[0].lower()
            if lowered in ("symmetry", "units", "unit", "no_reorient", "no_com",
                           "nocom", "noreorient", "pubchem:"):
                continue
            if len(parts) == 2 and not symbols:
                try:
                    charge, multiplicity = int(parts[0]), int(parts[1])
                    explicit = True
                    continue
                except ValueError:
                    pass
            if len(parts) < 4:
                continue
            try:
                [float(p) for p in parts[1:4]]
            except ValueError:
                continue  # not a Cartesian line (Z-matrix, variable, ...)

            token = parts[0]
            is_ghost = token.lower().startswith("gh(") and token.endswith(")")
            if is_ghost:
                token = token[3:-1]
            # Strip Psi4's mass/label decorations: "O@17.999", "C1", "Ni_a".
            token = token.split("@")[0].split("_")[0]
            symbol = token[:2].capitalize()
            if symbol not in _ELEMENT_Z:
                symbol = token[:1].upper()
            if symbol not in _ELEMENT_Z:
                raise SaptScopeError(
                    f"fragment {index}: cannot identify the element on line "
                    f"{line!r}. The electron count -- and therefore the "
                    f"closed-shell guard -- would be wrong, so this is refused "
                    f"rather than skipped."
                )
            if is_ghost:
                ghosts += 1
            else:
                symbols.append(symbol)
        if symbols or ghosts:
            fragments.append(
                Fragment(symbols, charge, multiplicity, explicit, n_ghosts=ghosts)
            )
    return fragments


def assert_closed_shell_fragments(geometry: str) -> list[Fragment]:
    """Refuse anything that is not a closed-shell dimer, with a real reason.

    Psi4 1.11 will *not* do this for us.  It ships an open-shell SAPT0
    (Gonthier's UHF-monomer formulation) and silently switches to it when a
    fragment carries a multiplicity above one -- so a mis-specified fragment
    does not fail, it quietly changes which theory produced the numbers you
    then plot.  Since the whole point of this module is a defensible chemical
    statement about the open site, that substitution has to be an error.
    """
    fragments = parse_fragments(geometry)
    if len(fragments) != 2:
        raise SaptScopeError(
            f"SAPT needs exactly two fragments separated by '--'; found "
            f"{len(fragments)}. Check that the geometry block uses '--' as the "
            f"separator and carries a 'charge multiplicity' line per fragment."
        )

    for i, frag in enumerate(fragments, start=1):
        if not frag.parity_consistent:
            raise SaptScopeError(
                f"fragment {i} ({''.join(sorted(set(frag.symbols)))}, "
                f"{frag.n_atoms} atoms) has {frag.n_electrons} electrons, which "
                f"cannot form a multiplicity-{frag.multiplicity} state. Check the "
                f"'charge multiplicity' line for that fragment."
            )
        if frag.multiplicity != 1:
            raise SaptScopeError(
                f"fragment {i} is specified as multiplicity {frag.multiplicity}, "
                f"but this module implements closed-shell SAPT only. Psi4 1.11 "
                f"does have an open-shell SAPT0, and it will run silently on "
                f"this input, but it is a different method with different error "
                f"characteristics and nothing in NiMROD validates it -- so it is "
                f"out of scope here. In particular the full sensor assembly is a "
                f"DOUBLET and is not a valid monomer: run the SAPT profile on the "
                f"bare catalyst at the same degradation geometry instead."
            )
        if frag.n_electrons % 2 != 0:
            raise SaptScopeError(
                f"fragment {i} has an odd electron count ({frag.n_electrons}) and "
                f"therefore cannot be closed-shell. It is an open-shell radical; "
                f"open-shell SAPT is out of scope for this module."
            )
    return fragments


def two_fragment_geometry(
    fragment_a: Structure,
    fragment_b: Structure,
    *,
    charge_a: int = 0,
    multiplicity_a: int = 1,
    charge_b: int = 0,
    multiplicity_b: int = 1,
) -> str:
    """Assemble the two-fragment Psi4 geometry string SAPT expects.

    Per-fragment ``charge multiplicity`` lines and a ``--`` separator, and
    *no* global charge line and no ``symmetry``/``no_com`` directives:
    :func:`nimrod.psi4_driver.build_molecule` detects the ``--`` and appends
    those itself while suppressing the global header.  Emitting them here would
    duplicate them.
    """
    return (
        f"{charge_a} {multiplicity_a}\n"
        f"{fragment_a.to_psi4()}\n"
        f"--\n"
        f"{charge_b} {multiplicity_b}\n"
        f"{fragment_b.to_psi4()}"
    )


@dataclass
class DimerGeometry:
    """A catalyst + probe dimer, with the bookkeeping needed to interpret it."""

    geometry: str                #: Psi4 two-fragment block
    structure: Structure         #: both fragments concatenated, A then B
    n_atoms_a: int
    n_atoms_b: int
    charges: tuple[int, int]
    multiplicities: tuple[int, int]
    metal_index: int             #: index of the metal in the *combined* structure
    oxygen_index: int            #: index of the probe oxygen in the combined structure
    metal_oxygen_distance: float
    site_direction: tuple[float, float, float]
    #: Shortest catalyst-probe distance **excluding the metal**, in angstrom.
    #: The metal is excluded because the metal-oxygen contact is the dative bond
    #: we are deliberately forming; including it would make every docked probe
    #: look like a clash.
    closest_contact: float
    closest_pair: tuple[int, int]

    def clash_warning(self, threshold: float = 2.0) -> str | None:
        """Flag a probe placement that has been jammed into the ligand shell.

        A coarse geometric tripwire, not a van der Waals analysis: any non-metal
        catalyst atom closer than ``threshold`` to a water atom will dominate the
        exchange term for reasons that have nothing to do with the metal site.

        The usual cause on a ligand-loss coordinate is the *departing donor
        itself*.  :func:`nimrod.geometry.degradation_series` pulls the nitrogen
        out along the Ni-N axis, which is exactly the axis the vacated site
        points down, so the receding nitrogen sits on top of the probe until the
        Ni-N distance exceeds roughly ``metal_oxygen_distance + 3``.  Relaxed
        scan geometries, where the arm swings aside, do not have this problem.
        """
        if self.closest_contact < threshold:
            i, j = self.closest_pair
            return (
                f"closest catalyst-probe contact is "
                f"{self.structure.symbols[i]}{i}...{self.structure.symbols[j]}{j} "
                f"at {self.closest_contact:.2f} A, below {threshold:.2f} A; the "
                f"probe is inside the ligand shell and the decomposition will be "
                f"exchange-dominated for geometric rather than chemical reasons"
            )
        return None


def catalyst_water_dimer(
    catalyst: Structure,
    metal_index: int,
    *,
    leaving_index: int | None = None,
    direction: Sequence[float] | None = None,
    metal_oxygen_distance: float = DEFAULT_NI_O_DISTANCE,
    charge: int = 0,
    multiplicity: int = 1,
    plane_hint: Sequence[float] | None = None,
) -> DimerGeometry:
    """Dock a water at the vacated coordination site, lone-pair-first.

    The oxygen goes at ``metal_oxygen_distance`` from the metal along the open
    direction, and the water is aimed so its lone-pair axis points back at the
    metal.  The HOH plane is set perpendicular to the coordination plane by
    default (see :func:`water_probe`), which keeps the hydrogens clear of the
    remaining cis donors.

    Supply either ``leaving_index`` (the dissociating donor -- the usual case
    along the degradation coordinate) or an explicit ``direction``.
    """
    if direction is None:
        axis = vacated_site_direction(catalyst, metal_index, leaving_index=leaving_index)
    else:
        axis = _unit(direction)

    if plane_hint is None:
        plane_hint = coordination_plane_normal(catalyst, metal_index)

    origin = catalyst.coords[metal_index] + metal_oxygen_distance * axis
    probe = water_probe(origin, -axis, plane_hint=plane_hint)

    combined = Structure(
        list(catalyst.symbols) + list(probe.symbols),
        np.vstack([catalyst.coords, probe.coords]),
        f"{catalyst.name or 'catalyst'}-h2o",
    )

    n_a = len(catalyst)
    separations = np.linalg.norm(
        catalyst.coords[:, None, :] - probe.coords[None, :, :], axis=-1
    )
    # Mask the metal: its short contact with the probe oxygen is the dative bond
    # being modelled, not a clash.
    masked = separations.copy()
    masked[metal_index, :] = np.inf
    flat = int(np.argmin(masked))
    ia, ib = divmod(flat, masked.shape[1])

    return DimerGeometry(
        geometry=two_fragment_geometry(
            catalyst, probe,
            charge_a=charge, multiplicity_a=multiplicity,
            charge_b=0, multiplicity_b=1,
        ),
        structure=combined,
        n_atoms_a=n_a,
        n_atoms_b=len(probe),
        charges=(charge, 0),
        multiplicities=(multiplicity, 1),
        metal_index=metal_index,
        oxygen_index=n_a,
        metal_oxygen_distance=float(
            np.linalg.norm(probe.coords[0] - catalyst.coords[metal_index])
        ),
        site_direction=(float(axis[0]), float(axis[1]), float(axis[2])),
        closest_contact=float(masked[ia, ib]),
        closest_pair=(int(ia), int(n_a + ib)),
    )


# --------------------------------------------------------------------------
# SAPT0
# --------------------------------------------------------------------------


@dataclass
class SaptComponents:
    """A SAPT decomposition, in kcal/mol.

    Sign convention is Psi4's and is the physical one: attractive terms
    negative, Pauli repulsion positive, ``total`` the sum of the four.
    """

    electrostatics: float
    exchange: float
    induction: float
    dispersion: float
    total: float
    basis: str
    level: str = "sapt0"
    converged: bool = True
    error: str | None = None
    label: str = ""
    wall_seconds: float = 0.0
    #: Higher-resolution terms (delta-HF, exch-ind20, disp20, ...) in kcal/mol.
    extras: dict[str, float] = field(default_factory=dict)

    # -- derived ------------------------------------------------------------

    @property
    def attractive(self) -> float:
        """Sum of the three attractive components (should be negative)."""
        return self.electrostatics + self.induction + self.dispersion

    @property
    def component_sum(self) -> float:
        return self.electrostatics + self.exchange + self.induction + self.dispersion

    def percent_attractive(self) -> dict[str, float]:
        """Each attractive term as a percentage of the total attraction.

        This is how a SAPT decomposition is normally read: the *ratio* of
        electrostatics to induction to dispersion characterises the bonding,
        and unlike the raw numbers it is only weakly sensitive to how far away
        the probe was placed.
        """
        magnitudes = {
            "electrostatics": abs(self.electrostatics),
            "induction": abs(self.induction),
            "dispersion": abs(self.dispersion),
        }
        denominator = sum(magnitudes.values())
        if denominator < 1e-12:
            return {k: float("nan") for k in magnitudes}
        return {k: 100.0 * v / denominator for k, v in magnitudes.items()}

    def character(self, margin: float = 10.0) -> str:
        """A one-word chemical reading of the site.

        ``margin`` is the percentage-point lead a component needs over the
        runner-up before the site is called after it; below that the site is
        reported as mixed.  This is a labelling convenience, not a measurement.
        """
        if not self.converged:
            return "unconverged"
        if self.total > 0.0:
            return "repulsive"
        shares = self.percent_attractive()
        ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        (top, top_share), (_, second_share) = ranked[0], ranked[1]
        if top_share - second_share < margin:
            return "mixed"
        return {
            "electrostatics": "electrostatic",
            "induction": "dative/charge-transfer",
            "dispersion": "dispersion-bound",
        }[top]

    def warnings(self, tolerance: float = 1e-4) -> list[str]:
        """Physical sanity checks.  An empty list means nothing looked wrong."""
        issues: list[str] = []
        if not self.converged:
            issues.append(f"calculation did not converge: {self.error}")
            return issues
        if self.exchange < 0.0:
            issues.append(
                f"exchange is {self.exchange:.3f} kcal/mol but Pauli repulsion "
                f"must be positive; the fragments or the harvest are wrong"
            )
        if self.electrostatics > 0.0:
            issues.append(
                f"electrostatics is repulsive ({self.electrostatics:+.3f} "
                f"kcal/mol), which happens for like-charged fragments but is "
                f"unexpected for a neutral probe on a neutral catalyst"
            )
        if self.dispersion > 0.0:
            issues.append(
                f"dispersion is {self.dispersion:+.3f} kcal/mol; London "
                f"dispersion is always attractive"
            )
        drift = abs(self.component_sum - self.total)
        if drift > tolerance:
            issues.append(
                f"components sum to {self.component_sum:.6f} but the reported "
                f"total is {self.total:.6f} kcal/mol (drift {drift:.2e}); the "
                f"psivar harvest is inconsistent"
            )
        return issues

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if not self.converged:
            return f"SAPT {self.level}/{self.basis}: FAILED ({self.error})"
        return (
            f"{self.level}/{self.basis}  "
            f"elst {self.electrostatics:+8.3f}  exch {self.exchange:+8.3f}  "
            f"ind {self.induction:+8.3f}  disp {self.dispersion:+8.3f}  "
            f"=> {self.total:+8.3f} kcal/mol  [{self.character()}]"
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def failed(cls, basis: str, level: str, error: str, label: str = "") -> "SaptComponents":
        nan = float("nan")
        return cls(nan, nan, nan, nan, nan, basis=basis, level=level,
                   converged=False, error=error, label=label)

    @classmethod
    def from_properties(
        cls,
        properties: Mapping[str, Any],
        *,
        basis: str,
        level: str = "sapt0",
        label: str = "",
        wall_seconds: float = 0.0,
    ) -> "SaptComponents":
        """Build from the ``properties`` dict left by :func:`_sapt_property_hook`."""
        missing = [k for k in SAPT_VARIABLES if k not in properties]
        if missing:
            return cls.failed(
                basis, level,
                f"SAPT psivars missing from the harvest: {missing}; "
                f"available keys were {sorted(properties)[:12]}",
                label,
            )
        return cls(
            electrostatics=float(properties["electrostatics"]),
            exchange=float(properties["exchange"]),
            induction=float(properties["induction"]),
            dispersion=float(properties["dispersion"]),
            total=float(properties["total"]),
            basis=basis,
            level=level,
            converged=True,
            label=label,
            wall_seconds=wall_seconds,
            extras={
                k: float(v) for k, v in properties.items()
                if k in SAPT_EXTRA_VARIABLES and isinstance(v, (int, float))
            },
        )


def _harvest(variables: Mapping[str, Any], name: str, level: str) -> float | None:
    """Read one psivar, preferring the level-prefixed name where it exists.

    Psi4 1.11 sets both ``'SAPT ELST ENERGY'`` and ``'SAPT0 ELST ENERGY'`` after
    a SAPT0 run, with identical values.  Higher levels (ssapt0, sapt2+) set
    their own prefixed family, so the prefixed name is tried first and the
    unprefixed one is the fallback.
    """
    candidates = [name]
    upper = level.upper()
    if name.startswith("SAPT ") and upper != "SAPT":
        candidates.insert(0, upper + name[len("SAPT"):])
    for key in candidates:
        if key in variables:
            try:
                return float(variables[key])
            except (TypeError, ValueError):
                continue
    return None


def _sapt_property_hook(level: str):
    """Build the ``property_hook`` that pulls the SAPT psivars out in kcal/mol."""

    def hook(wfn, molecule) -> dict[str, Any]:  # noqa: ANN001 - psi4 objects
        import psi4

        variables = dict(psi4.core.variables())
        try:
            variables.update(wfn.variables())
        except Exception:  # pragma: no cover - some wfns expose no variables
            pass

        out: dict[str, Any] = {}
        for key, name in {**SAPT_VARIABLES, **SAPT_EXTRA_VARIABLES}.items():
            value = _harvest(variables, name, level)
            if value is not None:
                out[key] = value * HARTREE_TO_KCAL
        out["sapt_variables_seen"] = sorted(k for k in variables if "SAPT" in k.upper())
        return out

    return hook


def _needs_metal_basis(geometry: str) -> bool:
    return any(
        symbol not in _JUN_CC_ELEMENTS
        for frag in parse_fragments(geometry)
        for symbol in frag.symbols
    )


def sapt0_interaction(
    dimer_geometry: str | DimerGeometry,
    basis: str | None = None,
    *,
    level: str = "sapt0",
    options: Mapping[str, Any] | None = None,
    label: str = "",
    use_cache: bool = True,
    presets: Sequence[Mapping[str, Any]] | None = None,
) -> SaptComponents:
    """Run a SAPT0 decomposition on a two-fragment geometry.

    Parameters
    ----------
    dimer_geometry
        Either a raw Psi4 two-fragment block or a :class:`DimerGeometry`.
    basis
        Defaults to :data:`DEFAULT_SAPT_BASIS` (jun-cc-pvdz), automatically
        swapped for :data:`DEFAULT_SAPT_METAL_BASIS` when a fragment contains an
        element the jun-cc family does not cover -- which is exactly the
        nickel case this module exists for.
    level
        The Psi4 SAPT method string.  ``'sapt0'`` is what is validated here;
        ``'ssapt0'`` (S^2-scaled exchange) and higher SAPT levels are passed
        through and harvested from their own prefixed psivars, untested.

    Raises
    ------
    SaptScopeError
        If the geometry is not two closed-shell fragments.  This check runs
        *before* anything is submitted -- see
        :func:`assert_closed_shell_fragments` for why it cannot be left to Psi4.
    """
    geometry = (
        dimer_geometry.geometry
        if isinstance(dimer_geometry, DimerGeometry)
        else dimer_geometry
    )
    assert_closed_shell_fragments(geometry)

    if basis is None:
        basis = DEFAULT_SAPT_METAL_BASIS if _needs_metal_basis(geometry) else DEFAULT_SAPT_BASIS

    spec = JobSpec(
        geometry=geometry,
        method=level,
        basis=basis,
        charge=0,
        multiplicity=1,
        reference="rhf",
        options={"freeze_core": True, **(dict(options) if options else {})},
        label=label or f"{level}-interaction",
    )

    result = run_energy(
        spec,
        use_cache=use_cache,
        property_hook=_sapt_property_hook(level),
        # SAPT monomers are closed-shell and converge from the default guess;
        # walking the full open-shell rescue ladder would only burn time.
        presets=list(presets) if presets is not None else list(SCF_PRESETS[:2]),
    )

    if not result.ok:
        return SaptComponents.failed(basis, level, result.error or "unknown failure", label)
    if "property_error" in result.properties:
        return SaptComponents.failed(
            basis, level,
            f"energy converged but psivar harvest failed: "
            f"{result.properties['property_error']}",
            label,
        )
    return SaptComponents.from_properties(
        result.properties, basis=basis, level=level, label=label,
        wall_seconds=result.wall_seconds,
    )


# --------------------------------------------------------------------------
# Counterpoise-corrected supermolecular binding: the independent cross-check
# --------------------------------------------------------------------------


@dataclass
class CounterpoiseBinding:
    """BSSE-corrected supermolecular binding energy, in kcal/mol.

    Independent of SAPT in every way that matters -- different theory
    (supermolecular DFT vs perturbation theory), different error modes -- so
    agreement between :attr:`binding` and a SAPT0 total is real evidence that
    both are describing the same physical interaction.  Disagreement beyond
    about 1-2 kcal/mol on a system like this means one of them is wrong.
    """

    method: str
    basis: str
    binding: float                 #: CP-corrected, kcal/mol
    raw_binding: float | None = None   #: uncorrected supermolecular, kcal/mol
    bsse: float | None = None      #: raw - CP, kcal/mol; positive = overbinding
    converged: bool = True
    error: str | None = None
    label: str = ""
    wall_seconds: float = 0.0
    #: Raw hartree totals from the n-body driver, for provenance.
    energies_hartree: dict[str, float] = field(default_factory=dict)

    def agrees_with(self, sapt: SaptComponents, tolerance: float = 2.0) -> bool:
        """Do the two independent estimates land within ``tolerance`` kcal/mol?"""
        if not (self.converged and sapt.converged):
            return False
        return abs(self.binding - sapt.total) <= tolerance

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        if not self.converged:
            return f"CP {self.method}/{self.basis}: FAILED ({self.error})"
        bsse = f"  BSSE {self.bsse:+.3f}" if self.bsse is not None else ""
        return (
            f"CP {self.method}/{self.basis}: {self.binding:+.3f} kcal/mol{bsse}"
        )

    @classmethod
    def failed(cls, method: str, basis: str, error: str, label: str = "") -> "CounterpoiseBinding":
        return cls(method, basis, float("nan"), converged=False, error=error, label=label)


def counterpoise_binding(
    dimer_geometry: str | DimerGeometry,
    *,
    method: str = DEFAULT_CP_METHOD,
    basis: str | None = None,
    options: Mapping[str, Any] | None = None,
    include_uncorrected: bool = True,
    label: str = "",
    use_cache: bool = True,
) -> CounterpoiseBinding:
    """Counterpoise-corrected supermolecular binding energy.

    Uses Psi4's n-body driver, ``psi4.energy(method, bsse_type='cp')``, which
    computes both monomers in the full dimer basis and subtracts.  With
    ``include_uncorrected`` the ``nocp`` value is requested in the same call --
    Psi4 reuses the dimer energy, so the extra cost is only the two
    monomer-in-monomer-basis SCFs -- and the difference is reported as the
    basis-set superposition error, which is the honest way to show how much of
    a raw binding energy was an artefact.

    Not routed through :func:`nimrod.psi4_driver.run_energy`, which has no
    ``bsse_type`` hook, but it uses the same on-disk cache and the same SCF
    preset ladder so the provenance stays uniform.
    """
    import psi4

    geometry = (
        dimer_geometry.geometry
        if isinstance(dimer_geometry, DimerGeometry)
        else dimer_geometry
    )
    fragments = assert_closed_shell_fragments(geometry)

    if basis is None:
        basis = DEFAULT_SAPT_METAL_BASIS if _needs_metal_basis(geometry) else DEFAULT_SAPT_BASIS

    lowered = method.lower()
    if any(lowered.endswith(suffix) for suffix in _UNAVAILABLE_DISPERSION):
        raise SaptScopeError(
            f"method {method!r} needs an external dispersion program, but the "
            f"'dftd3' Python module is not installed in this environment "
            f"(QCEngine raises \"Program s-dftd3 is registered with QCEngine, but "
            f"cannot be found\"). Use a functional with native dispersion "
            f"instead: {DEFAULT_CP_METHOD!r}, 'wb97x-v', 'wb97m-v' or 'b97m-v'."
        )

    bsse_types = ["cp", "nocp"] if include_uncorrected else ["cp"]
    spec = JobSpec(
        geometry=geometry,
        method=method,
        basis=basis,
        charge=sum(f.charge for f in fragments),
        multiplicity=1,
        options={
            **(dict(options) if options else {}),
            "__bsse__": ",".join(bsse_types),
        },
        label=label or "counterpoise",
    )

    if use_cache:
        cached = load_cached(spec)
        if cached is not None:
            return _cp_from_job_result(cached, method, basis, label)

    reference = spec.resolved_reference()
    started = time.time()
    last_error: str | None = None

    for preset in SCF_PRESETS[:2]:
        with clean_context():
            try:
                mol = psi4.geometry(
                    f"{geometry.strip()}\nsymmetry c1\nno_reorient\nno_com\n"
                )
                opts = {
                    "basis": basis,
                    "reference": reference,
                    **preset["options"],
                }
                opts.update({k: v for k, v in spec.options.items() if not k.startswith("__")})
                psi4.set_options(opts)

                psi4.energy(method, bsse_type=bsse_types, molecule=mol)
                variables = dict(psi4.core.variables())

                harvested = {
                    key: float(value)
                    for key, value in variables.items()
                    if ("INTERACTION ENERGY" in key or "N-BODY" in key)
                    and isinstance(value, (int, float))
                }
                cp = variables.get("CP-CORRECTED INTERACTION ENERGY")
                if cp is None:
                    raise RuntimeError(
                        "n-body driver returned but 'CP-CORRECTED INTERACTION "
                        f"ENERGY' is absent; saw {sorted(harvested)}"
                    )
                nocp = variables.get("NOCP-CORRECTED INTERACTION ENERGY")

                result = JobResult(
                    spec_fingerprint=spec.fingerprint(),
                    label=spec.label,
                    method=method,
                    basis=basis,
                    charge=spec.charge,
                    multiplicity=1,
                    reference=reference,
                    converged=True,
                    energy=float(cp),
                    preset=preset["label"],
                    wall_seconds=time.time() - started,
                    properties={
                        "cp_hartree": float(cp),
                        "nocp_hartree": float(nocp) if nocp is not None else None,
                        "energies_hartree": harvested,
                    },
                )
                store_cached(result)
                return _cp_from_job_result(result, method, basis, label)

            except Exception as exc:
                last_error = f"[{preset['label']}] {type(exc).__name__}: {str(exc)[:400]}"
                continue

    result = JobResult(
        spec_fingerprint=spec.fingerprint(),
        label=spec.label,
        method=method,
        basis=basis,
        charge=spec.charge,
        multiplicity=1,
        reference=reference,
        converged=False,
        wall_seconds=time.time() - started,
        error=last_error,
    )
    store_cached(result)
    return CounterpoiseBinding.failed(method, basis, last_error or "unknown failure", label)


def _cp_from_job_result(
    result: JobResult, method: str, basis: str, label: str
) -> CounterpoiseBinding:
    if not result.ok:
        return CounterpoiseBinding.failed(method, basis, result.error or "unknown", label)
    cp = float(result.properties["cp_hartree"]) * HARTREE_TO_KCAL
    nocp_raw = result.properties.get("nocp_hartree")
    nocp = float(nocp_raw) * HARTREE_TO_KCAL if nocp_raw is not None else None
    return CounterpoiseBinding(
        method=method,
        basis=basis,
        binding=cp,
        raw_binding=nocp,
        bsse=(cp - nocp) if nocp is not None else None,
        converged=True,
        label=label,
        wall_seconds=result.wall_seconds,
        energies_hartree=dict(result.properties.get("energies_hartree", {})),
    )


# --------------------------------------------------------------------------
# The degradation profile: what the open site becomes
# --------------------------------------------------------------------------


@dataclass
class CapturePoint:
    """Solvent capture at one point of the degradation coordinate."""

    coordinate: float                       #: Ni-N distance, angstrom
    metal_oxygen_distance: float
    closest_contact: float
    sapt: SaptComponents
    counterpoise: CounterpoiseBinding | None = None
    geometry_warning: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "metal_oxygen_distance": self.metal_oxygen_distance,
            "closest_contact": self.closest_contact,
            "sapt": self.sapt.to_json(),
            "counterpoise": self.counterpoise.to_json() if self.counterpoise else None,
            "geometry_warning": self.geometry_warning,
        }


@dataclass
class CaptureProfile:
    """SAPT decomposition of solvent capture along the degradation coordinate."""

    label: str
    basis: str
    level: str
    points: list[CapturePoint] = field(default_factory=list)

    @property
    def distances(self) -> list[float]:
        return [p.coordinate for p in self.points]

    def component_series(self) -> dict[str, list[float | None]]:
        """Components keyed for :func:`nimrod.plots.plot_sapt_components`."""
        keys = ("electrostatics", "exchange", "induction", "dispersion", "total")
        return {
            key: [
                getattr(p.sapt, key) if p.sapt.converged else None
                for p in self.points
            ]
            for key in keys
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "label": self.label,
            "basis": self.basis,
            "level": self.level,
            "points": [p.to_json() for p in self.points],
        }, indent=2, default=str))
        return path


def capture_profile(
    series: Iterable[tuple[float, Structure]],
    metal_index: int,
    *,
    leaving_index: int | None = None,
    basis: str | None = None,
    level: str = "sapt0",
    metal_oxygen_distance: float = DEFAULT_NI_O_DISTANCE,
    charge: int = 0,
    multiplicity: int = 1,
    cross_check: bool = False,
    cross_check_method: str = DEFAULT_CP_METHOD,
    label: str = "solvent-capture",
    use_cache: bool = True,
) -> CaptureProfile:
    """Decompose H2O binding at every point of a ligand-loss coordinate.

    ``series`` is a sequence of ``(coordinate, catalyst_structure)`` pairs, i.e.
    exactly what :func:`nimrod.geometry.degradation_series` returns, or the
    relaxed geometries from :mod:`nimrod.degradation`.

    The structures must be the **bare catalyst** (closed shell).  The full
    sensor assembly carries the colour-centre radical and is a doublet, so it is
    not a legal SAPT monomer; :func:`assert_closed_shell_fragments` will refuse
    it with that explanation.

    At the intact end of the coordinate the water is being pushed against a
    coordinatively saturated square-planar complex and the decomposition should
    come out exchange-dominated and weakly bound.  As the arm dissociates,
    induction should grow -- that growth *is* the open coordination site.
    """
    points: list[CapturePoint] = []
    resolved_basis = basis

    for coordinate, structure in series:
        dimer = catalyst_water_dimer(
            structure,
            metal_index,
            leaving_index=leaving_index,
            metal_oxygen_distance=metal_oxygen_distance,
            charge=charge,
            multiplicity=multiplicity,
        )
        if resolved_basis is None:
            resolved_basis = (
                DEFAULT_SAPT_METAL_BASIS
                if _needs_metal_basis(dimer.geometry)
                else DEFAULT_SAPT_BASIS
            )

        sapt = sapt0_interaction(
            dimer,
            resolved_basis,
            level=level,
            label=f"{label}-{coordinate:.2f}",
            use_cache=use_cache,
        )
        cp = None
        if cross_check:
            cp = counterpoise_binding(
                dimer,
                method=cross_check_method,
                basis=resolved_basis,
                label=f"{label}-cp-{coordinate:.2f}",
                use_cache=use_cache,
            )

        points.append(CapturePoint(
            coordinate=float(coordinate),
            metal_oxygen_distance=dimer.metal_oxygen_distance,
            closest_contact=dimer.closest_contact,
            sapt=sapt,
            counterpoise=cp,
            geometry_warning=dimer.clash_warning(),
        ))

    return CaptureProfile(
        label=label,
        basis=resolved_basis or DEFAULT_SAPT_BASIS,
        level=level,
        points=points,
    )
