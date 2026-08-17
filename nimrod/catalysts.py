"""Real Ni(II) salen, and the pentacene colour centre that reports on it.

This module upgrades both halves of the NiMROD sensor from the truncated models
in :mod:`nimrod.geometry` to the compounds the project actually claims to be
studying.

``ni_salen``
    N,N'-bis(salicylidene)ethylenediaminato-nickel(II), NiC16H14N2O2 — the real
    literature compound, not the 19-atom bis(iminoenolato) stand-in.  The free
    ligand salenH2 is C16H16N2O2; double deprotonation of the two phenols gives
    salen(2-) = C16H14N2O2, which balances Ni(2+) to a neutral, closed-shell
    (square-planar d8, S = 0) complex.

    The chemically important difference from the truncated model is *denticity*.
    The truncated catalyst is two independent bidentate arms, so its donors sit
    trans to each other (N trans N, O trans O).  Salen is a single tetradentate
    ligand whose four donors are threaded onto one chain

        O(1)-C(aryl A)...C(aryl A)-CH=N(1)-CH2-CH2-N(2)=CH-C(aryl B)...C(aryl B)-O(2)

    and a chain cannot span trans positions.  Every donor is therefore *cis* to
    its neighbours in the chain: the two O are cis, the two N are cis, and three
    fused chelate rings close around the metal —

        Ni-O(1)-C(ar)-C(ar)-CH=N(1)      six-membered, salicylaldiminato
        Ni-N(1)-CH2-CH2-N(2)             five-membered, diamine bridge
        Ni-N(2)=CH-C(ar)-C(ar)-O(2)      six-membered, salicylaldiminato

    Getting that topology wrong (building the trans isomer) yields a structure
    that looks perfectly reasonable in a viewer and is a different molecule, so
    :mod:`tests.test_catalysts` asserts the cis relationships explicitly.

``pentacene_sensor``
    Pentacene C22H14 carrying an sp3 defect whose substituent is the whole
    Ni(salen) unit, bonded through position 5 of one salicylidene ring — the
    standard synthetic handle on salicylaldehyde (5-bromo-, 5-nitro-,
    5-tert-butylsalicylaldehyde are all commercial).  Pentacene replaces pyrene
    because it has a far richer low-lying spin manifold (T1 ~ 0.86 eV, S1 ~ 1.83
    eV, appreciable diradical character in the ground state) and a strong,
    easily monitored visible absorption, so both sensor readouts have more to
    work with.

    The defect carbon keeps its hydrogen and becomes sp3.  That both breaks the
    conjugation locally, creating the colour centre, and leaves an odd electron
    count: NiC38H27N2O2 is a ground-state doublet.  An even electron count means
    the defect was not formed.

``open_salen_arm``
    The degradation coordinate: one salicylidene arm swung off the metal by
    rotating the two ethylenediamine torsions.  Salen cannot be opened by any
    single rigid rotation.  The generic :func:`~nimrod.geometry.chelate_hinge`
    assumes the donor sits in one chelate ring; a salen nitrogen sits in two,
    and the hinge it finds through the five-membered ring reaches only 4.37 A.
    Hinging further along the bridge reaches far enough, but *only* because it
    rotates about the chelate-plane normal, which bends the sp3 bridge carbon it
    pivots on: at Ni-N = 4.5 A that angle opens to 174 deg and the two methylene
    hydrogens collide at 1.49 A.  Bond lengths stay perfect throughout, which is
    exactly why such a geometry survives a length-only check.

    Rotating about the two bridge *bonds* instead preserves every bond angle as
    well as every bond length — the atoms defining each angle either all move or
    all stay — and two torsions together span 1.85 to 5.29 A, comfortably past
    dissociation.  See the module tests, which pin both the reach of the generic
    hinge and the angles the torsion route preserves.

As in :mod:`nimrod.geometry`, everything here is a *starting guess* with correct
connectivity and no clashes, meant to be relaxed by
:func:`nimrod.psi4_driver.run_optimize` before any number is quoted from it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .geometry import (
    CH_AROMATIC,
    CH_SP3,
    TETRAHEDRAL_OUT_OF_PLANE,
    Structure,
    _rotate_about,
    _unit,
    build_pah,
    occ_radical,
    relieve_torsion,
)

# --------------------------------------------------------------------------
# Ni(salen) internal coordinates
#
# Bond lengths are the standard values for this class of complex; the angles
# are the crystallographic ones for square-planar Ni(salen).  The three bite
# angles are what fix the cis topology: 94 + 86 + 94 = 274 leaves 86 deg for
# O-Ni-O, so the four donors close a full 360 deg around the metal in one
# plane.  Two trans donors would need a 180 deg span, which the ligand chain
# is far too short to reach.
# --------------------------------------------------------------------------

NI_O_SALEN = 1.85           # Ni-O(phenolate)
NI_N_SALEN = 1.85           # Ni-N(imine)
BITE_ON = 94.0              # O-Ni-N, the six-membered chelate ring
BITE_NN = 86.0              # N-Ni-N, the five-membered diamine ring

CC_ARYL = 1.40              # benzene ring
CO_PHENOLATE = 1.31         # C(aryl)-O(-)
CN_IMINE = 1.30             # C=N
C_ARYL_IMINE = 1.45         # C(aryl)-CH=N
N_CH2 = 1.47                # N-CH2 of the diamine bridge
CH2_CH2 = 1.52              # the bridge C-C

#: Labels of one salicylidene arm, in the order the builder emits them.
#: Ring carbons use salicylaldehyde numbering: C1 carries the imine, C2 carries
#: the phenolate oxygen, and C5 — para to C2 — is the substitution handle.
_ARM_ORDER = (
    "O", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "N", "H3", "H4", "H5", "H6", "H7",
)
_ARM_SYMBOLS = {
    "O": "O", "N": "N",
    **{f"C{k}": "C" for k in range(1, 8)},
    **{f"H{k}": "H" for k in range(3, 8)},
}


def _polar(radius: float, degrees: float) -> np.ndarray:
    """In-plane point at ``radius`` and ``degrees`` from the origin."""
    theta = math.radians(degrees)
    return np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0])


def _salicylaldiminato(imine_angle: float) -> dict[str, np.ndarray]:
    """One flat, internally ideal salicylidene-imine arm, keyed by label.

    The benzene ring is a regular hexagon of side :data:`CC_ARYL`; the
    phenolate C-O and the aryl-imine C-C both leave the ring radially, which
    makes every exocyclic ring angle exactly 120 deg.  The only free parameter
    is ``imine_angle`` = C1-C7=N, which sets how far the imine nitrogen swings
    back towards the oxygen and therefore the O...N donor separation — that is
    what :func:`_solve_imine_angle` tunes to make the chelate ring close.
    """
    ring_angle = {"C2": 0.0, "C1": 60.0, "C6": 120.0, "C5": 180.0, "C4": 240.0, "C3": 300.0}
    atoms = {label: _polar(CC_ARYL, deg) for label, deg in ring_angle.items()}

    atoms["O"] = atoms["C2"] + CO_PHENOLATE * _polar(1.0, ring_angle["C2"])
    atoms["C7"] = atoms["C1"] + C_ARYL_IMINE * _polar(1.0, ring_angle["C1"])
    # Measured from the C7->C1 direction, rotating towards the oxygen so that
    # the nitrogen ends up on the same face of the arm as the phenolate — the
    # cis arrangement that lets one arm chelate the metal at all.
    atoms["N"] = atoms["C7"] + CN_IMINE * _polar(1.0, ring_angle["C1"] + 180.0 + imine_angle)

    for label in ("C3", "C4", "C5", "C6"):
        atoms["H" + label[1]] = atoms[label] + CH_AROMATIC * _polar(1.0, ring_angle[label])

    # sp2 imine hydrogen: opposite the bisector of the other two bonds at C7.
    bisector = _unit(_unit(atoms["C1"] - atoms["C7"]) + _unit(atoms["N"] - atoms["C7"]))
    atoms["H7"] = atoms["C7"] - CH_AROMATIC * bisector

    # As built, the aryl ring lies on the same side of the O...N line as the
    # metal will, which would drop the benzene ring on top of the nickel.
    # Reflecting the (planar) arm in y puts the ring on the far side.  A planar
    # fragment reflected in its own plane is the same molecule, so this costs
    # no chemistry.
    return {label: np.array([p[0], -p[1], p[2]]) for label, p in atoms.items()}


def _solve_imine_angle(target_on: float, bracket: tuple[float, float] = (110.0, 160.0)) -> float:
    """Imine angle C1-C7=N that puts the two donors ``target_on`` apart.

    A salicylidene arm built with *all* internal angles ideal spans only 2.58 A
    between its O and N, but a 94 deg bite at 1.85 A needs 2.71 A: the six-
    membered chelate ring cannot close with an undistorted ligand, which is why
    real salen complexes are measurably strained.  Rather than smear the
    mismatch over every bond by fudging bond lengths, park it in the softest
    coordinate available — the wide sp2 angle at the imine carbon — and let the
    optimiser redistribute it later.  It comes out at 130.5 deg against a
    literature 126 deg, a 4.5 deg distortion, with every bond length exact.
    """
    def span(angle: float) -> float:
        arm = _salicylaldiminato(angle)
        return float(np.linalg.norm(arm["O"] - arm["N"]))

    lo, hi = bracket
    if not span(lo) <= target_on <= span(hi):
        raise ValueError(
            f"O...N = {target_on:.3f} A is outside the reach of the arm "
            f"({span(lo):.3f}-{span(hi):.3f} A over {bracket} deg)"
        )
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if span(mid) < target_on:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _place_in_plane(
    fragment: dict[str, np.ndarray],
    first: str,
    second: str,
    first_target: np.ndarray,
    second_target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rigidly move a flat fragment so two of its atoms hit two target points.

    Rotation about z plus a translation — three degrees of freedom against four
    constraints, so this is only exact when the fragment's internal
    ``first``-``second`` distance already matches the target separation.  It
    does, by construction: :func:`_solve_imine_angle` made it so.  The atom
    ``first`` is placed exactly and the fragment is rotated to line up
    ``second``; any residual mismatch would show up as a wrong Ni-N distance,
    which the tests check.
    """
    local = fragment[second] - fragment[first]
    wanted = second_target - first_target
    delta = math.atan2(wanted[1], wanted[0]) - math.atan2(local[1], local[0])
    cos, sin = math.cos(delta), math.sin(delta)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return {
        label: rotation @ (point - fragment[first]) + first_target
        for label, point in fragment.items()
    }


def ni_salen(name: str = "ni-salen") -> tuple[Structure, dict[str, int]]:
    """N,N'-bis(salicylidene)ethylenediaminato-nickel(II), NiC16H14N2O2.

    Returns the structure and an index map.  The four donors are labelled
    ``Ni``, ``O1``, ``N1``, ``O2``, ``N2``; ring atoms carry an arm suffix, so
    arm 1 (the one holding O1/N1) is ``C1a``...``C7a`` with hydrogens
    ``H3a``...``H7a``, and arm 2 is ``...b``.  ``C5a``/``H5a`` are therefore the
    aryl carbon para to phenolate O1 and its hydrogen — position 5 of
    salicylaldehyde, where the colour centre gets bolted on.  The two diamine
    bridge carbons are ``Cbr1`` (on N1) and ``Cbr2`` (on N2).

    The complex is built flat.  Real Ni(salen) is very slightly stepped, with
    the two salicylidene planes folded a few degrees out of the NiN2O2 plane,
    but that is a soft mode and a planar guess relaxes into it.

    What the construction actually delivers, against crystallographic Ni(salen):
    every bond length exact by fiat (Ni-O = Ni-N = 1.850, C-O 1.310, C=N 1.300,
    N-CH2 1.470, CH2-CH2 1.520 A) and every bite angle exact (94.0 / 86.0 /
    94.0 / 86.0, summing to 360).  The strain of closing the six-membered ring
    then lands in the angles: C(aryl)-CH=N 130.5 deg against 126, Ni-O-C 132.9
    against 128, Ni-N=C 122.6 against 127.  All three are within 5 deg and all
    three are soft, so this is a starting guess a geometry optimisation will
    move a few degrees, not a structure to quote angles from.
    """
    # Donor directions.  Put the C2 axis of the ligand along +y, so the two
    # nitrogens straddle it at half the five-ring bite and each oxygen sits one
    # six-ring bite further round.  The two O then necessarily land next to
    # each other on the -y side: cis, as a tetradentate chain demands.
    angle_n1 = 90.0 - BITE_NN / 2.0
    angle_n2 = 90.0 + BITE_NN / 2.0
    angle_o1 = angle_n1 - BITE_ON
    angle_o2 = angle_n2 + BITE_ON

    target_on = 2.0 * NI_O_SALEN * math.sin(math.radians(BITE_ON) / 2.0)
    arm = _salicylaldiminato(_solve_imine_angle(target_on))

    arm_a = _place_in_plane(
        arm, "O", "N", _polar(NI_O_SALEN, angle_o1), _polar(NI_N_SALEN, angle_n1)
    )
    # Arm 2 is arm 1 reflected in the C2 axis (x -> -x).  Fitting it
    # independently would work equally well; mirroring guarantees the two arms
    # are the same molecule to machine precision, which makes any asymmetry in
    # a later optimisation genuinely physical rather than a build artefact.
    arm_b = {label: np.array([-p[0], p[1], p[2]]) for label, p in arm_a.items()}

    symbols: list[str] = ["Ni"]
    coords: list[np.ndarray] = [np.zeros(3)]
    index: dict[str, int] = {"Ni": 0}

    for suffix, placed in (("a", arm_a), ("b", arm_b)):
        arm_number = "1" if suffix == "a" else "2"
        for label in _ARM_ORDER:
            # Donors are keyed by arm number (O1/N1) to match the rest of the
            # project; ring atoms by arm letter (C5a) to match the way
            # substituted salicylaldehydes are named.
            key = f"{label}{arm_number}" if label in ("O", "N") else f"{label}{suffix}"
            index[key] = len(symbols)
            symbols.append(_ARM_SYMBOLS[label])
            coords.append(placed[label])

    # --- ethylenediamine bridge -------------------------------------------
    # N1-CH2-CH2-N2 closing the five-membered ring.  Both carbons sit on the
    # mirror plane's normal at +/- half a C-C bond, which is the only planar
    # placement consistent with the molecular C2 axis; their distance from the
    # N...N line then follows from the N-CH2 bond length.
    n1 = coords[index["N1"]]
    n2 = coords[index["N2"]]
    along = _unit(n2 - n1)
    outward = _unit(np.cross(np.array([0.0, 0.0, 1.0]), along))
    if float(np.dot(outward, 0.5 * (n1 + n2))) < 0.0:
        outward = -outward                      # away from the metal
    reach = 0.5 * float(np.linalg.norm(n2 - n1)) - 0.5 * CH2_CH2
    rise = math.sqrt(N_CH2**2 - reach**2)
    bridge = {
        "Cbr1": n1 + reach * along + rise * outward,
        "Cbr2": n2 - reach * along + rise * outward,
    }

    normal = np.array([0.0, 0.0, 1.0])
    for key, position in bridge.items():
        neighbour_n = n1 if key == "Cbr1" else n2
        neighbour_c = bridge["Cbr2"] if key == "Cbr1" else bridge["Cbr1"]
        index[key] = len(symbols)
        symbols.append("C")
        coords.append(position)
        # The two methylene hydrogens straddle the ligand plane: this carbon is
        # sp3, and putting both H in plane would make it planar and wrong.
        bisector = _unit(position - 0.5 * (neighbour_n + neighbour_c))
        for tag, sign in (("u", 1.0), ("d", -1.0)):
            direction = _unit(
                math.cos(TETRAHEDRAL_OUT_OF_PLANE) * bisector
                + sign * math.sin(TETRAHEDRAL_OUT_OF_PLANE) * normal
            )
            index[f"H{key[1:]}{tag}"] = len(symbols)
            symbols.append("H")
            coords.append(position + CH_SP3 * direction)

    return Structure(symbols, np.array(coords), name), index


# --------------------------------------------------------------------------
# The pentacene colour centre
# --------------------------------------------------------------------------

#: Linear polyacenes as hexagon-centre lattice coordinates, for
#: :func:`nimrod.geometry.build_pah`.
ACENE_RINGS = {n: tuple((i, 0) for i in range(n)) for n in range(1, 8)}


def pentacene() -> Structure:
    """Pentacene, C22H14 — five linearly fused rings, the colour-centre host."""
    return build_pah(ACENE_RINGS[5], "pentacene")


def tetracene() -> Structure:
    """Tetracene, C18H12 — the next acene down, for host-size comparisons."""
    return build_pah(ACENE_RINGS[4], "tetracene")


def meso_site(host: Structure) -> int:
    """The CH carbon of an acene closest to the molecular centre.

    For pentacene these are positions 6 and 13, the central-ring CH carbons.
    They carry the largest frontier-orbital amplitude and are where pentacene
    actually reacts — 6,13-substitution is the standard way to functionalise it
    — so that is where the sp3 defect belongs.  Defaulting to whichever CH
    carbon the builder happened to emit first would put the defect on a
    terminal ring, where the perturbation to the pi system is much weaker.
    """
    centroid = host.coords[[i for i, s in enumerate(host.symbols) if s == "C"]].mean(axis=0)
    candidates = [
        i
        for i, symbol in enumerate(host.symbols)
        if symbol == "C" and len(host.hydrogens_on(i)) == 1
    ]
    if not candidates:
        raise ValueError("host has no CH carbon to functionalise")
    return min(candidates, key=lambda i: float(np.linalg.norm(host.coords[i] - centroid)))


def pentacene_occ(name: str = "pentacene-occ") -> tuple[Structure, dict[str, int]]:
    """The free colour centre: pentacene with a phenyl sp3 defect, C28H19.

    This is the reference against which the catalyst-bound sensor's optical
    shift is measured, so it must differ from the sensor *only* by the
    substituent — same host, same defect position, same doublet ground state.
    """
    host = pentacene()
    return occ_radical(host=host, site=meso_site(host), name=name)


def _non_bonded_pairs(struct: Structure) -> np.ndarray:
    """Index pairs that are neither bonded (1-2) nor geminal (1-3).

    These are the only pairs whose separation a torsion is free to change, so
    they are the only ones worth scoring when judging a starting geometry.  A
    1-2 or 1-3 distance is fixed by a bond length or a bond angle; including
    them just pins the objective at the shortest bond in the molecule.
    """
    neighbours = [set(struct.neighbours(i)) for i in range(len(struct))]
    return np.array(
        [
            (i, j)
            for i in range(len(struct))
            for j in range(i + 1, len(struct))
            if j not in neighbours[i] and not (neighbours[i] & neighbours[j])
        ]
    )


def relieve_contacts(
    struct: Structure,
    axis_a: int,
    axis_b: int,
    moving: Sequence[int],
    step_degrees: float = 5.0,
) -> Structure:
    """Torsion scan that scores on the closest *non-bonded* contact.

    :func:`nimrod.geometry.relieve_torsion` maximises
    :meth:`Structure.min_interatomic_distance`, which is a minimum over *all*
    pairs — bonded ones included.  That works when the alternative is a
    catastrophe: folding the truncated catalyst onto pyrene gave 0.71 A
    contacts, far below any bond length, so the bad torsions stood out.  It
    stops working as soon as the worst contact is merely bad rather than
    absurd.  On this sensor every torsion angle bottoms out at 1.08 A — the
    aromatic C-H bond length — so the objective is flat, the scan cannot tell
    a 1.08 A H...H collision from a C-H bond, and ``relieve_torsion`` returns
    its input unchanged while the closest real contact ranges from 1.08 to
    2.27 A across the scan.

    Scoring only pairs that are neither bonded nor geminal fixes that.  Geminal
    (1-3) pairs have to be excluded too, not just bonded ones: the methylene
    H...H distance is 1.78 A and is fixed by the bond angles, so leaving it in
    would re-flatten the objective at 1.78 A.

    Note what this does *not* buy.  The objective is still bounded above by the
    closest contact inside either rigid fragment, which no rotation can change:
    on this sensor that is the 2.267 A H...H across the eclipsed ethylenediamine
    bridge.  Every torsion whose pentacene-to-catalyst contact clears 2.267 A
    therefore scores identically — 18 of the 72 angles scanned here do — and the
    first of them is kept.  That is good enough (the tie is between geometries
    that are all clash-free, and the kept one comes within 0.02 A of the best
    available cross-fragment contact) but it is a ceiling, not a fix: this
    raises the flat floor from 1.08 A to 2.267 A rather than removing it.
    """
    pairs = _non_bonded_pairs(struct)
    if len(pairs) == 0:
        return struct

    def closest(coords: np.ndarray) -> float:
        return float(
            np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1).min()
        )

    origin = struct.coords[axis_a]
    axis = struct.coords[axis_b] - origin
    indices = list(moving)

    best, best_contact = struct, closest(struct.coords)
    for degrees in np.arange(step_degrees, 360.0, step_degrees):
        candidate = struct.copy()
        candidate.coords[indices] = _rotate_about(
            struct.coords[indices], origin, axis, math.radians(float(degrees))
        )
        contact = closest(candidate.coords)
        if contact > best_contact:
            best_contact, best = contact, candidate
    return best


def pentacene_sensor(name: str = "pentacene-ni-salen",
                     site: str = "C5") -> tuple[Structure, dict[str, int]]:
    """The full sensor: pentacene colour centre carrying a real Ni(salen).

    The nickel complex is the sp3 defect's substituent, bonded through position
    5 of one salicylidene ring — the handle that 5-substituted salicylaldehydes
    actually give you.  Returns ``NiC38H27N2O2``, 70 atoms, a ground-state
    doublet.

    Be aware of what that attachment point costs magnetically.  C5 is *para* to
    the phenolate oxygen, so the defect looks at the metal across the full
    width of the benzene ring: Ni...sp3 is 7.0 A here, against 8.75 A for the
    phenylene-tethered pyrene construct (which came out magnetically dead at
    J ~ 0.4 cm^-1) and 4.5 A for the spacer-free variant that recovered the
    coupling.  This construct buys synthetic realism at the price of most of
    that recovery, so the optical channel is expected to carry the signal and
    the exchange channel needs checking rather than assuming.  Substituting at
    C3 instead — ortho to the oxygen, one bond from the chelate ring — is the
    obvious lever if the coupling proves too weak.

    The index map carries the four donors through the join, plus
    ``sp3_carbon`` (the defect), ``ipso_carbon`` (C5 of arm 1, the aryl carbon
    it bonds to) and the ``N_labile``/``O_labile`` aliases the degradation
    machinery uses.  The labile arm is arm 2, across the metal from the
    reporting defect, so the signal travels through the metal rather than
    straight down the bond.
    """
    host = pentacene()
    catalyst, catalyst_index = ni_salen()

    joined, info = occ_radical(
        host=host,
        site=meso_site(host),
        aryl=catalyst,
        aryl_attach=catalyst_index[f"{site}a"],
        aryl_hydrogen=catalyst_index[f"H{site[1:]}a"],
        name=name,
        track_aryl={
            key: catalyst_index[key]
            for key in ("Ni", "O1", "N1", "O2", "N2", "C5b", "H5b", "Cbr1", "Cbr2",
                        "C7a", "C7b", "C2a", "C2b")
        },
    )

    # The whole nickel complex hangs off a single bond whose torsion attach()
    # says nothing about.  The standard guard runs first, then the contact-
    # aware scan picks among the angles it cannot distinguish; see
    # relieve_contacts for why one is not enough.  Measured on this construct:
    # as attached 1.81 A closest contact, after relieve_contacts 2.27 A, and
    # the worst torsion available is 1.08 A.
    rotor = range(info["n_host_atoms"], len(joined))
    joined = relieve_torsion(
        joined, axis_a=info["sp3_carbon"], axis_b=info["ipso_carbon"], moving=rotor
    )
    joined = relieve_contacts(
        joined, axis_a=info["sp3_carbon"], axis_b=info["ipso_carbon"], moving=rotor
    )

    info["N_labile"] = info["N2"]
    info["O_labile"] = info["O2"]
    info["N_spectator"] = info["N1"]
    info["O_spectator"] = info["O1"]
    return joined, info


# --------------------------------------------------------------------------
# Opening a salen arm
# --------------------------------------------------------------------------

#: Closest non-bonded contact (A) beyond which an opened geometry is considered
#: as good as the intact one, so the search stops trying to improve it.  The
#: intact complex already carries a 2.267 A H...H across the eclipsed
#: ethylenediamine bridge, and no torsion can open that, so 2.2 A is the most
#: any candidate can be asked for.  Capping matters: an uncapped "take the
#: largest closest contact" rule picks whichever torsion branch happens to win
#: at each scan point independently, and the resulting scan is discontinuous —
#: measured on this sensor, Ni-O of the departing arm went 5.05, 4.31, 6.40 A
#: over three consecutive points.  Capped, the tie-break is least motion from
#: the previous point, and the same three come out 5.05, 5.86, 6.40.
CONTACT_TARGET = 2.2


def _beyond(
    struct: Structure, axis_a: int, axis_b: int, blocked: set[int]
) -> list[int]:
    """Atoms reachable from ``axis_b`` without passing through ``axis_a`` or
    anything in ``blocked``.  ``axis_b`` itself is excluded.

    This is the set that a torsion about the ``axis_a``-``axis_b`` bond rotates.
    Blocking the metal is what stops the walk going round the chelate ring and
    picking up the whole molecule, which would make the torsion a no-op.
    """
    seen = set(blocked) | {axis_a, axis_b}
    out: list[int] = []
    stack = [j for j in struct.neighbours(axis_b) if j not in seen]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        out.append(current)
        stack.extend(j for j in struct.neighbours(current) if j not in seen)
    return sorted(out)


def salen_arm_torsions(
    struct: Structure, anchor: int, donor: int
) -> list[tuple[int, int, list[int]]]:
    """The two ethylenediamine torsions that swing one salicylidene arm off the
    metal, outermost first.

    Each entry is ``(axis_a, axis_b, moving)``: rotate ``moving`` about the
    ``axis_a``-``axis_b`` bond.  For a donor N2 the bridge is
    N1-CH2(far)-CH2(near)-N2 and the two torsions are about N1-CH2(far) and
    CH2(far)-CH2(near).

    Why torsions rather than a hinge.  Rotating about a *bond* leaves every bond
    angle in the molecule untouched, because each angle's three atoms either all
    move or all stay: the two axis atoms are fixed points of the rotation, so an
    angle centred on one of them keeps both its arms.  Rotating about anything
    else through a pivot atom does not — the pivot's own bond angles open and
    close freely, and only the *lengths* survive.  That is the trap this
    replaced: hinging about the chelate-plane normal reproduces every bond
    length to 1e-15 while bending the sp3 bridge carbon from 110 to 174 deg and
    driving its two methylene hydrogens to 1.49 A apart.

    The moving sets exclude the metal, so the spectator arm keeps both of its
    Ni bonds exactly.  They *include* everything bolted to the departing arm,
    which for the sensor's arm 1 means the whole pentacene: salen is one chain,
    a donor cannot leave on its own, and the mode really is tetradentate ->
    bidentate.
    """
    if struct.symbols[donor] != "N":
        raise ValueError(
            f"salen_arm_torsions expects a nitrogen donor, got {struct.symbols[donor]}"
        )
    if donor not in struct.neighbours(anchor):
        raise ValueError("donor is not bonded to the metal")

    # The bridge methylene is the donor's sp3 carbon neighbour: two hydrogens,
    # against one on the imine carbon.  Graph-derived, so it survives every join
    # the sensor is built from.
    near = [
        j
        for j in struct.neighbours(donor)
        if struct.symbols[j] == "C" and len(struct.hydrogens_on(j)) == 2
    ]
    if len(near) != 1:
        raise ValueError(f"expected one diamine methylene on atom {donor}, found {near}")
    near_c = near[0]
    far = [
        j
        for j in struct.neighbours(near_c)
        if struct.symbols[j] == "C" and j != donor and len(struct.hydrogens_on(j)) == 2
    ]
    if len(far) != 1:
        raise ValueError(f"atom {near_c} does not look like an ethylenediamine bridge")
    far_c = far[0]
    other = [j for j in struct.neighbours(far_c) if struct.symbols[j] == "N"]
    if len(other) != 1:
        raise ValueError(f"bridge carbon {far_c} does not carry exactly one nitrogen")

    return [
        (other[0], far_c, _beyond(struct, other[0], far_c, {anchor})),
        (far_c, near_c, _beyond(struct, far_c, near_c, {anchor})),
    ]


def _torsion_roots(
    point: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    anchor_pos: np.ndarray,
    target: float,
) -> tuple[list[float], float, float]:
    """Angles rotating ``point`` about ``axis`` to sit ``target`` from the anchor.

    Returns ``(roots, reach_lo, reach_hi)``.  Rotating a point about an axis
    traces a circle, so its squared distance to any fixed anchor is exactly
    ``A + R cos(theta - phi)`` — one cosine, no more.  Solving that in closed
    form beats bracketing a numerical root: it cannot miss a solution that is
    tangent rather than crossing, which is precisely the case at the first scan
    point, where the intact complex already sits at the target distance and it
    is the *minimum* of the sweep.
    """
    k = _unit(axis)
    u = point - origin
    u_par = float(u @ k) * k
    u_perp = u - u_par
    w = origin - anchor_pos

    a = float(np.dot(u_par + w, u_par + w) + np.dot(u_perp, u_perp))
    b = 2.0 * float(np.dot(u_perp, w))
    c = 2.0 * float(np.dot(np.cross(k, u_perp), w))
    r = math.hypot(b, c)

    reach_hi = math.sqrt(max(a + r, 0.0))
    reach_lo = math.sqrt(max(a - r, 0.0))
    if r < 1e-12:
        return [], reach_lo, reach_hi

    cosine = (target * target - a) / r
    if not -1.0 <= cosine <= 1.0:
        return [], reach_lo, reach_hi
    phi = math.atan2(c, b)
    delta = math.acos(cosine)
    return [phi + delta, phi - delta], reach_lo, reach_hi


def _open_once(
    struct: Structure,
    anchor: int,
    donor: int,
    target_distance: float,
    torsions: list[tuple[int, int, list[int]]],
    seed: tuple[float, float],
    pairs: np.ndarray,
    outer_step: float,
) -> tuple[Structure, tuple[float, float]]:
    """One scan point: scan the outer torsion, solve the inner one exactly."""
    (outer_a, outer_b, outer_moving), (inner_a, inner_b, inner_moving) = torsions
    anchor_pos = struct.coords[anchor]
    outer_origin = struct.coords[outer_a]
    outer_axis = struct.coords[outer_b] - struct.coords[outer_a]

    best: Structure | None = None
    best_score: tuple[float, float] | None = None
    best_angles = (0.0, 0.0)
    reach_hi = -math.inf
    reach_lo = math.inf

    for degrees in np.arange(0.0, 360.0, outer_step):
        outer = math.radians(float(degrees))
        stage = struct.copy()
        stage.coords[outer_moving] = _rotate_about(
            struct.coords[outer_moving], outer_origin, outer_axis, outer
        )
        # The inner axis rides on the outer rotation, so read it off the
        # intermediate geometry rather than the input one.
        origin = stage.coords[inner_a]
        axis = stage.coords[inner_b] - stage.coords[inner_a]
        roots, lo, hi = _torsion_roots(
            stage.coords[donor], origin, axis, anchor_pos, target_distance
        )
        reach_hi = max(reach_hi, hi)
        reach_lo = min(reach_lo, lo)
        for inner in roots:
            candidate = stage.copy()
            candidate.coords[inner_moving] = _rotate_about(
                stage.coords[inner_moving], origin, axis, inner
            )
            if abs(candidate.distance(anchor, donor) - target_distance) > 1e-6:
                continue
            contact = float(
                np.linalg.norm(
                    candidate.coords[pairs[:, 0]] - candidate.coords[pairs[:, 1]],
                    axis=1,
                ).min()
            )
            walk = abs(math.remainder(outer - seed[0], 2.0 * math.pi)) + abs(
                math.remainder(inner - seed[1], 2.0 * math.pi)
            )
            score = (min(contact, CONTACT_TARGET), -walk)
            if best_score is None or score > best_score:
                best_score, best, best_angles = score, candidate, (outer, inner)

    if best is None:
        raise ValueError(
            f"cannot open the arm to {target_distance:.2f} A; the ethylenediamine "
            f"torsions only reach {reach_lo:.2f}-{reach_hi:.2f} A"
        )
    return best, best_angles


def open_salen_arm(
    struct: Structure,
    anchor: int,
    donor: int,
    target_distance: float,
    *,
    torsions: list[tuple[int, int, list[int]]] | None = None,
    outer_step: float = 5.0,
) -> Structure:
    """Swing one salicylidene arm off the metal to a given Ni-N separation.

    Both ethylenediamine torsions are used, so every bond length *and* every
    bond angle in the molecule is preserved exactly (to ~1e-15 A and 1e-13 deg);
    only the two bridge dihedrals change.  Between them they span Ni-N from
    1.85 to 5.29 A on this complex, which covers the whole degradation scan.

    Among the solutions that hit the requested distance, the one with the
    largest closest non-bonded contact wins, capped at :data:`CONTACT_TARGET`;
    ties go to the least rotation.  Measured on the sensor, the closest contact
    stays above 2.22 A at every scan point, against 1.49 A for the single rigid
    hinge this replaced.

    This is arm dissociation, not selective imine loss: salen is one chain, so
    the departing nitrogen takes its own phenolate with it (Ni-O runs 1.85 ->
    6.40 A alongside Ni-N 1.85 -> 4.50 A) while the spectator arm holds both of
    its bonds at 1.8500 A exactly.  Read the scan as tetradentate -> bidentate.
    """
    torsions = torsions or salen_arm_torsions(struct, anchor, donor)
    if abs(struct.distance(anchor, donor) - target_distance) < 1e-9:
        return struct.copy()
    geometry, _ = _open_once(
        struct,
        anchor,
        donor,
        target_distance,
        torsions,
        (0.0, 0.0),
        _non_bonded_pairs(struct),
        outer_step,
    )
    return geometry


def salen_degradation_series(
    struct: Structure,
    anchor: int,
    donor: int,
    distances: Sequence[float] = (1.85, 2.10, 2.35, 2.60, 2.90, 3.30, 3.80, 4.50),
    outer_step: float = 5.0,
) -> list[tuple[float, Structure]]:
    """Starting geometries along the arm-opening coordinate.

    The torsion definitions are computed once on the intact complex and reused
    at every point, so the moving sets cannot change under a graph that has been
    distorted — that is how a scan quietly acquires a discontinuity.

    Each point is also seeded from the previous one, so the walk stays on one
    torsion branch.  Solving every point independently does not: the objective
    is capped, several branches tie, and the scan jumps between them.
    """
    torsions = salen_arm_torsions(struct, anchor, donor)
    pairs = _non_bonded_pairs(struct)
    seed = (0.0, 0.0)
    series: list[tuple[float, Structure]] = []
    for target in distances:
        if abs(struct.distance(anchor, donor) - target) < 1e-9:
            series.append((target, struct.copy()))
            continue
        geometry, seed = _open_once(
            struct, anchor, donor, target, torsions, seed, pairs, outer_step
        )
        series.append((target, geometry))
    return series
