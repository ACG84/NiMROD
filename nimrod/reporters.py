"""Spin reporters that could survive a working reactor.

The colour centre used so far is a synthetic construct: an sp3 aryl defect on a
polycyclic aromatic, which puts an S = 1/2 radical on a large pi system.  It is a
fine object for asking whether the physics works, and the calculations say it
does.  It is a poor object for an *operando* measurement, for reasons the
calculations themselves surfaced:

**The host bleeds the spin away.**  Going from pyrene (16 carbons) to pentacene
(22) dropped the exchange coupling from -23..-51 cm^-1 to -1.1 cm^-1, because the
larger pi system delocalises the defect's spin over more carbons and leaves less
of it at the tether.  The trend says the reporter should be *small and pinned*,
not large and delocalised -- the opposite of the direction the host was taken in.

**The sp3 defect is the least robust part.**  A benzylic sp3 C-H on a PAH is a
weak bond that autoxidises, and pentacene additionally forms an endoperoxide with
O2 under light within minutes.  A sensor that decomposes faster than the catalyst
it is watching measures nothing.

**Nothing was checked against the catalyst's redox window.**  Ni(salen)
catalysis cycles Ni(I)/Ni(II)/Ni(III) over roughly -1.7 to +0.8 V vs Fc.  A
reporter that oxidises or reduces inside that window is not a spectator; it is a
co-catalyst, or it is consumed.

So this module builds the reporters that are actually used as persistent spin
labels:

    phenalenyl          C13H9, the smallest odd-alternant PAH radical.  Same
                        physics as the colour centre -- a pi radical on a fused
                        aromatic -- but intrinsic rather than defect-derived, so
                        no sp3 C-H exists to oxidise.  Its SOMO is a
                        non-bonding orbital with strictly zero amplitude at
                        positions 2, 5, 8 and on all four interior carbons.

    nitronyl nitroxide  The metal-radical approach's standard coupler.  The SOMO
                        is confined to the O-N-C-N-O unit and, in C2v, is
                        antisymmetric about the aryl-bearing carbon, so the SOMO
                        has an exact node there.

    imino nitroxide     Nitronyl nitroxide with one oxygen removed.  Intended as
                        a control for the node -- see the warning below about
                        why it is not one.

A NODE IS NOT AN ABSENCE OF SPIN, AND IT DOES NOT SET THE SIGN OF J.
====================================================================

This module was first written around two claims that a screen (see
``scripts/run_reporter_screen.py`` and ``data/results/reporter_spin_screen.jsonl``)
was run to test.  Both failed, and the failures are recorded here rather than
quietly edited out, because they are the kind that produce clean converged
numbers answering the wrong question.

*A Huckel or symmetry node carries zero SOMO amplitude and non-zero spin
density.*  Spin density is not |SOMO|^2 -- it includes alpha/beta polarisation of
the doubly occupied orbitals.  Phenalenyl's nodal carbons 2, 5, 8 come out
NEGATIVE, not zero, as does C2 on both nitroxides.  So any claim of the form
"attaching at a node couples to nothing" is false about the sign.  This much
survives every check applied: PBE with 0% exact exchange gives a negative centre
in both Mulliken and Lowdin partitionings, so it is not an artefact of exact
exchange, spin contamination, or one population analysis.

*The magnitude, however, is not established.*  Across four functionals, two
bases and two partitionings (``data/results/spin_density_verify.jsonl``)
phenalenyl's nodal population runs from -0.164 to -0.0103 e, a factor of 16, and
it tracks the exact-exchange fraction in lockstep with <S^2> (0.768 -> 0.981
from 0% to 50% HF).  At the least-polarising functional with the largest basis
and the orthogonalised partitioning it is -0.010 e against +0.32 e at the SOMO
positions: three percent, which is a node for any practical purpose.  Quote the
sign; do not quote the magnitude without the method string and the spread.

*The nitronyl-versus-imino contrast is not evidence about a node.*  It looked
like it was: -0.272 e at the C2 node against -0.144 e at the control's non-node.
But total spin is exactly 1.000, so C2 is the sink that balances the sum rule,
and the difference in C2 populations (-0.1283 e) equals the difference in what
Mulliken over-assigns to the heteroatoms (0.1270 e) to within 1%.  "The node
carries twice the negative spin" is arithmetically the same sentence as
"nitronyl nitroxide has two N-O groups and imino nitroxide has one".  Imino
nitroxide is not a node control in any case -- it deletes an atom carrying 0.53 e
of spin and turns a 5-centre/7-electron pi system into a 4-centre/5-electron one.

*And the sign of J does not follow the sign of the local spin density.*  In
broken-symmetry DFT the antiferromagnetic term goes as -t^2/U, quadratic in the
magnetic-orbital overlap and therefore blind to the sign of any coefficient.
A node suppresses |J| by switching off the kinetic-exchange pathway; what
survives is a weak polarisation-mediated residue whose sign has to be computed,
not read off a free-radical population.  This project's own data already shows J
changing sign with the tether's spin density held positive -- +4.4 to +6.5 cm^-1
on the truncated model against -23 to -51 cm^-1 on real Ni(salen) at nearly the
same distance -- so pathway and geometry, not sign(rho), are what move it.

:func:`phenalenyl_nbmo` computes the non-bonding orbital by Huckel
diagonalisation rather than asserting where the nodes are.  That part was
right; what was wrong was reading its output as a spin density.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import (
    CH_AROMATIC,
    CH_SP3,
    TETRAHEDRAL_OUT_OF_PLANE,
    Structure,
    _rotate_about,
    _unit,
    build_pah,
)

# Nitronyl nitroxide, from the crystal structures of 2-aryl derivatives.  The
# two N-O bonds are equivalent: the SOMO is delocalised over O-N-C-N-O, which is
# what makes the radical persistent and what puts the node on C2.
NN_N_O = 1.28
NN_C2_N = 1.35
NN_N_C = 1.49
NN_C4_C5 = 1.55
NN_C2_ARYL = 1.47
NN_C_METHYL = 1.53
IN_C2_N_IMINE = 1.30        # the C=N that replaces one N-O
IN_N_C = 1.47


def phenalenyl() -> Structure:
    """Phenalenyl, C13H9 -- three hexagons peri-fused around a shared carbon."""
    return build_pah(((0, 0), (1, 0), (0, 1)), "phenalenyl")


def _carbon_adjacency(struct: Structure) -> tuple[list[int], np.ndarray]:
    carbons = [i for i, s in enumerate(struct.symbols) if s == "C"]
    position = {atom: k for k, atom in enumerate(carbons)}
    adjacency = np.zeros((len(carbons), len(carbons)))
    for atom in carbons:
        for neighbour in struct.neighbours(atom):
            if struct.symbols[neighbour] == "C":
                adjacency[position[atom], position[neighbour]] = 1.0
    return carbons, adjacency


def phenalenyl_nbmo(struct: Structure | None = None) -> dict[int, float]:
    """Huckel non-bonding orbital of phenalenyl, as {atom index: coefficient}.

    Phenalenyl is an odd alternant hydrocarbon, so exactly one eigenvalue of the
    carbon adjacency matrix is zero and its eigenvector is the singly occupied
    orbital.

    These are SOMO amplitudes, NOT spin densities, and the distinction is the
    one this module got wrong first time round.  UKS puts -0.16 e on each carbon
    where this returns exactly zero, because alpha/beta polarisation of the
    doubly occupied orbitals contributes to the spin density and not to the
    SOMO.  What the zeros do predict is that the kinetic-exchange pathway
    through those carbons vanishes -- that term goes as the SOMO amplitude
    squared -- so a tether there should give a small coupling rather than a
    sign-flipped one.  Even that has to be computed on the assembled complex.

    Returned coefficients are normalised so the largest is 1.
    """
    struct = struct or phenalenyl()
    carbons, adjacency = _carbon_adjacency(struct)
    values, vectors = np.linalg.eigh(adjacency)
    zero = int(np.argmin(np.abs(values)))
    if abs(values[zero]) > 1e-8:
        raise ValueError(
            f"no non-bonding orbital: closest eigenvalue is {values[zero]:.3e}; "
            f"this is not an odd alternant hydrocarbon")
    vector = vectors[:, zero]
    vector = vector / np.abs(vector).max()
    return {atom: float(c) for atom, c in zip(carbons, vector)}


def phenalenyl_sites(struct: Structure | None = None,
                     threshold: float = 1e-6) -> dict[str, list[int]]:
    """Split phenalenyl's C-H carbons into SOMO-bearing and nodal positions.

    "Nodal" means zero SOMO amplitude, not zero spin density: those carbons
    carry -0.16 e of spin-polarised density at UKS/B3LYP/def2-SVP, falling to
    -0.010 e at PBE/def2-TZVP with Lowdin populations.  The name refers to the
    orbital, which is what the Huckel calculation returns.

    The split still decides where a tether goes, but for the orbital reason
    rather than the population one: kinetic exchange scales as the SOMO
    amplitude squared at the tether, so the six SOMO-bearing positions are where
    a strong coupling can come from and the three nodal ones are where it should
    collapse to a weak polarisation-mediated residue.

    Usefully, the same three carbons are where a bench-stable phenalenyl is
    protected -- Kubo's 2,5,8-tri-tert-butyl radical -- so the protecting groups
    and the tether want different positions and the molecule stays usable.  Note
    the protection works sterically, by shielding the adjacent high-spin alpha
    carbons; the reactive positions are those alpha carbons, not the nodes.
    """
    struct = struct or phenalenyl()
    coefficients = phenalenyl_nbmo(struct)
    ch_carbons = [i for i, s in enumerate(struct.symbols)
                  if s == "C" and len(struct.hydrogens_on(i)) == 1]
    bearing = [i for i in ch_carbons if abs(coefficients[i]) > threshold]
    nodal = [i for i in ch_carbons if abs(coefficients[i]) <= threshold]
    return {"somo_bearing": bearing, "nodal": nodal}


#: How far C4 and C5 sit out of the O-N-C-N-O plane, in opposite directions.
#: The TETRAMETHYL five-ring is not flat.  Built flat, the four methyls eclipse
#: across the C4-C5 bond and the closest non-bonded contact is 1.68 A, which is
#: not a conformation, it is a clash that would wreck an SCF before any
#: chemistry was asked of it.  Real tetramethyl nitronyl nitroxides twist by a
#: few tenths of an angstrom; 0.35 A opens the worst contact to 2.19 A and puts
#: the cis methyl carbons at 3.31 A, the crystallographic range.
#:
#: The des-methyl model gets NO pucker, and the distinction is not cosmetic.
#: With only hydrogens on C4 and C5 there is nothing to relieve -- the flat ring
#: already clears 2.34 A -- and puckering it destroys BOTH mirror planes, taking
#: the point group from C2v to C2.  Every argument about the SOMO node on C2
#: rests on the C2v symmetry, so applying a methyl-derived correction to a
#: molecule with no methyls silently removes the symmetry the model exists to
#: exhibit.  It did, for one round of calculations, and the resulting "exact
#: symmetry node" was not exact.
NN_PUCKER = 0.35
NN_PUCKER_DESMETHYL = 0.0


def _five_ring_backbone(pucker: float | None = None) -> dict[str, np.ndarray]:
    """The N1-C2-N3-C4-C5 ring, built symmetric about the C2-aryl axis.

    Placing it by mirror symmetry rather than by walking the ring guarantees the
    ring closes exactly, and the C2v symmetry is not incidental -- it is what
    makes the two N-O groups equivalent and puts the SOMO node on C2.

    ``pucker`` twists C4 up and C5 down by that distance, with their in-plane
    coordinates re-solved so both the C4-C5 and N-C bond lengths are preserved
    exactly.  The O-N-C-N-O unit stays planar, which is the part the SOMO needs.
    """
    pucker = NN_PUCKER if pucker is None else pucker
    # C2 at the origin; the N-C2-N angle and the two C2-N bonds fix the nitrogens.
    n_c2_n = math.radians(110.0)
    y_n = math.sqrt(NN_C2_N ** 2 * (1.0 + math.cos(n_c2_n)) / 2.0)
    x_n = math.sqrt(NN_C2_N ** 2 - y_n ** 2)
    # C4/C5 straddle the axis; the twist eats into their in-plane separation, so
    # x shrinks to keep |C4-C5| fixed and y follows from the N-C bond length.
    span = NN_C4_C5 ** 2 - 4.0 * pucker ** 2
    if span <= 0.0:
        raise ValueError(f"pucker {pucker} exceeds half the C4-C5 bond")
    x_c = math.sqrt(span) / 2.0
    height = NN_N_C ** 2 - pucker ** 2 - (x_n - x_c) ** 2
    if height <= 0.0:
        raise ValueError(f"pucker {pucker} cannot close the five-membered ring")
    y_c = y_n + math.sqrt(height)
    return {
        "C2": np.array([0.0, 0.0, 0.0]),
        "N3": np.array([x_n, y_n, 0.0]),
        "N1": np.array([-x_n, y_n, 0.0]),
        "C4": np.array([x_c, y_c, pucker]),
        "C5": np.array([-x_c, y_c, -pucker]),
    }


def _external_bisector(centre: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector pointing away from both neighbours, in their plane."""
    return _unit(-(_unit(a - centre) + _unit(b - centre)))


def _sp3_substituents(centre: np.ndarray, a: np.ndarray, b: np.ndarray,
                      length: float) -> tuple[np.ndarray, np.ndarray]:
    """The two remaining positions on a tetrahedral centre bonded to a and b.

    The perpendicular is taken from the local a-centre-b plane rather than from
    a fixed axis, so this stays correct once the ring is puckered -- with a
    hard-coded z normal the twist would place the methyls as if the ring were
    still flat, which is the failure it is there to fix.
    """
    outward = _external_bisector(centre, a, b)
    normal = _unit(np.cross(a - centre, b - centre))
    tilt = TETRAHEDRAL_OUT_OF_PLANE
    up = math.cos(tilt) * outward + math.sin(tilt) * normal
    down = math.cos(tilt) * outward - math.sin(tilt) * normal
    return centre + length * _unit(up), centre + length * _unit(down)


def _methyl(anchor: np.ndarray, direction: np.ndarray,
            phase: float = 0.0) -> tuple[list[str], list[np.ndarray]]:
    """A methyl group whose carbon sits along ``direction`` from ``anchor``.

    ``phase`` rotates the three hydrogens about the C-C axis.  It has to be a
    parameter rather than a fixed choice: a nitronyl nitroxide carries two
    gem-dimethyl pairs whose methyls are close enough that an arbitrary
    rotation puts hydrogens 1.6 A apart, and :func:`_stagger_methyls` picks the
    angles that do not.
    """
    carbon = anchor + NN_C_METHYL * _unit(direction)
    axis = _unit(direction)
    # Any two vectors orthogonal to the axis span the H positions.
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, axis))) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(axis, seed))
    v = np.cross(axis, u)
    tilt = math.radians(180.0 - 109.5)
    symbols, coords = ["C"], [carbon]
    for k in range(3):
        phi = phase + 2.0 * math.pi * k / 3.0
        radial = math.cos(phi) * u + math.sin(phi) * v
        offset = math.cos(tilt) * axis + math.sin(tilt) * radial
        symbols.append("H")
        coords.append(carbon + CH_SP3 * _unit(offset))
    return symbols, coords


def _non_bonded_pairs(struct: Structure) -> np.ndarray:
    neighbours = [set(struct.neighbours(i)) for i in range(len(struct))]
    return np.array([(i, j)
                     for i in range(len(struct))
                     for j in range(i + 1, len(struct))
                     if j not in neighbours[i] and not (neighbours[i] & neighbours[j])])


def _stagger_methyls(struct: Structure, methyls: list[tuple[int, int, list[int]]],
                     steps: int = 36) -> Structure:
    """Rotate each methyl about its own C-C bond to open up the worst contact.

    Placing four methyls by construction alone leaves their rotations arbitrary,
    and on a tetramethyl nitroxide the arbitrary choice puts two hydrogens
    1.62 A apart -- closer than a hydrogen bond, and enough to wreck an SCF
    before any chemistry is asked of it.  A rigid rotation about the C-C axis
    cannot distort anything, so scanning it is free.

    Greedy one methyl at a time: the couplings between them are weak enough that
    a joint search buys nothing, and each pass can only improve the objective.
    """
    struct = struct.copy()
    pairs = _non_bonded_pairs(struct)
    if not len(pairs):
        return struct

    def worst(coords: np.ndarray) -> float:
        return float(np.linalg.norm(
            coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1).min())

    for anchor, carbon, hydrogens in methyls:
        axis = struct.coords[carbon] - struct.coords[anchor]
        origin = struct.coords[carbon]
        best_angle, best_score = 0.0, worst(struct.coords)
        for step in range(1, steps):
            angle = 2.0 * math.pi * step / (3 * steps)   # one full methyl period
            trial = struct.coords.copy()
            trial[hydrogens] = _rotate_about(
                trial[hydrogens], origin, axis, angle)
            score = worst(trial)
            if score > best_score:
                best_angle, best_score = angle, score
        if best_angle:
            struct.coords[hydrogens] = _rotate_about(
                struct.coords[hydrogens], origin, axis, best_angle)
    return struct


def nitronyl_nitroxide(methylated: bool = True) -> tuple[Structure, dict[str, int]]:
    """2-substituted nitronyl nitroxide, C2 first and its hydrogen second.

    Ordered so :func:`~nimrod.geometry.attach` can bond it straight onto an aryl
    ring: atom 0 is C2, atom 1 is the hydrogen that gets replaced.

    ``methylated`` keeps the four methyls that make the real radical persistent
    (they block the alpha positions of the sp3 carbons).  Turning them off gives
    a des-methyl model with the same SOMO and 12 fewer atoms, which is the right
    trade when the question is electronic rather than chemical.

    The des-methyl model is built FLAT.  The pucker exists only to unclash the
    methyls, and carrying it over to a molecule that has none costs the two
    mirror planes: puckered, this is C2, and only the pi component of the SOMO
    node on C2 is symmetry-protected.  Flat, it is C2v and the node is exact
    over every atomic orbital on C2, which is the property the model is for.
    """
    ring = _five_ring_backbone(
        pucker=NN_PUCKER if methylated else NN_PUCKER_DESMETHYL)
    symbols: list[str] = ["C"]
    coords: list[np.ndarray] = [ring["C2"]]

    # The aryl handle points away from the ring, along -y from C2.
    symbols.append("H")
    coords.append(ring["C2"] + CH_AROMATIC * np.array([0.0, -1.0, 0.0]))

    index: dict[str, int] = {"C2": 0, "H2": 1}
    for label in ("N1", "N3", "C4", "C5"):
        index[label] = len(symbols)
        symbols.append("N" if label.startswith("N") else "C")
        coords.append(ring[label])

    # N-oxide oxygens, bisecting the ring angle externally but held in the
    # O-N-C-N-O plane.  The bisector is taken from the *unpuckered* ring: using
    # the twisted C4/C5 would tilt each N-O out of the plane by 0.2 A and in
    # opposite senses, which breaks the C2v symmetry that makes the two N-O
    # groups equivalent -- and that equivalence is the reason the SOMO spans
    # all five atoms instead of localising on one nitroxide.
    flat = _five_ring_backbone(pucker=0.0)
    for nitrogen, ring_neighbour in (("N1", "C5"), ("N3", "C4")):
        direction = _external_bisector(flat[nitrogen], flat["C2"], flat[ring_neighbour])
        index[f"O_{nitrogen}"] = len(symbols)
        symbols.append("O")
        coords.append(ring[nitrogen] + NN_N_O * direction)

    # The two sp3 carbons each carry a pair of out-of-plane substituents.
    methyls: list[tuple[int, int, list[int]]] = []
    for carbon, ring_neighbour in (("C4", "N3"), ("C5", "N1")):
        up, down = _sp3_substituents(
            ring[carbon], ring[ring_neighbour],
            ring["C5" if carbon == "C4" else "C4"],
            NN_C_METHYL if methylated else CH_SP3)
        for position in (up, down):
            if methylated:
                extra_symbols, extra_coords = _methyl(
                    ring[carbon], position - ring[carbon])
                start = len(symbols)
                symbols.extend(extra_symbols)
                coords.extend(extra_coords)
                methyls.append((index[carbon], start,
                                [start + 1, start + 2, start + 3]))
            else:
                symbols.append("H")
                coords.append(position)

    name = "nitronyl-nitroxide" if methylated else "nitronyl-nitroxide-desmethyl"
    struct = Structure(symbols, np.array(coords), name)
    if methyls:
        struct = _stagger_methyls(struct, methyls)
    return struct, index


def imino_nitroxide(methylated: bool = True) -> tuple[Structure, dict[str, int]]:
    """Imino nitroxide: nitronyl nitroxide with one N-oxide oxygen removed.

    Removing the oxygen destroys the C2v symmetry, so the SOMO is no longer
    antisymmetric about C2 and the aryl carbon stops being a SOMO node.  It was
    built as the control for whether that node suppresses the coupling.

    IT IS NOT A CLEAN CONTROL, and the spin-density screen showed why.  Deleting
    the oxygen changes four things at once: it removes an atom carrying 0.53 e of
    spin, turns a 5-centre/7-electron pi system into a 4-centre/5-electron one,
    converts a C-N single bond into a C=N double bond, and lifts the symmetry.
    The measured difference in spin density at C2 (-0.272 vs -0.144 e) matches
    the difference in what Mulliken assigns to the heteroatoms to within 1%, so
    it is the change in composition being observed, not the change in symmetry.

    A control that isolates the node has to break the symmetry at FIXED
    composition -- an antisymmetric N-O stretch on nitronyl nitroxide itself
    lifts the node continuously with the same atoms and the same electron count.
    This variant remains useful as a second real reporter with different
    stability and a different SOMO; it should not be used as the node control.
    """
    struct, index = nitronyl_nitroxide(methylated=methylated)
    oxygen = index["O_N3"]
    remaining = [i for i in range(len(struct)) if i != oxygen]

    # N3 becomes an imine nitrogen: shorten C2=N3 and N3-C4 to match.
    coords = struct.coords.copy()
    n3, c2, c4 = index["N3"], index["C2"], index["C4"]
    coords[n3] = coords[c2] + IN_C2_N_IMINE * _unit(coords[n3] - coords[c2])
    coords[c4] = coords[n3] + IN_N_C * _unit(coords[c4] - coords[n3])

    trimmed = Structure([struct.symbols[i] for i in remaining], coords[remaining],
                        "imino-nitroxide" if methylated else "imino-nitroxide-desmethyl")
    remap = {label: idx - (1 if idx > oxygen else 0)
             for label, idx in index.items() if idx != oxygen}
    return trimmed, remap


#: Why each reporter is or is not usable in a running reactor.  Redox potentials
#: are vs ferrocene in MeCN; Ni(salen) catalysis spans roughly -1.7 to +0.8 V,
#: so anything with a potential inside that window is not a spectator.
OPERANDO_PROFILE: dict[str, dict[str, object]] = {
    "pentacene-occ": {
        "heavy_atoms": 22,
        "radical_source": "sp3 aryl defect",
        "air_stable": False,
        "why": "forms an endoperoxide with O2 under light in minutes, and the "
               "benzylic sp3 C-H autoxidises; decomposes faster than the "
               "catalyst it watches",
        "redox_window_v": (-1.3, 0.6),
        "inside_catalyst_window": True,
    },
    "pyrene-occ": {
        "heavy_atoms": 16,
        "radical_source": "sp3 aryl defect",
        "air_stable": False,
        "why": "more robust pi system than pentacene, but the sp3 defect is "
               "still a weak benzylic C-H",
        "redox_window_v": (-2.1, 1.2),
        "inside_catalyst_window": False,
    },
    "phenalenyl": {
        "heavy_atoms": 13,
        "radical_source": "intrinsic (odd alternant)",
        "air_stable": False,
        "why": "the parent radical sigma-dimerises and is attacked by O2 at the "
               "six alpha carbons that carry the positive spin density; "
               "tert-butyl at 2, 5, 8 shields them sterically and gives a "
               "bench-stable radical without touching the SOMO, because those "
               "three positions are its nodes",
        "redox_window_v": (-1.0, 0.1),
        "inside_catalyst_window": True,
    },
    "phenalenyl-tri-tert-butyl": {
        "heavy_atoms": 25,
        "radical_source": "intrinsic (odd alternant)",
        "air_stable": True,
        "why": "nodal positions blocked; persistent in the solid state and in "
               "solution, with a near-IR absorption that clears the visible "
               "window a reaction medium crowds",
        "redox_window_v": (-1.0, 0.1),
        "inside_catalyst_window": True,
    },
    "nitronyl-nitroxide": {
        "heavy_atoms": 11,
        "radical_source": "intrinsic (N-O-delocalised SOMO)",
        "air_stable": True,
        "why": "persistent in air, in water, and above 100 C; the standard "
               "coupler of the metal-radical approach, so its exchange "
               "behaviour with 3d metals is characterised experimentally",
        "redox_window_v": (-1.9, 0.9),
        "inside_catalyst_window": False,
        "caveat": "the SOMO has an exact node on the aryl-bearing carbon in "
                  "C2v, so the kinetic-exchange pathway through the tether -- "
                  "which scales as the SOMO amplitude squared -- should be "
                  "suppressed and |J| small.  The node does NOT mean C2 has no "
                  "spin density (it carries -0.27 e of polarisation) and it "
                  "does NOT reverse the sign of J.  Measure J on the assembly",
        "coordination_hazard": "the two N-oxide oxygens carry more spin than any "
                               "other atom (+0.39 e each) and are exactly the "
                               "donors TEMPO is rejected for.  The event being "
                               "sensed opens a coordination site, so a reporter "
                               "that can reach it would poison what it measures. "
                               "Tethered at C3 it cannot: the closest Ni...O "
                               "approach over a full torsion scan is 2.99 A "
                               "against the 1.9-2.1 A a dative bond needs.  This "
                               "is a rigid-torsion bound, not a proof -- bending "
                               "could close it, and only the assembled "
                               "optimisation settles it",
    },
    "imino-nitroxide": {
        "heavy_atoms": 10,
        "radical_source": "intrinsic (N-C-N-O SOMO)",
        "air_stable": True,
        "why": "no SOMO node on the aryl carbon, so the kinetic-exchange "
               "pathway is open; less persistent than nitronyl nitroxide but "
               "still a bench radical",
        "redox_window_v": (-1.8, 1.0),
        "inside_catalyst_window": False,
        "caveat": "NOT a clean control for the node: deleting the oxygen changes "
                  "the pi electron count, the bond orders and an atom carrying "
                  "0.53 e of spin at the same time as the symmetry",
    },
    "TEMPO": {
        "heavy_atoms": 10,
        "radical_source": "intrinsic (N-O SOMO)",
        "air_stable": True,
        "why": "rejected: the nitroxide oxygen coordinates nickel and the "
               "TEMPO+/TEMPO couple sits inside the catalytic window, so it is "
               "a reagent in this system rather than a spectator",
        "redox_window_v": (-1.2, 0.2),
        "inside_catalyst_window": True,
    },
}
