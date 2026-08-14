"""Structural tests for the real Ni(salen) catalyst and the pentacene sensor.

Like :mod:`tests.test_geometry`, every assertion here is anchored to known
chemistry — published formulas, coordination numbers, chelate ring sizes,
literature bond lengths — rather than to a previous run of the builder.  The
failure this file exists to catch is a Ni(salen) that looks entirely plausible
in a viewer but has its donors trans instead of cis; salen is a single
tetradentate chain and physically cannot span trans positions, so the cis
tests below are the ones that decide whether the molecule is the right one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nimrod.catalysts import (
    ni_salen,
    meso_site,
    open_salen_arm,
    pentacene,
    pentacene_occ,
    pentacene_sensor,
    salen_arm_torsions,
    salen_degradation_series,
    tetracene,
)
from nimrod.geometry import chelate_hinge, named_pah, plane_normal, swing_arm

DEGRADATION_TARGETS = (1.85, 2.10, 2.35, 2.60, 2.90, 3.30, 3.80, 4.50)


def angle(struct, i: int, j: int, k: int) -> float:
    """Bond angle i-j-k in degrees."""
    a = struct.coords[i] - struct.coords[j]
    b = struct.coords[k] - struct.coords[j]
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def all_angles(struct) -> dict[tuple[int, int, int], float]:
    """Every bond angle in the molecule, keyed by (i, j, k) with j the vertex."""
    out = {}
    for j in range(len(struct)):
        neighbours = sorted(struct.neighbours(j))
        for a_i, i in enumerate(neighbours):
            for k in neighbours[a_i + 1 :]:
                out[(i, j, k)] = angle(struct, i, j, k)
    return out


def closest_non_bonded(struct) -> float:
    """Closest contact that is neither a bond nor fixed by a bond angle.

    ``min_interatomic_distance`` cannot be used to judge a scan geometry: it is a
    minimum over *all* pairs, so it sits at 1.08 A — the aromatic C-H bond — no
    matter how badly two hydrogens collide.
    """
    neighbours = [set(struct.neighbours(i)) for i in range(len(struct))]
    return min(
        struct.distance(i, j)
        for i in range(len(struct))
        for j in range(i + 1, len(struct))
        if j not in neighbours[i] and not (neighbours[i] & neighbours[j])
    )


# --------------------------------------------------------------------------
# Ni(salen): composition
# --------------------------------------------------------------------------


def test_ni_salen_formula_is_the_literature_compound() -> None:
    """salenH2 = C16H16N2O2; salen(2-) = C16H14N2O2; with Ni(2+), NiC16H14N2O2."""
    catalyst, _ = ni_salen()
    assert catalyst.formula == "C16H14N2NiO2"
    assert len(catalyst) == 35


def test_ni_salen_atom_counts_by_element() -> None:
    catalyst, _ = ni_salen()
    counts = {s: catalyst.symbols.count(s) for s in set(catalyst.symbols)}
    # 2 x (6 aryl C + 1 imine C) + 2 bridge C; 2 x (4 aryl H + 1 imine H) + 4 CH2 H
    assert counts == {"Ni": 1, "C": 16, "H": 14, "N": 2, "O": 2}


def test_ni_salen_is_closed_shell() -> None:
    """Square-planar d8 Ni(II) with a dianionic ligand: neutral and S = 0."""
    catalyst, _ = ni_salen()
    assert catalyst.n_electrons % 2 == 0
    assert catalyst.multiplicity_for(0) == 1


# --------------------------------------------------------------------------
# Ni(salen): the coordination sphere
# --------------------------------------------------------------------------


def test_nickel_is_four_coordinate_n2o2() -> None:
    catalyst, index = ni_salen()
    neighbours = catalyst.neighbours(index["Ni"])
    assert len(neighbours) == 4
    assert sorted(catalyst.symbols[j] for j in neighbours) == ["N", "N", "O", "O"]


@pytest.mark.parametrize("donor", ["O1", "N1", "O2", "N2"])
def test_donor_distances_are_physical(donor: str) -> None:
    catalyst, index = ni_salen()
    assert 1.75 < catalyst.distance(index["Ni"], index[donor]) < 1.95


def test_the_two_oxygens_are_cis() -> None:
    """The tetradentate signature.  Trans donors would give ~180 deg and would
    mean the builder made two independent bidentate arms, not salen."""
    catalyst, index = ni_salen()
    assert angle(catalyst, index["O1"], index["Ni"], index["O2"]) < 120.0


def test_the_two_nitrogens_are_cis() -> None:
    catalyst, index = ni_salen()
    assert angle(catalyst, index["N1"], index["Ni"], index["N2"]) < 120.0


@pytest.mark.parametrize("arm", ["1", "2"])
def test_six_ring_bite_angle(arm: str) -> None:
    """O-Ni-N of a salicylaldiminato chelate is ~94 deg."""
    catalyst, index = ni_salen()
    assert 88.0 < angle(catalyst, index[f"O{arm}"], index["Ni"], index[f"N{arm}"]) < 100.0


def test_five_ring_bite_is_tighter_than_the_six_ring_bite() -> None:
    """N-Ni-N (five-membered) must be smaller than O-Ni-N (six-membered)."""
    catalyst, index = ni_salen()
    n_n = angle(catalyst, index["N1"], index["Ni"], index["N2"])
    o_n = angle(catalyst, index["O1"], index["Ni"], index["N1"])
    assert 80.0 < n_n < 92.0
    assert n_n < o_n


def test_the_four_bite_angles_close_the_plane() -> None:
    """Square planar: the four cis angles around the metal sum to 360 deg."""
    catalyst, index = ni_salen()
    total = (
        angle(catalyst, index["O1"], index["Ni"], index["N1"])
        + angle(catalyst, index["N1"], index["Ni"], index["N2"])
        + angle(catalyst, index["N2"], index["Ni"], index["O2"])
        + angle(catalyst, index["O2"], index["Ni"], index["O1"])
    )
    assert total == pytest.approx(360.0, abs=1.0)


# --------------------------------------------------------------------------
# Ni(salen): the three fused chelate rings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm,suffix", [("1", "a"), ("2", "b")])
def test_six_membered_chelate_ring_closes(arm: str, suffix: str) -> None:
    """Ni-O-C(ar)-C(ar)-CH=N-Ni, the salicylaldiminato ring."""
    catalyst, index = ni_salen()
    ring = ["Ni", f"O{arm}", f"C2{suffix}", f"C1{suffix}", f"C7{suffix}", f"N{arm}"]
    assert len(set(index[label] for label in ring)) == 6
    for a, b in zip(ring, ring[1:] + ["Ni"]):
        assert index[b] in catalyst.neighbours(index[a]), f"{a}-{b} not bonded"


def test_five_membered_diamine_ring_closes() -> None:
    """Ni-N-CH2-CH2-N-Ni, the ethylenediamine bridge."""
    catalyst, index = ni_salen()
    ring = ["Ni", "N1", "Cbr1", "Cbr2", "N2"]
    assert len(set(index[label] for label in ring)) == 5
    for a, b in zip(ring, ring[1:] + ["Ni"]):
        assert index[b] in catalyst.neighbours(index[a]), f"{a}-{b} not bonded"


@pytest.mark.parametrize("suffix", ["a", "b"])
def test_benzene_rings_are_intact(suffix: str) -> None:
    catalyst, index = ni_salen()
    ring = [index[f"C{k}{suffix}"] for k in range(1, 7)]
    assert len(set(ring)) == 6
    assert all(catalyst.symbols[i] == "C" for i in ring)
    for k in range(6):
        assert ring[(k + 1) % 6] in catalyst.neighbours(ring[k])
        # aromatic C-C, not a stretched or collapsed one
        assert 1.34 < catalyst.distance(ring[k], ring[(k + 1) % 6]) < 1.46


@pytest.mark.parametrize("suffix", ["a", "b"])
def test_c5_is_para_to_the_phenolate_oxygen(suffix: str) -> None:
    """Position 5 of salicylaldehyde is para to the O-bearing carbon; that is
    the standard functionalisation handle and where the colour centre goes."""
    catalyst, index = ni_salen()
    c_o, c5 = index[f"C2{suffix}"], index[f"C5{suffix}"]
    # para carbons of a benzene ring sit 2 x 1.40 A apart across the ring
    assert catalyst.distance(c_o, c5) == pytest.approx(2.80, abs=0.05)
    assert c5 not in catalyst.neighbours(c_o)
    assert not set(catalyst.neighbours(c_o)) & set(catalyst.neighbours(c5))
    # and it is an unsubstituted CH, ready to be replaced
    assert sorted(catalyst.symbols[j] for j in catalyst.neighbours(c5)) == ["C", "C", "H"]
    assert index[f"H5{suffix}"] in catalyst.neighbours(c5)


# --------------------------------------------------------------------------
# Ni(salen): local chemistry of each site
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,symbol,neighbours",
    [
        ("O1", "O", ["C", "Ni"]),          # phenolate: metal + aryl, no H left
        ("O2", "O", ["C", "Ni"]),
        ("N1", "N", ["C", "C", "Ni"]),     # imine N: metal, imine C, CH2
        ("N2", "N", ["C", "C", "Ni"]),
        ("C7a", "C", ["C", "H", "N"]),     # imine CH
        ("C7b", "C", ["C", "H", "N"]),
        ("C2a", "C", ["C", "C", "O"]),     # aryl C bearing the phenolate O
        ("C1a", "C", ["C", "C", "C"]),     # aryl C bearing the imine
        ("Cbr1", "C", ["C", "H", "H", "N"]),  # bridge methylene
        ("Cbr2", "C", ["C", "H", "H", "N"]),
    ],
)
def test_site_connectivity(label: str, symbol: str, neighbours: list[str]) -> None:
    catalyst, index = ni_salen()
    i = index[label]
    assert catalyst.symbols[i] == symbol
    assert sorted(catalyst.symbols[j] for j in catalyst.neighbours(i)) == neighbours


@pytest.mark.parametrize(
    "a,b,low,high",
    [
        ("C2a", "O1", 1.25, 1.38),   # C-O phenolate
        ("C7a", "N1", 1.24, 1.36),   # C=N imine
        ("C1a", "C7a", 1.40, 1.50),  # aryl-C(imine)
        ("N1", "Cbr1", 1.42, 1.52),  # N-CH2
        ("Cbr1", "Cbr2", 1.46, 1.58),  # CH2-CH2
    ],
)
def test_ligand_bond_lengths(a: str, b: str, low: float, high: float) -> None:
    catalyst, index = ni_salen()
    assert low < catalyst.distance(index[a], index[b]) < high


def test_imine_nitrogen_is_planar_sp2() -> None:
    """Its three bond angles must sum to 360, or the imine is pyramidalised."""
    catalyst, index = ni_salen()
    total = (
        angle(catalyst, index["Ni"], index["N1"], index["C7a"])
        + angle(catalyst, index["Ni"], index["N1"], index["Cbr1"])
        + angle(catalyst, index["C7a"], index["N1"], index["Cbr1"])
    )
    assert total == pytest.approx(360.0, abs=1.0)


def test_bridge_methylene_hydrogens_straddle_the_ligand_plane() -> None:
    """A planar CH2 would be sp2 and wrong; the two H must sit either side."""
    catalyst, index = ni_salen()
    z_up = catalyst.coords[index["Hbr1u"], 2]
    z_down = catalyst.coords[index["Hbr1d"], 2]
    assert z_up > 0.5 and z_down < -0.5
    assert angle(catalyst, index["Hbr1u"], index["Cbr1"], index["Hbr1d"]) == pytest.approx(
        109.5, abs=3.0
    )


def test_heavy_atom_skeleton_is_planar() -> None:
    """Only the sp3 methylene hydrogens leave the plane."""
    catalyst, _ = ni_salen()
    heavy = [i for i, s in enumerate(catalyst.symbols) if s != "H"]
    assert np.allclose(catalyst.coords[heavy, 2], 0.0, atol=1e-9)


def test_ni_salen_has_no_clashes() -> None:
    catalyst, _ = ni_salen()
    assert catalyst.min_interatomic_distance() > 0.95


def test_every_hydrogen_has_exactly_one_heavy_neighbour() -> None:
    """Catches hydrogens left behind by a bad join, or stacked on each other."""
    catalyst, _ = ni_salen()
    for i, symbol in enumerate(catalyst.symbols):
        if symbol == "H":
            assert len(catalyst.neighbours(i)) == 1


def test_the_two_arms_are_identical() -> None:
    """Arm 2 is arm 1 mirrored, so any later asymmetry is real, not a build bug."""
    catalyst, index = ni_salen()
    for label in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        assert catalyst.distance(index["Ni"], index[f"{label}a"]) == pytest.approx(
            catalyst.distance(index["Ni"], index[f"{label}b"]), abs=1e-9
        )


# --------------------------------------------------------------------------
# Pentacene
# --------------------------------------------------------------------------


def test_pentacene_is_c22h14() -> None:
    struct = pentacene()
    assert struct.formula == "C22H14"
    assert len(struct) == 36
    assert struct.min_interatomic_distance() > 1.0
    assert np.allclose(struct.coords[:, 2], 0.0, atol=1e-9)


def test_tetracene_is_c18h12() -> None:
    assert tetracene().formula == "C18H12"


def test_named_acenes_agree_with_the_builder() -> None:
    """PAH_RINGS and ACENE_RINGS must not drift apart."""
    for name, struct in (("pentacene", pentacene()), ("tetracene", tetracene())):
        assert np.allclose(named_pah(name).coords, struct.coords)


def test_meso_site_is_on_the_central_ring() -> None:
    """Positions 6 and 13 of pentacene are the reactive meso carbons; a defect
    on a terminal ring perturbs the pi system far less."""
    struct = pentacene()
    site = meso_site(struct)
    assert struct.symbols[site] == "C"
    assert len(struct.hydrogens_on(site)) == 1
    centroid = struct.coords.mean(axis=0)
    # the central-ring CH carbons sit one aromatic bond from the long axis
    assert np.linalg.norm(struct.coords[site] - centroid) == pytest.approx(1.42, abs=0.1)


# --------------------------------------------------------------------------
# The free colour centre
# --------------------------------------------------------------------------


def test_pentacene_occ_composition() -> None:
    """Pentacene C22H14 + phenyl C6H5 = C28H19, an odd-electron doublet."""
    occ, _ = pentacene_occ()
    assert occ.formula == "C28H19"
    assert len(occ) == 47
    assert occ.n_electrons % 2 == 1
    assert occ.multiplicity_for(0) == 2
    assert occ.min_interatomic_distance() > 0.95


def test_pentacene_occ_defect_is_sp3() -> None:
    occ, info = pentacene_occ()
    site = info["sp3_carbon"]
    assert occ.symbols[site] == "C"
    assert sorted(occ.symbols[j] for j in occ.neighbours(site)) == ["C", "C", "C", "H"]


# --------------------------------------------------------------------------
# The assembled sensor
# --------------------------------------------------------------------------


def test_sensor_composition() -> None:
    """C22H14 + NiC16H14N2O2 - H (the two replaced hydrogens, one kept as the
    sp3 C-H) = NiC38H27N2O2."""
    sensor, _ = pentacene_sensor()
    assert sensor.formula == "C38H27N2NiO2"
    assert len(sensor) == 70


def test_sensor_is_a_doublet() -> None:
    """The sp3 defect is what creates the spin.  An even electron count here
    means the defect carbon did not end up four-coordinate."""
    sensor, _ = pentacene_sensor()
    assert sensor.n_electrons % 2 == 1
    assert sensor.multiplicity_for(0) == 2


def test_sensor_has_no_clashes() -> None:
    sensor, _ = pentacene_sensor()
    assert sensor.min_interatomic_distance() > 0.95


def test_sensor_defect_carbon_is_sp3() -> None:
    sensor, info = pentacene_sensor()
    site = info["sp3_carbon"]
    neighbours = sensor.neighbours(site)
    assert len(neighbours) == 4
    assert sorted(sensor.symbols[j] for j in neighbours) == ["C", "C", "C", "H"]


def test_sensor_defect_is_bonded_to_the_salicylidene_c5() -> None:
    sensor, info = pentacene_sensor()
    assert info["ipso_carbon"] in sensor.neighbours(info["sp3_carbon"])
    assert 1.45 < sensor.distance(info["sp3_carbon"], info["ipso_carbon"]) < 1.60


def test_sensor_keeps_the_intact_coordination_sphere() -> None:
    """Building the sensor must not disturb the catalyst it reports on."""
    sensor, info = pentacene_sensor()
    catalyst, catalyst_index = ni_salen()
    neighbours = sensor.neighbours(info["Ni"])
    assert sorted(sensor.symbols[j] for j in neighbours) == ["N", "N", "O", "O"]
    for donor in ("O1", "N1", "O2", "N2"):
        assert sensor.distance(info["Ni"], info[donor]) == pytest.approx(
            catalyst.distance(catalyst_index["Ni"], catalyst_index[donor]), abs=1e-9
        )


def test_sensor_pentacene_survives_intact() -> None:
    """22 aromatic carbons in, 21 aromatic plus one sp3 out."""
    sensor, info = pentacene_sensor()
    host_atoms = range(info["n_host_atoms"])
    assert sum(1 for i in host_atoms if sensor.symbols[i] == "C") == 22
    for i in host_atoms:
        if sensor.symbols[i] != "C" or i == info["sp3_carbon"]:
            continue
        carbons = [j for j in sensor.neighbours(i) if sensor.symbols[j] == "C"]
        assert 2 <= len(carbons) <= 3


# --------------------------------------------------------------------------
# The degradation coordinate
# --------------------------------------------------------------------------
#
# A salen nitrogen sits in *two* chelate rings, so the generic hinge in
# geometry.py — written for a donor in one ring — does something different
# here, and the tests below pin down exactly what.


def test_generic_chelate_hinge_pivots_on_the_other_nitrogen() -> None:
    """chelate_hinge takes the shortest path from the donor back to the metal.
    For salen that is the five-membered diamine ring, N1-CH2-CH2-N2-Ni, so the
    hinge lands on the second nitrogen rather than anywhere in the six-ring."""
    sensor, info = pentacene_sensor()
    pivot, axis, moving = chelate_hinge(sensor, info["Ni"], info["N1"])
    assert pivot == info["N2"]
    assert sensor.symbols[pivot] == "N"
    assert np.isclose(np.linalg.norm(axis), 1.0)
    # The metal and the far arm stay put; the near arm's own oxygen does not.
    assert info["Ni"] not in moving
    assert info["N2"] not in moving and info["O2"] not in moving
    assert info["N1"] in moving and info["O1"] in moving


def test_generic_chelate_hinge_cannot_reach_the_end_of_the_scan() -> None:
    """The lever it gives is |Ni-N2| + |N2-N1| = 4.37 A, so the 4.5 A point of
    the degradation scan is unreachable.  Reaching further needs a different
    coordinate, and the next test shows why it cannot be another rigid hinge."""
    sensor, info = pentacene_sensor()
    pivot, _, _ = chelate_hinge(sensor, info["Ni"], info["N1"])
    reach = sensor.distance(info["Ni"], pivot) + sensor.distance(pivot, info["N1"])
    assert reach == pytest.approx(4.37, abs=0.05)
    with pytest.raises(ValueError, match="reach|hinge|extends"):
        swing_arm(sensor, info["Ni"], info["N1"], 4.50)


def test_a_rigid_hinge_cannot_open_salen_without_wrecking_an_angle() -> None:
    """Why the coordinate is two torsions and not one rotation.

    Any single rotation about a pivot atom preserves every *distance* in the
    moving set — and nothing else.  The bond angles at the pivot are free, and
    on salen they go: hinging the arm about the chelate-plane normal at the far
    bridge methylene reproduces every bond length to 1e-15 A while opening that
    sp3 carbon past linear and slamming its two methylene hydrogens together.
    A length-only check passes such a geometry, which is exactly how it got in.
    """
    sensor, info = pentacene_sensor()
    pivot, donor = info["Cbr1"], info["N2"]
    axis = plane_normal(
        np.array(
            [sensor.coords[info["Ni"]], sensor.coords[pivot], sensor.coords[donor]]
        )
    )
    blocked = {info["Ni"], pivot}
    moving, seen, stack = [], set(blocked), [donor]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        moving.append(current)
        stack.extend(j for j in sensor.neighbours(current) if j not in seen)
    hinged = swing_arm(
        sensor, info["Ni"], donor, 4.50, hinge=(pivot, axis, sorted(moving))
    )

    # every bond length perfect ...
    for a, b in sensor.bonds():
        if info["Ni"] in (a, b):
            continue
        assert hinged.distance(a, b) == pytest.approx(sensor.distance(a, b), abs=1e-9)
    # ... and the chemistry destroyed anyway
    assert angle(hinged, info["N1"], pivot, info["Cbr2"]) > 170.0
    assert closest_non_bonded(hinged) < 1.6
    # min_interatomic_distance sees none of it: it is pinned by the C-H bonds
    assert hinged.min_interatomic_distance() == pytest.approx(1.08, abs=1e-6)


# --------------------------------------------------------------------------
# The degradation coordinate: two ethylenediamine torsions
# --------------------------------------------------------------------------


def test_salen_arm_torsions_are_the_two_bridge_bonds() -> None:
    sensor, info = pentacene_sensor()
    (outer_a, outer_b, _), (inner_a, inner_b, _) = salen_arm_torsions(
        sensor, info["Ni"], info["N2"]
    )
    assert (outer_a, outer_b) == (info["N1"], info["Cbr1"])
    assert (inner_a, inner_b) == (info["Cbr1"], info["Cbr2"])
    # both are real bonds, or they are not torsions
    assert outer_b in sensor.neighbours(outer_a)
    assert inner_b in sensor.neighbours(inner_a)


def test_salen_arm_torsions_move_one_whole_arm_and_leave_the_other() -> None:
    """Salen is one chain, so a single donor cannot leave the metal alone: the
    arm's imine N and phenolate O go together, tetradentate -> bidentate."""
    sensor, info = pentacene_sensor()
    _, (_, _, moving) = salen_arm_torsions(sensor, info["Ni"], info["N2"])
    assert info["N2"] in moving and info["O2"] in moving
    assert info["Ni"] not in moving
    assert info["N1"] not in moving and info["O1"] not in moving
    # the reporting colour centre hangs off arm 1, so it must not move either
    assert info["sp3_carbon"] not in moving


def test_salen_arm_torsions_reject_an_oxygen_donor() -> None:
    sensor, info = pentacene_sensor()
    with pytest.raises(ValueError, match="nitrogen"):
        salen_arm_torsions(sensor, info["Ni"], info["O1"])


@pytest.mark.parametrize("donor", ["N1", "N2"])
@pytest.mark.parametrize("target", DEGRADATION_TARGETS)
def test_open_salen_arm_hits_every_target(donor: str, target: float) -> None:
    sensor, info = pentacene_sensor()
    opened = open_salen_arm(sensor, info["Ni"], info[donor], target)
    assert opened.distance(info["Ni"], info[donor]) == pytest.approx(target, abs=1e-4)
    assert len(opened) == len(sensor)
    assert opened.symbols == sensor.symbols


@pytest.mark.parametrize("donor", ["N1", "N2"])
@pytest.mark.parametrize("target", DEGRADATION_TARGETS)
def test_open_salen_arm_preserves_every_bond_length(donor: str, target: float) -> None:
    """A translation-based scan collapses the imine N=C bond to 0.99 A instead."""
    sensor, info = pentacene_sensor()
    opened = open_salen_arm(sensor, info["Ni"], info[donor], target)
    for a, b in sensor.bonds():
        if info["Ni"] in (a, b):
            continue
        assert opened.distance(a, b) == pytest.approx(sensor.distance(a, b), abs=1e-9)
    # the bond most often corrupted by a bad scan, named explicitly
    imine = "C7b" if donor == "N2" else "C7a"
    assert 1.2 < opened.distance(info[imine], info[donor]) < 1.4


@pytest.mark.parametrize("donor", ["N1", "N2"])
@pytest.mark.parametrize("target", DEGRADATION_TARGETS)
def test_open_salen_arm_preserves_every_bond_angle(donor: str, target: float) -> None:
    """The check a length-only test cannot make.  Rotating about the two bridge
    *bonds* leaves all three atoms of every angle on the same side of every
    rotation, so the angles are exact, not merely reasonable."""
    sensor, info = pentacene_sensor()
    opened = open_salen_arm(sensor, info["Ni"], info[donor], target)
    # Angles at or through the metal are the coordinate itself and are expected
    # to change; the metal also sheds a neighbour once the donor clears the
    # bonding cutoff, so those angles stop existing.  Everything else is ligand
    # internal geometry and must survive untouched.
    before = {k: v for k, v in all_angles(sensor).items() if info["Ni"] not in k}
    after = {k: v for k, v in all_angles(opened).items() if info["Ni"] not in k}
    assert set(before) == set(after), "the ligand bond graph itself changed"
    assert before, "no angles were compared"
    for key, value in before.items():
        assert after[key] == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize("donor", ["N1", "N2"])
@pytest.mark.parametrize("target", DEGRADATION_TARGETS)
def test_open_salen_arm_leaves_no_clash(donor: str, target: float) -> None:
    """Scored on non-bonded contacts only.  The single rigid hinge this replaced
    reached 1.49 A here while reporting min_interatomic_distance = 1.08."""
    sensor, info = pentacene_sensor()
    opened = open_salen_arm(sensor, info["Ni"], info[donor], target)
    assert closest_non_bonded(opened) > 1.8
    assert opened.min_interatomic_distance() > 0.95


@pytest.mark.parametrize("donor", ["N1", "N2"])
def test_the_spectator_arm_stays_coordinated(donor: str) -> None:
    """Opening one arm must leave the other arm's two Ni bonds untouched, or
    the scan is measuring the loss of both arms at once."""
    sensor, info = pentacene_sensor()
    spectator = ("O2", "N2") if donor == "N1" else ("O1", "N1")
    for target in DEGRADATION_TARGETS:
        opened = open_salen_arm(sensor, info["Ni"], info[donor], target)
        for label in spectator:
            assert opened.distance(info["Ni"], info[label]) == pytest.approx(
                sensor.distance(info["Ni"], info[label]), abs=1e-9
            )


def test_the_opening_arm_takes_its_own_oxygen_with_it() -> None:
    """The mode is arm dissociation, not selective imine loss; assert it
    explicitly so nobody reads the scan as a pure Ni-N coordinate."""
    sensor, info = pentacene_sensor()
    opened = open_salen_arm(sensor, info["Ni"], info["N2"], 4.50)
    assert opened.distance(info["Ni"], info["O2"]) > 3.5


def test_open_salen_arm_returns_the_intact_complex_at_its_own_distance() -> None:
    """The first scan point is a tangency, not a crossing — the intact geometry
    is the *minimum* of the sweep — so a bracketing root finder misses it."""
    sensor, info = pentacene_sensor()
    opened = open_salen_arm(sensor, info["Ni"], info["N2"], 1.85)
    assert np.allclose(opened.coords, sensor.coords, atol=1e-9)


def test_open_salen_arm_refuses_the_impossible() -> None:
    sensor, info = pentacene_sensor()
    with pytest.raises(ValueError, match="reach|open"):
        open_salen_arm(sensor, info["Ni"], info["N1"], 25.0)


def test_salen_degradation_series_is_monotonic_and_physical() -> None:
    sensor, info = pentacene_sensor()
    series = salen_degradation_series(sensor, info["Ni"], info["N_labile"])
    distances = [d for d, _ in series]
    assert distances == sorted(distances)
    assert distances[0] < 2.0 < distances[-1]
    for target, geometry in series:
        assert geometry.distance(info["Ni"], info["N_labile"]) == pytest.approx(
            target, abs=1e-4
        )
        assert closest_non_bonded(geometry) > 1.8
        assert len(geometry) == len(sensor)


def test_the_series_walks_one_continuous_branch() -> None:
    """Solving each point independently lets the search jump torsion branches —
    measured: Ni-O of the departing arm went 5.05, 4.31, 6.40 A over three
    consecutive points.  The departing oxygen must recede monotonically with
    its own nitrogen, or the scan is not a coordinate."""
    sensor, info = pentacene_sensor()
    series = salen_degradation_series(sensor, info["Ni"], info["N_labile"])
    ni_o = [g.distance(info["Ni"], info["O_labile"]) for _, g in series]
    assert ni_o == sorted(ni_o)
    assert ni_o[0] == pytest.approx(1.85, abs=1e-6)
    assert ni_o[-1] > 6.0
