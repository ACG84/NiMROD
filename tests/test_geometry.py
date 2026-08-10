"""Structural tests for the NiMROD builders.

These run without Psi4 and are the guard rail on the part of the project that is
easiest to get silently wrong: atom indices and connectivity.  Every assertion
here is anchored to known chemistry rather than to a previous run of the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nimrod.geometry import (
    PAH_RINGS,
    Structure,
    arm_atoms,
    attach,
    degradation_series,
    elongate_bond,
    ni_salen_model,
    named_pah,
    occ_radical,
    phenyl,
    pyrene,
    rotation_between,
    sensor_assembly,
)

# Molecular formulas from chemistry, not from the code.
KNOWN_PAH_FORMULAS = {
    "benzene": "C6H6",
    "naphthalene": "C10H8",
    "anthracene": "C14H10",
    "phenanthrene": "C14H10",
    "pyrene": "C16H10",
    "coronene": "C24H12",
}


@pytest.mark.parametrize("name,formula", sorted(KNOWN_PAH_FORMULAS.items()))
def test_pah_formula_matches_chemistry(name: str, formula: str) -> None:
    assert named_pah(name).formula == formula


@pytest.mark.parametrize("name", sorted(PAH_RINGS))
def test_pah_is_planar_and_unclashed(name: str) -> None:
    struct = named_pah(name)
    assert struct.min_interatomic_distance() > 1.0
    assert np.allclose(struct.coords[:, 2], 0.0, atol=1e-9)


@pytest.mark.parametrize("name", sorted(PAH_RINGS))
def test_pah_carbon_valences(name: str) -> None:
    """Every aromatic carbon has 2 or 3 carbon neighbours; CH carbons have one H."""
    struct = named_pah(name)
    for i, symbol in enumerate(struct.symbols):
        if symbol != "C":
            continue
        carbons = [j for j in struct.neighbours(i) if struct.symbols[j] == "C"]
        hydrogens = struct.hydrogens_on(i)
        assert len(carbons) in (2, 3)
        # A carbon shared by three rings is internal and carries no hydrogen.
        assert len(hydrogens) == (1 if len(carbons) == 2 else 0)


def test_pyrene_has_two_internal_carbons() -> None:
    """Pyrene is peri-fused: exactly two carbons belong to three rings."""
    struct = pyrene()
    internal = [
        i
        for i, s in enumerate(struct.symbols)
        if s == "C" and len([j for j in struct.neighbours(i) if struct.symbols[j] == "C"]) == 3
    ]
    assert len(internal) == 6  # 2 peri + 4 ring-fusion carbons
    assert len(struct.hydrogens_on(internal[0])) == 0


def test_rotation_between_handles_antiparallel() -> None:
    a = np.array([0.0, 0.0, 1.0])
    rot = rotation_between(a, -a)
    assert np.allclose(rot @ a, -a, atol=1e-10)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)


def test_rotation_between_is_a_proper_rotation() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        rot = rotation_between(a, b)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)
        mapped = rot @ (a / np.linalg.norm(a))
        assert np.allclose(mapped, b / np.linalg.norm(b), atol=1e-10)


# --------------------------------------------------------------------------
# Organic colour centre
# --------------------------------------------------------------------------


def test_occ_is_a_doublet_radical() -> None:
    """Pyrene + phenyl gives C22H15: odd electron count, hence S = 1/2."""
    occ, _ = occ_radical()
    assert occ.formula == "C22H15"
    assert occ.n_electrons % 2 == 1
    assert occ.multiplicity_for(0) == 2


def test_occ_defect_carbon_is_sp3() -> None:
    occ, info = occ_radical()
    neighbours = occ.neighbours(info["sp3_carbon"])
    assert len(neighbours) == 4
    symbols = sorted(occ.symbols[j] for j in neighbours)
    assert symbols == ["C", "C", "C", "H"]


def test_occ_has_no_clashes() -> None:
    occ, _ = occ_radical()
    assert occ.min_interatomic_distance() > 0.95


def test_occ_defect_carbon_is_pyramidalised() -> None:
    """The defect carbon must be a genuine sp3 centre, not a planar sp2 one.

    Three independent signatures are checked: the H and the aryl group sit on
    opposite faces of the aromatic plane, the H-C-C(aryl) angle is tetrahedral,
    and the carbon itself is displaced off the plane of the remaining pi system.
    """
    from nimrod.geometry import plane_normal

    occ, info = occ_radical()
    sp3 = info["sp3_carbon"]
    ipso = info["ipso_carbon"]

    # Plane of the aromatic carbons that are still sp2.
    aromatic = np.array(
        [
            occ.coords[i]
            for i, s in enumerate(occ.symbols)
            if s == "C" and i not in (sp3, ipso) and i < info["n_host_atoms"]
        ]
    )
    normal = plane_normal(aromatic)
    origin = aromatic.mean(axis=0)

    hydrogen = next(j for j in occ.neighbours(sp3) if occ.symbols[j] == "H")
    h_side = np.dot(occ.coords[hydrogen] - origin, normal)
    aryl_side = np.dot(occ.coords[ipso] - origin, normal)
    assert h_side * aryl_side < 0, "H and aryl must straddle the aromatic plane"

    v_h = occ.coords[hydrogen] - occ.coords[sp3]
    v_aryl = occ.coords[ipso] - occ.coords[sp3]
    cosine = np.dot(v_h, v_aryl) / (np.linalg.norm(v_h) * np.linalg.norm(v_aryl))
    angle = math.degrees(math.acos(np.clip(cosine, -1, 1)))
    assert 100.0 < angle < 120.0, f"H-C-aryl angle {angle:.1f} is not tetrahedral"

    displacement = abs(np.dot(occ.coords[sp3] - origin, normal))
    assert displacement > 0.01, "defect carbon has not left the aromatic plane"


# --------------------------------------------------------------------------
# Catalyst
# --------------------------------------------------------------------------


def test_catalyst_is_square_planar_nin2o2() -> None:
    cat, idx = ni_salen_model()
    assert cat.formula == "C6H8N2NiO2"
    donors = [cat.symbols[j] for j in cat.neighbours(idx["Ni"])]
    assert sorted(donors) == ["N", "N", "O", "O"]
    assert np.allclose(cat.coords[:, 2], 0.0, atol=1e-9)


def test_catalyst_is_closed_shell() -> None:
    cat, _ = ni_salen_model()
    assert cat.n_electrons % 2 == 0
    assert cat.multiplicity_for(0) == 1


def test_catalyst_donors_are_trans() -> None:
    """The two nitrogens sit trans across the metal, as do the two oxygens."""
    cat, idx = ni_salen_model()
    ni = cat.coords[idx["Ni"]]
    for a, b in (("N1", "N2"), ("O1", "O2")):
        va = cat.coords[idx[a]] - ni
        vb = cat.coords[idx[b]] - ni
        cosine = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
        assert math.degrees(math.acos(np.clip(cosine, -1, 1))) > 170.0


def test_catalyst_bite_angle_is_chemically_sensible() -> None:
    cat, idx = ni_salen_model()
    ni = cat.coords[idx["Ni"]]
    vo = cat.coords[idx["O1"]] - ni
    vn = cat.coords[idx["N1"]] - ni
    cosine = np.dot(vo, vn) / (np.linalg.norm(vo) * np.linalg.norm(vn))
    angle = math.degrees(math.acos(np.clip(cosine, -1, 1)))
    assert 85.0 < angle < 100.0


def test_chelate_ring_is_six_membered() -> None:
    """Ni-O-Ca-Cb-Cc-N must close into a six-membered ring."""
    cat, idx = ni_salen_model()
    path = ["Ni", "O1", "Ca1", "Cb1", "Cc1", "N1"]
    for a, b in zip(path, path[1:] + ["Ni"]):
        assert idx[b] in cat.neighbours(idx[a]), f"{a}-{b} not bonded"


# --------------------------------------------------------------------------
# Assembly and degradation
# --------------------------------------------------------------------------


def test_sensor_assembly_composition() -> None:
    sensor, _ = sensor_assembly()
    assert sensor.formula == "C28H21N2NiO2"
    assert len(sensor) == 54
    assert sensor.multiplicity_for(0) == 2  # the colour-centre radical survives


@pytest.mark.parametrize(
    "label,symbol,degree",
    [
        ("Ni", "Ni", 4),
        ("sp3_carbon", "C", 4),
        ("N_labile", "N", 3),
        ("N_spectator", "N", 3),
        ("O_labile", "O", 2),
        ("O_spectator", "O", 2),
        ("meso_carbon", "C", 3),
        ("tether_carbon", "C", 3),
        ("ipso_carbon", "C", 3),
    ],
)
def test_sensor_assembly_indices(label: str, symbol: str, degree: int) -> None:
    """Guards the index bookkeeping through two successive fragment joins."""
    sensor, info = sensor_assembly()
    index = info[label]
    assert sensor.symbols[index] == symbol
    assert len(sensor.neighbours(index)) == degree


def test_labile_arm_is_not_the_tethered_arm() -> None:
    """Degradation must happen across the metal from the reporting defect,
    so the signal travels through the metal rather than along the tether."""
    sensor, info = sensor_assembly()
    meso_neighbours = set(sensor.neighbours(info["meso_carbon"]))
    # The labile nitrogen's chelate arm must not contain the tether meso carbon.
    labile_arm = set()
    frontier = [info["N_labile"]]
    seen = {info["Ni"]}
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        labile_arm.add(current)
        frontier.extend(sensor.neighbours(current))
    assert info["meso_carbon"] not in labile_arm
    assert info["sp3_carbon"] not in labile_arm


def test_elongate_bond_hits_the_target_distance() -> None:
    sensor, info = sensor_assembly()
    carry = arm_atoms(sensor, info["Ni"], info["N_labile"])
    for target in (2.2, 3.0, 4.5):
        moved = elongate_bond(sensor, info["Ni"], info["N_labile"], target, carry=carry)
        assert moved.distance(info["Ni"], info["N_labile"]) == pytest.approx(target, abs=1e-6)


def test_arm_atoms_carries_nitrogen_and_its_hydrogens_only() -> None:
    sensor, info = sensor_assembly()
    carried = arm_atoms(sensor, info["Ni"], info["N_labile"])
    assert carried[0] == info["N_labile"]
    assert all(sensor.symbols[i] == "H" for i in carried[1:])


def test_degradation_series_is_monotonic() -> None:
    sensor, info = sensor_assembly()
    carry = arm_atoms(sensor, info["Ni"], info["N_labile"])
    series = degradation_series(sensor, info["Ni"], info["N_labile"], carry=carry)
    distances = [d for d, _ in series]
    assert distances == sorted(distances)
    assert distances[0] < 2.0 < distances[-1]
    for target, geom in series:
        assert geom.distance(info["Ni"], info["N_labile"]) == pytest.approx(target, abs=1e-6)
        assert len(geom) == len(sensor)


# --------------------------------------------------------------------------
# Round-tripping
# --------------------------------------------------------------------------


def test_psi4_block_roundtrip() -> None:
    sensor, _ = sensor_assembly()
    restored = Structure.from_psi4(sensor.to_psi4())
    assert restored.symbols == sensor.symbols
    assert np.allclose(restored.coords, sensor.coords, atol=1e-7)


def test_xyz_roundtrip() -> None:
    occ, _ = occ_radical()
    restored = Structure.from_xyz(occ.to_xyz("test comment"))
    assert restored.symbols == occ.symbols
    assert np.allclose(restored.coords, occ.coords, atol=1e-7)


def test_attach_rejects_deleted_index_tracking() -> None:
    """Tracking an atom that the join deletes must fail loudly, not silently."""
    base = pyrene()
    site = next(i for i, s in enumerate(base.symbols) if s == "C" and base.hydrogens_on(i))
    hydrogen = base.hydrogens_on(site)[0]
    with pytest.raises(ValueError):
        attach(
            base,
            base_atom=site,
            base_hydrogen=hydrogen,
            fragment=phenyl(),
            fragment_atom=0,
            fragment_hydrogen=1,
            track_base={"doomed": hydrogen},
        )


# --------------------------------------------------------------------------
# Opening the chelate arm
# --------------------------------------------------------------------------
#
# Regression guard for a defect that produced physically impossible geometries
# while looking entirely plausible in a plot.  Translating the imine nitrogen
# along the Ni-N axis compresses the N=C bond it is ring-bonded to, down to
# 0.99 A at intermediate separations.  Since the ligand field at the metal is
# what drives the spin flip, that silently corrupts the whole scan.


def test_translation_really_does_compress_the_imine_bond() -> None:
    """Documents *why* swing_arm exists; if this ever stops being true the
    hinge machinery can be reconsidered."""
    catalyst, index = ni_salen_model()
    ni, nitrogen, carbon = index["Ni"], index["N1"], index["Cc1"]
    carry = arm_atoms(catalyst, ni, nitrogen)
    squashed = elongate_bond(catalyst, ni, nitrogen, 2.60, carry=carry)
    assert squashed.distance(nitrogen, carbon) < 1.05


@pytest.mark.parametrize("target", [1.87, 2.10, 2.35, 2.60, 2.90, 3.30, 3.80, 4.50])
def test_swing_arm_keeps_the_geometry_physical(target: float) -> None:
    from nimrod.geometry import swing_arm

    catalyst, index = ni_salen_model()
    ni, nitrogen, carbon = index["Ni"], index["N1"], index["Cc1"]
    opened = swing_arm(catalyst, ni, nitrogen, target)

    assert opened.distance(ni, nitrogen) == pytest.approx(target, abs=1e-4)
    # The imine bond must survive untouched.
    assert opened.distance(nitrogen, carbon) == pytest.approx(
        catalyst.distance(nitrogen, carbon), abs=1e-6
    )
    # And the arm must not have swung through the trans ligand.
    assert opened.min_interatomic_distance() > 0.95


@pytest.mark.parametrize("target", [2.35, 3.30, 4.50])
def test_swing_arm_is_a_rigid_body_rotation(target: float) -> None:
    """Every bond not involving the metal is preserved to machine precision."""
    from nimrod.geometry import swing_arm

    catalyst, index = ni_salen_model()
    ni, nitrogen = index["Ni"], index["N1"]
    opened = swing_arm(catalyst, ni, nitrogen, target)
    for a, b in catalyst.bonds():
        if ni in (a, b):
            continue
        assert opened.distance(a, b) == pytest.approx(catalyst.distance(a, b), abs=1e-9)


def test_swing_arm_works_on_the_full_assembly() -> None:
    from nimrod.geometry import swing_arm

    sensor, info = sensor_assembly()
    ni, nitrogen = info["Ni"], info["N_labile"]
    opened = swing_arm(sensor, ni, nitrogen, 4.50)
    assert opened.distance(ni, nitrogen) == pytest.approx(4.50, abs=1e-4)
    assert opened.min_interatomic_distance() > 0.95
    assert len(opened) == len(sensor)


def test_chelate_hinge_pivots_on_the_far_side_of_the_ring() -> None:
    from nimrod.geometry import chelate_hinge

    catalyst, index = ni_salen_model()
    pivot, axis, moving = chelate_hinge(catalyst, index["Ni"], index["N1"])
    assert pivot == index["Ca1"]
    assert index["N1"] in moving
    assert index["Cb1"] in moving and index["Cc1"] in moving
    # The metal, the oxygen and the spectator arm must all stay put.
    for label in ("Ni", "O1", "N2", "O2"):
        assert index[label] not in moving
    assert np.isclose(np.linalg.norm(axis), 1.0)


def test_swing_arm_refuses_unreachable_targets() -> None:
    from nimrod.geometry import swing_arm

    catalyst, index = ni_salen_model()
    with pytest.raises(ValueError, match="reach|hinge"):
        swing_arm(catalyst, index["Ni"], index["N1"], 25.0)
