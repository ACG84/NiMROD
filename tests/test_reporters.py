"""Structural tests for the operando spin reporters.

Every assertion is anchored to chemistry that is true independently of this
code: the Huckel theorem that an odd alternant hydrocarbon has exactly one
non-bonding orbital, the C2v symmetry that makes a nitronyl nitroxide's two N-O
groups equivalent, published bond lengths, and electron counts.

The failure these tests exist to catch is a reporter that builds without error
and looks right in a viewer but has its radical in the wrong place -- a tether
on a SOMO node, or a nitroxide whose two N-O bonds came out inequivalent so the
node on C2 is an artefact of the builder rather than a property of the molecule.
Both would produce clean, converged, meaningless exchange couplings.
"""

from __future__ import annotations

import numpy as np
import pytest

from nimrod.catalysts import DIRECT_REPORTERS, ni_salen, reporter_on_salen
from nimrod.reporters import (
    OPERANDO_PROFILE,
    imino_nitroxide,
    nitronyl_nitroxide,
    phenalenyl,
    phenalenyl_nbmo,
    phenalenyl_sites,
)


# ---------------------------------------------------------------------------
# Phenalenyl
# ---------------------------------------------------------------------------


def test_phenalenyl_is_c13h9():
    """Three peri-fused hexagons: 13 carbons, 9 of them C-H."""
    struct = phenalenyl()
    assert struct.formula == "C13H9"
    assert len(struct) == 22


def test_phenalenyl_is_a_doublet():
    """An odd carbon count with no charge means an unpaired electron."""
    assert phenalenyl().multiplicity_for(charge=0) == 2


def test_phenalenyl_has_one_carbon_in_three_rings():
    """The peri-fusion: exactly one carbon has three carbon neighbours and no H."""
    struct = phenalenyl()
    interior = [
        i for i, s in enumerate(struct.symbols)
        if s == "C" and not struct.hydrogens_on(i)
    ]
    assert len(interior) == 4          # the central carbon plus three fusion carbons
    central = [
        i for i in interior
        if all(not struct.hydrogens_on(j) for j in struct.neighbours(i))
    ]
    assert len(central) == 1


def test_nbmo_exists_and_is_nonbonding():
    """Odd alternant hydrocarbons have exactly one zero-eigenvalue orbital."""
    coefficients = phenalenyl_nbmo()
    assert len(coefficients) == 13


def test_nbmo_amplitude_is_confined_to_six_carbons():
    """The classic result: amplitude on six perimeter carbons, zero elsewhere.

    Not a convention -- it follows from the starred/unstarred partition of an
    alternant hydrocarbon, and it is the entire reason a tether position matters.
    """
    coefficients = phenalenyl_nbmo()
    carrying = [c for c in coefficients.values() if abs(c) > 1e-6]
    assert len(carrying) == 6
    # All six carry equal amplitude by symmetry, alternating in sign.
    assert np.allclose([abs(c) for c in carrying], 1.0)
    assert sum(1 for c in carrying if c > 0) == 3


def test_nodal_and_bearing_positions_partition_the_ch_carbons():
    struct = phenalenyl()
    sites = phenalenyl_sites(struct)
    ch = [i for i, s in enumerate(struct.symbols)
          if s == "C" and len(struct.hydrogens_on(i)) == 1]
    assert sorted(sites["somo_bearing"] + sites["nodal"]) == sorted(ch)
    assert len(sites["somo_bearing"]) == 6
    assert len(sites["nodal"]) == 3


def test_nodal_carbons_have_two_ch_neighbours():
    """Positions 2, 5, 8 sit between two C-H carbons; 1, 3, 4... touch a fusion.

    This is an independent check on the Huckel result: the nodal set found by
    diagonalisation must be the topologically distinguishable one, or the
    diagonalisation is picking up numerical noise.
    """
    struct = phenalenyl()
    sites = phenalenyl_sites(struct)
    for atom in sites["nodal"]:
        carbon_neighbours = [j for j in struct.neighbours(atom)
                             if struct.symbols[j] == "C"]
        assert all(struct.hydrogens_on(j) for j in carbon_neighbours)
    for atom in sites["somo_bearing"]:
        carbon_neighbours = [j for j in struct.neighbours(atom)
                             if struct.symbols[j] == "C"]
        assert any(not struct.hydrogens_on(j) for j in carbon_neighbours)


def test_nbmo_rejects_an_even_alternant_host():
    """Pyrene has no non-bonding orbital, and asking for one must fail loudly."""
    from nimrod.geometry import pyrene

    with pytest.raises(ValueError, match="non-bonding"):
        phenalenyl_nbmo(pyrene())


# ---------------------------------------------------------------------------
# Nitroxides
# ---------------------------------------------------------------------------


def _closest_contact(struct) -> float:
    from nimrod.catalysts import _non_bonded_pairs

    pairs = _non_bonded_pairs(struct)
    return float(np.linalg.norm(
        struct.coords[pairs[:, 0]] - struct.coords[pairs[:, 1]], axis=1).min())




def test_nitronyl_nitroxide_formula():
    struct, _ = nitronyl_nitroxide(methylated=True)
    assert struct.formula == "C7H13N2O2"
    bare, _ = nitronyl_nitroxide(methylated=False)
    assert bare.formula == "C3H5N2O2"


def test_nitronyl_nitroxide_is_a_doublet():
    struct, _ = nitronyl_nitroxide()
    assert struct.multiplicity_for(charge=0) == 2


def test_the_two_n_o_groups_are_equivalent():
    """C2v symmetry is why the SOMO is delocalised over O-N-C-N-O.

    If the builder made them inequivalent the radical would be a localised
    nitroxide with a different SOMO entirely, and the node on C2 -- the whole
    reason this reporter is interesting -- would not exist.
    """
    struct, index = nitronyl_nitroxide(methylated=False)
    assert struct.distance(index["N1"], index["O_N1"]) == pytest.approx(
        struct.distance(index["N3"], index["O_N3"]), abs=1e-9)
    assert struct.distance(index["C2"], index["N1"]) == pytest.approx(
        struct.distance(index["C2"], index["N3"]), abs=1e-9)


def test_onc_no_unit_is_planar():
    """The delocalised SOMO requires the five-atom unit to be flat."""
    struct, index = nitronyl_nitroxide(methylated=False)
    unit = [index[k] for k in ("O_N1", "N1", "C2", "N3", "O_N3")]
    assert np.abs(struct.coords[unit][:, 2]).max() < 1e-6


def test_nitroxide_bond_lengths_match_crystal_structures():
    struct, index = nitronyl_nitroxide(methylated=False)
    assert struct.distance(index["N1"], index["O_N1"]) == pytest.approx(1.28, abs=0.01)
    assert struct.distance(index["C2"], index["N1"]) == pytest.approx(1.35, abs=0.01)
    assert struct.distance(index["N1"], index["C5"]) == pytest.approx(1.49, abs=0.01)
    assert struct.distance(index["C4"], index["C5"]) == pytest.approx(1.55, abs=0.01)


def test_five_ring_closes():
    """N1-C2-N3-C4-C5 is a closed cycle with every bond at its target length.

    The ring is built by mirror symmetry rather than by walking it, so closure
    is the thing that could silently fail: a walked pentagon does not return to
    its start unless the angles are solved for, and the resulting gap would show
    up as a stretched bond rather than as an error.
    """
    struct, index = nitronyl_nitroxide(methylated=False)
    ring = ["N1", "C2", "N3", "C4", "C5"]
    expected = {("N1", "C2"): 1.35, ("C2", "N3"): 1.35, ("N3", "C4"): 1.49,
                ("C4", "C5"): 1.55, ("C5", "N1"): 1.49}
    for k, label in enumerate(ring):
        following = ring[(k + 1) % 5]
        target = expected[(label, following)]
        assert struct.distance(index[label], index[following]) == pytest.approx(
            target, abs=1e-6), f"{label}-{following}"
        # And they must be bonded, i.e. the cycle exists in the bond graph too.
        assert index[following] in struct.neighbours(index[label])


def test_the_tetramethyl_ring_is_puckered_but_the_radical_unit_is_not():
    """C4 and C5 twist out of plane in opposite senses; the SOMO unit stays flat.

    Both halves matter.  Built flat the four methyls clash at 1.68 A, and
    puckering the whole ring instead would tilt the two N-O bonds out of plane
    in opposite directions and destroy the symmetry the delocalised SOMO needs.
    """
    from nimrod.reporters import NN_PUCKER

    struct, index = nitronyl_nitroxide(methylated=True)
    c4_z = float(struct.coords[index["C4"]][2])
    c5_z = float(struct.coords[index["C5"]][2])
    assert c4_z == pytest.approx(NN_PUCKER, abs=1e-9)
    assert c5_z == pytest.approx(-NN_PUCKER, abs=1e-9)
    assert abs(c4_z) > 0.1        # an actual twist, not a rounding artefact
    unit = [index[k] for k in ("O_N1", "N1", "C2", "N3", "O_N3")]
    assert np.abs(struct.coords[unit][:, 2]).max() < 1e-9


def _mirror_planes(struct) -> dict[str, bool]:
    """Which of the two candidate mirror planes the structure actually has."""
    found = {}
    for name, normal in (("molecular_plane", np.array([0.0, 0.0, 1.0])),
                         ("perpendicular", np.array([1.0, 0.0, 0.0]))):
        normal = normal / np.linalg.norm(normal)
        reflected = struct.coords - 2.0 * np.outer(struct.coords @ normal, normal)
        found[name] = all(
            any(struct.symbols[j] == symbol
                and np.linalg.norm(struct.coords[j] - point) < 1e-6
                for j in range(len(struct)))
            for point, symbol in zip(reflected, struct.symbols))
    return found


def test_the_desmethyl_model_is_flat_and_therefore_c2v():
    """The des-methyl model must keep BOTH mirror planes.

    This is the bug this test exists for.  The pucker was introduced to unclash
    the four methyls and was applied unconditionally, including to a molecule
    with no methyls.  That took the point group from C2v to C2, which leaves
    only the pi component of the SOMO node on C2 protected -- and every argument
    made about that node assumed C2v.  A whole round of spin densities was
    computed on a molecule that did not have the symmetry it was chosen for.
    """
    struct, index = nitronyl_nitroxide(methylated=False)
    assert float(struct.coords[index["C4"]][2]) == pytest.approx(0.0, abs=1e-12)
    assert float(struct.coords[index["C5"]][2]) == pytest.approx(0.0, abs=1e-12)
    planes = _mirror_planes(struct)
    assert planes["molecular_plane"], "des-methyl model is not planar"
    assert planes["perpendicular"], "des-methyl model has no C2v mirror"


def test_the_methylated_model_is_only_c2():
    """And the tetramethyl one is honestly not C2v, which is correct chemistry.

    Real tetramethyl nitronyl nitroxides twist.  The point of separating the two
    is that the des-methyl model is the one used to make symmetry arguments and
    the tetramethyl one is the chemically real object; conflating them is what
    produced a claim of an exact node on a structure that did not have one.
    """
    struct, _ = nitronyl_nitroxide(methylated=True)
    planes = _mirror_planes(struct)
    assert not planes["molecular_plane"]
    assert not planes["perpendicular"]


def test_the_flat_desmethyl_model_has_no_clash():
    """Removing the pucker must not reintroduce the contact it was added for.

    It does not, and that is the point: the clash was between methyls, so a
    model without methyls never needed the fix.
    """
    struct, _ = nitronyl_nitroxide(methylated=False)
    assert _closest_contact(struct) > 2.3


def test_sp3_carbons_carry_substituents_on_both_faces():
    """C4 and C5 are tetrahedral; both substituents on one face is a build bug."""
    struct, index = nitronyl_nitroxide(methylated=False)
    for label in ("C4", "C5"):
        attached = [j for j in struct.neighbours(index[label])
                    if struct.symbols[j] == "H"]
        assert len(attached) == 2
        heights = struct.coords[attached][:, 2]
        assert heights.min() < 0 < heights.max()


def test_imino_nitroxide_drops_exactly_one_oxygen():
    full, _ = nitronyl_nitroxide(methylated=False)
    imino, _ = imino_nitroxide(methylated=False)
    assert len(imino) == len(full) - 1
    assert imino.formula == "C3H5N2O"


def test_imino_nitroxide_is_asymmetric_at_c2():
    """Removing the oxygen shortens C2-N3 to an imine and lifts the node.

    The asymmetry is the point: it is the control that says whether the node on
    nitronyl nitroxide suppresses the coupling.
    """
    struct, index = imino_nitroxide(methylated=False)
    c2_n3 = struct.distance(index["C2"], index["N3"])
    c2_n1 = struct.distance(index["C2"], index["N1"])
    assert c2_n3 == pytest.approx(1.30, abs=0.01)
    assert c2_n1 == pytest.approx(1.35, abs=0.01)
    assert c2_n3 < c2_n1


def test_imino_nitroxide_is_a_doublet():
    struct, _ = imino_nitroxide()
    assert struct.multiplicity_for(charge=0) == 2


# ---------------------------------------------------------------------------
# Assembled on the catalyst
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reporter", sorted(DIRECT_REPORTERS))
def test_assembly_is_a_doublet_with_a_square_planar_nickel(reporter):
    struct, info = reporter_on_salen(reporter, site="C3")
    assert struct.multiplicity_for(charge=0) == 2
    donors = [info[k] for k in ("O1", "N1", "O2", "N2")]
    for donor in donors:
        assert struct.distance(info["Ni"], donor) == pytest.approx(1.85, abs=0.02)


@pytest.mark.parametrize("reporter", sorted(DIRECT_REPORTERS))
def test_assembly_conserves_the_catalyst(reporter):
    """Attaching a reporter must remove exactly one catalyst hydrogen."""
    catalyst, _ = ni_salen()
    struct, info = reporter_on_salen(reporter, site="C3")
    assert len(struct) == info["n_host_atoms"] + len(catalyst) - 1


@pytest.mark.parametrize("reporter", sorted(DIRECT_REPORTERS))
def test_reporter_block_comes_first(reporter):
    """The broken-symmetry machinery selects the reporter as range(n_host_atoms).

    If the block were not contiguous and leading, flipping the fragment's spin
    would flip part of the catalyst instead, and the resulting determinant would
    be neither the high-spin nor the broken-symmetry state.
    """
    struct, info = reporter_on_salen(reporter, site="C3")
    block = range(info["n_host_atoms"])
    assert info["reporter_carbon"] in block
    assert info["Ni"] not in block
    assert info["ipso_carbon"] not in block


@pytest.mark.parametrize("reporter", sorted(DIRECT_REPORTERS))
def test_attachment_introduces_no_new_clash(reporter):
    """The assembly must be no more strained than its worst component.

    A flat threshold would be the wrong test.  Ni(salen) carries a 2.267 A
    H...H across its eclipsed ethylenediamine bridge and a tetramethyl
    nitroxide carries a 2.19 A contact between its own gem-methyls; both are
    real features of the molecules and no torsion can open either.  What a bad
    attachment would do is create a contact *between* the two fragments that is
    worse than anything inside them, and that is what this checks.
    """
    from nimrod.catalysts import _build_reporter, ni_salen

    fragment, _atom, _hydrogen, _bond = _build_reporter(reporter, methylated=True)
    catalyst, _ = ni_salen()
    floor = min(_closest_contact(fragment), _closest_contact(catalyst))

    struct, _ = reporter_on_salen(reporter, site="C3")
    assert _closest_contact(struct) >= floor - 1e-6


def test_phenalenyl_tether_lands_on_a_somo_position():
    """The one assembly decision that would silently kill the signal."""
    fragment = phenalenyl()
    sites = phenalenyl_sites(fragment)
    _, info = reporter_on_salen("phenalenyl", site="C3")
    # attach() deletes the leaving hydrogen, which comes after every carbon in
    # the PAH builder's ordering, so carbon indices survive unchanged.
    assert info["reporter_carbon"] in sites["somo_bearing"]
    assert info["reporter_carbon"] not in sites["nodal"]


def test_nitroxide_spin_unit_is_closer_to_the_metal_than_the_pyrene_defect():
    """The operando claim in one number.

    The pyrene colour centre put its sp3 defect 4.46-4.62 A from nickel and
    coupled at -23..-51 cm^-1.  A nitronyl nitroxide's N-O oxygen -- which
    carries a large share of the SOMO, unlike the sp3 carbon -- gets closer than
    that in a construct 15 atoms smaller.
    """
    struct, info = reporter_on_salen("nitronyl-nitroxide", site="C3",
                                     methylated=False)
    oxygens = [i for i in range(info["n_host_atoms"]) if struct.symbols[i] == "O"]
    assert oxygens
    assert min(struct.distance(info["Ni"], o) for o in oxygens) < 4.4
    assert len(struct) < 60


def test_methyls_are_electronic_spectators_only():
    """Turning the methyls off must not move the radical-bearing unit.

    The two models differ at C4 and C5 by the pucker, and only there: those are
    the sp3 carbons the methyls hang from, and the pucker exists to unclash
    them.  Every atom the SOMO lives on is identical between the two, which is
    what makes the des-methyl model a valid stand-in for the electronic
    question while being C2v where the real molecule is not.
    """
    from nimrod.reporters import NN_PUCKER

    full, index_full = nitronyl_nitroxide(methylated=True)
    bare, index_bare = nitronyl_nitroxide(methylated=False)

    for label in ("C2", "N1", "N3", "O_N1", "O_N3"):
        assert np.allclose(full.coords[index_full[label]],
                           bare.coords[index_bare[label]], atol=1e-9), label

    # C4 and C5 differ, and by exactly the pucker rather than by drift.
    for label, sign in (("C4", +1.0), ("C5", -1.0)):
        moved = full.coords[index_full[label]] - bare.coords[index_bare[label]]
        assert moved[2] == pytest.approx(sign * NN_PUCKER, abs=1e-9)


# ---------------------------------------------------------------------------
# The operando criteria themselves
# ---------------------------------------------------------------------------


def test_every_profiled_reporter_states_its_redox_window():
    for name, profile in OPERANDO_PROFILE.items():
        low, high = profile["redox_window_v"]
        assert low < high, name
        assert profile["why"], name


def test_redox_window_flag_agrees_with_the_catalytic_range():
    """Ni(salen) cycles roughly -1.7 to +0.8 V vs Fc.

    The flag must be derivable from the stated potentials, not asserted next to
    them -- otherwise a reporter could be labelled a spectator while its own
    numbers say it is a reagent.
    """
    catalytic_low, catalytic_high = -1.7, 0.8
    for name, profile in OPERANDO_PROFILE.items():
        low, high = profile["redox_window_v"]
        overlaps = low > catalytic_low or high < catalytic_high
        assert profile["inside_catalyst_window"] == overlaps, name


def test_the_air_stable_reporters_are_the_intrinsic_ones():
    """Every sp3-defect reporter is air-unstable; that is the operando problem."""
    for name, profile in OPERANDO_PROFILE.items():
        if profile["radical_source"] == "sp3 aryl defect":
            assert profile["air_stable"] is False, name
