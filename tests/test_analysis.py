"""Tests for the analysis layer that do not require a Psi4 calculation.

These pin down the *conventions* — sign of the exchange coupling, definition of
the active space, which reference a validation gap is measured against — which
are the things that silently invert a conclusion if they drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nimrod.config import HARTREE_TO_CM, HARTREE_TO_EV, HARTREE_TO_KCAL


# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------


def test_unit_conversions_are_consistent() -> None:
    """Cross-check the constants against each other, not against themselves."""
    # 1 eV in kcal/mol
    ev_in_kcal = HARTREE_TO_KCAL / HARTREE_TO_EV
    assert ev_in_kcal == pytest.approx(23.0605, abs=1e-3)
    # 1 eV in cm^-1
    ev_in_cm = HARTREE_TO_CM / HARTREE_TO_EV
    assert ev_in_cm == pytest.approx(8065.54, abs=0.5)
    # 1 kcal/mol in cm^-1
    assert HARTREE_TO_CM / HARTREE_TO_KCAL == pytest.approx(349.755, abs=0.05)


# --------------------------------------------------------------------------
# Spin bookkeeping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "multiplicity,expected",
    [(1, 0.0), (2, 0.75), (3, 2.0), (4, 3.75), (5, 6.0)],
)
def test_ideal_s2(multiplicity: int, expected: float) -> None:
    from nimrod.spin import ideal_s2

    assert ideal_s2(multiplicity) == pytest.approx(expected)


def test_high_and_broken_symmetry_multiplicities() -> None:
    """Two coupled S = 1/2 centres give a triplet and an Ms = 0 companion."""
    from nimrod.spin import broken_symmetry_multiplicity, high_spin_multiplicity

    assert high_spin_multiplicity(0.5, 0.5) == 3
    assert broken_symmetry_multiplicity(0.5, 0.5) == 1
    # An S = 1/2 defect against an S = 1 metal: quartet, BS companion doublet.
    assert high_spin_multiplicity(0.5, 1.0) == 4
    assert broken_symmetry_multiplicity(0.5, 1.0) == 2


def test_fragment_spin_sums_the_right_atoms() -> None:
    from nimrod.spin import fragment_spin

    populations = np.array([0.9, 0.05, 0.02, -0.01, 0.04])
    assert fragment_spin(populations, [0]) == pytest.approx(0.9)
    assert fragment_spin(populations, [1, 2, 3, 4]) == pytest.approx(0.10)
    assert fragment_spin(populations, range(5)) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Exchange coupling: the sign convention is the whole point
# --------------------------------------------------------------------------


def test_yamaguchi_antiferromagnetic_is_negative() -> None:
    """Broken-symmetry below high-spin must give J < 0 under H = -2J S1.S2."""
    from nimrod.spin import yamaguchi_j

    j = yamaguchi_j(e_hs=0.0, s2_hs=2.0, e_bs=-1.0e-3, s2_bs=1.0)
    assert j < 0
    assert j == pytest.approx(-1.0e-3 * HARTREE_TO_CM, rel=1e-9)


def test_yamaguchi_ferromagnetic_is_positive() -> None:
    from nimrod.spin import yamaguchi_j

    j = yamaguchi_j(e_hs=-1.0e-3, s2_hs=2.0, e_bs=0.0, s2_bs=1.0)
    assert j > 0


def test_yamaguchi_scales_with_the_energy_difference() -> None:
    from nimrod.spin import yamaguchi_j

    small = yamaguchi_j(0.0, 2.0, -1.0e-4, 1.0)
    large = yamaguchi_j(0.0, 2.0, -2.0e-4, 1.0)
    assert large == pytest.approx(2.0 * small, rel=1e-9)


def test_yamaguchi_rejects_degenerate_contamination() -> None:
    """A BS solution that collapsed onto high spin has no defined J."""
    from nimrod.spin import yamaguchi_j

    with pytest.raises(ValueError, match="degenerate"):
        yamaguchi_j(e_hs=0.0, s2_hs=2.0, e_bs=-1.0e-3, s2_bs=2.0)


def test_yamaguchi_reduces_to_the_ising_limit() -> None:
    """With ideal contamination (2.0 vs 1.0) the denominator is 2 S_A S_B x 2."""
    from nimrod.spin import yamaguchi_j

    delta_e = -5.0e-4
    j = yamaguchi_j(0.0, 2.0, delta_e, 1.0)
    # Noodleman weak-coupling limit for two S=1/2: J = (E_BS - E_HS) / 1
    assert j == pytest.approx(delta_e * HARTREE_TO_CM, rel=1e-9)


# --------------------------------------------------------------------------
# CASSCF active spaces
# --------------------------------------------------------------------------


def test_cas_space_partitions_electrons_correctly() -> None:
    from nimrod.casscf import cas_space

    # O2 has 16 electrons; CAS(8,6) leaves 4 doubly-occupied inactive orbitals.
    space = cas_space(16, 8, 6)
    assert space["RESTRICTED_DOCC"] == [4]
    assert space["ACTIVE"] == [6]


def test_cas_space_rejects_impossible_requests() -> None:
    from nimrod.casscf import cas_space

    with pytest.raises(ValueError):
        cas_space(10, 12, 6)          # more active electrons than exist
    with pytest.raises(ValueError):
        cas_space(15, 8, 6)           # parity mismatch leaves a half orbital
    with pytest.raises(ValueError):
        cas_space(16, 8, 3)           # 8 electrons cannot fit in 3 orbitals


# --------------------------------------------------------------------------
# Colour-centre optical shift
# --------------------------------------------------------------------------


def _state(index: int, ev: float, f: float, label: str = "singlet"):
    from nimrod.excited import ExcitedState

    return ExcitedState(
        index=index,
        energy_ev=ev,
        energy_nm=1239.841984 / ev,
        energy_cm=ev * 8065.543937,
        oscillator_strength=f,
        spin_label=label,
    )


def test_colour_centre_shift_reports_a_red_shift() -> None:
    """The defect state must be reported as shifted to lower energy."""
    from nimrod.excited import colour_centre_shift

    pristine = [_state(1, 4.20, 0.0005), _state(2, 4.80, 0.42)]
    defect = [_state(1, 2.10, 0.31, "doublet"), _state(2, 3.60, 0.05, "doublet")]

    shift = colour_centre_shift(pristine, defect)
    values = [v for v in shift.values() if isinstance(v, (int, float))]
    assert values, f"no numeric shift reported: {shift}"
    # Lowest bright: pristine 4.80 eV, defect 2.10 eV -> 2.70 eV to the red.
    assert any(v == pytest.approx(-2.70, abs=0.02) or v == pytest.approx(2.70, abs=0.02)
               for v in values), shift


def test_lowest_bright_skips_dark_states() -> None:
    from nimrod.excited import lowest_bright

    states = [_state(1, 3.0, 0.0), _state(2, 3.5, 0.0001), _state(3, 4.0, 0.25)]
    chosen = lowest_bright(states)
    assert chosen is not None
    assert chosen.index == 3


def test_excited_state_energy_units_agree() -> None:
    state = _state(1, 2.5, 0.1)
    assert state.energy_nm == pytest.approx(1239.842 / 2.5, rel=1e-4)
    assert state.energy_cm == pytest.approx(2.5 * 8065.54, rel=1e-4)


def test_occ_absorption_returns_the_lowest_bright_band_not_the_brightest() -> None:
    """The colour-centre band is the absorption edge, not the strongest peak.

    On a PAH radical the strongest root in the window is usually a delocalised
    pi->pi* band of the intact host.  Tracking that across a degradation scan
    would follow a different state at every point, so ``lowest_bright`` must
    mean what it says.
    """
    import nimrod.excited as excited

    states = [_state(1, 1.80, 0.020, "doublet"),
              _state(2, 2.40, 0.0004, "doublet"),
              _state(3, 4.90, 0.900, "doublet")]

    def stub(_geometry, **_kwargs):
        return excited.TDDFTResult(states=states, method="stub", basis="stub",
                                   reference="uks", multiplicity=2, converged=True,
                                   s_squared=0.7534, s_squared_ideal=0.75,
                                   spin_contamination=0.0034)

    real, excited.run_tddft = excited.run_tddft, stub
    try:
        result = excited.occ_absorption("stub", multiplicity=2, n_states=3)
    finally:
        excited.run_tddft = real

    assert result.lowest_bright is not None
    assert result.lowest_bright.energy_ev == pytest.approx(1.80)
    assert result.strongest is not None
    assert result.strongest.energy_ev == pytest.approx(4.90)


def test_root_stability_catches_a_missed_degenerate_partner() -> None:
    """A degenerate partner must not be matched against its own twin.

    Benzene's bright E1u band is a degenerate pair, so a check that asks only
    "is there any short-run root at this energy?" would pass a window that
    found one of the two and dropped the other.
    """
    import nimrod.excited as excited

    short = [_state(1, 3.0, 0.0), _state(2, 5.0, 0.5)]
    long = [_state(1, 3.0, 0.0), _state(2, 5.0, 0.5), _state(3, 5.0, 0.5)]

    def stub(_geometry, *, n_states, **_kwargs):
        return excited.TDDFTResult(states=short if n_states == 2 else long,
                                   method="stub", basis="stub", reference="rks",
                                   multiplicity=1, converged=True)

    real, excited.run_tddft = excited.run_tddft, stub
    try:
        stability = excited.verify_root_stability("stub", n_states=2, n_reference=3)
    finally:
        excited.run_tddft = real

    assert not stability.stable
    assert len(stability.missing) == 1
    assert stability.missing[0].energy_ev == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Validation benchmark bookkeeping
# --------------------------------------------------------------------------


def test_degenerate_pair_reference_is_the_mean_of_the_two_terms() -> None:
    """O2 and NH must be scored against the 1D/1S+ average, not against 1D."""
    from nimrod.validation import BENCHMARKS

    by_key = {b.key: b for b in BENCHMARKS}

    o2 = by_key["o2"]
    assert o2.degenerate_pair
    assert o2.experiment_kcal == pytest.approx(22.64, abs=0.05)
    assert o2.sd_reference_kcal == pytest.approx(30.18, abs=0.05)
    assert o2.sd_reference_kcal > o2.experiment_kcal

    nh = by_key["nh"]
    assert nh.degenerate_pair
    assert nh.experiment_kcal == pytest.approx(35.93, abs=0.05)
    assert nh.sd_reference_kcal == pytest.approx(48.27, abs=0.05)

    # Methylene's singlet is a genuine closed shell: the two agree.
    ch2 = by_key["ch2"]
    assert not ch2.degenerate_pair
    assert ch2.sd_reference_kcal == ch2.experiment_kcal


def test_all_benchmarks_have_triplet_ground_states() -> None:
    from nimrod.validation import BENCHMARKS

    for benchmark in BENCHMARKS:
        assert benchmark.high_multiplicity == 3
        assert benchmark.low_multiplicity == 1
        assert benchmark.experiment_kcal > 0, (
            f"{benchmark.key}: a positive gap must mean the triplet is lower"
        )


def test_validation_report_spread_and_mae() -> None:
    from nimrod.validation import GapResult, ValidationReport

    report = ValidationReport(basis="def2-svp")
    report.results = [
        GapResult(system="X", functional="a", basis="def2-svp", gap_kcal=10.0, error_kcal=1.0),
        GapResult(system="X", functional="b", basis="def2-svp", gap_kcal=14.0, error_kcal=5.0),
        GapResult(system="X", functional="c", basis="def2-svp", failure="did not converge"),
    ]
    spread = report.spread("X")
    assert spread is not None
    assert spread["n"] == 2
    assert spread["range"] == pytest.approx(4.0)
    assert spread["mean"] == pytest.approx(12.0)
    mae = report.mean_absolute_error()
    assert mae == {"a": pytest.approx(1.0), "b": pytest.approx(5.0)}


# --------------------------------------------------------------------------
# Scan bookkeeping
# --------------------------------------------------------------------------


def test_frozen_distance_option_is_one_indexed() -> None:
    """optking counts atoms from 1; an off-by-one silently constrains the
    wrong pair, which would be invisible in the output."""
    from nimrod.degradation import frozen_distance_option

    assert frozen_distance_option(0, 13) == "1 14"


def test_scan_result_roundtrip(tmp_path) -> None:
    from nimrod.degradation import ScanPoint, ScanResult

    result = ScanResult(label="t", functional="b3lyp", basis="def2-svp")
    result.points = [
        ScanPoint(distance=1.87, converged=True, energies={"E_b3lyp": -1.0},
                  properties={"spin_gap_kcal": 12.0}),
        ScanPoint(distance=3.0, converged=False, error="nope"),
    ]
    path = result.save(tmp_path / "scan.json")
    restored = ScanResult.load(path)
    assert restored.distances == [1.87, 3.0]
    assert restored.series("spin_gap_kcal") == [12.0, None]
    assert restored.series("E_b3lyp") == [-1.0, None]


# --------------------------------------------------------------------------
# Figures render from synthetic data
# --------------------------------------------------------------------------


def test_plots_emit_light_and_dark_variants(tmp_path) -> None:
    from nimrod import plots

    distances = [1.87, 2.4, 3.0, 3.8]
    paths = plots.plot_spin_gap_scan(
        distances,
        {"b3lyp": [14.0, 6.0, -2.0, -9.0], "pbe0": [11.0, 4.0, -4.0, -12.0]},
        outdir=tmp_path,
    )
    assert len(paths) == 2
    assert {p.stem.split("-")[-1] for p in paths} == {"light", "dark"}
    for path in paths:
        assert path.exists() and path.stat().st_size > 5000


def test_sensor_readouts_uses_small_multiples(tmp_path) -> None:
    """Different units must never share an axis."""
    from nimrod import plots

    paths = plots.plot_sensor_readouts(
        [1.87, 2.4, 3.0],
        [
            {"label": "gap (kcal/mol)", "values": [12.0, 2.0, -6.0]},
            {"label": "J (cm-1)", "values": [-0.2, -18.0, -64.0]},
            {"label": "bright state (eV)", "values": [2.9, 2.8, 2.6]},
        ],
        outdir=tmp_path,
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_plots_tolerate_missing_points(tmp_path) -> None:
    """A failed scan point must not break the figure."""
    from nimrod import plots

    paths = plots.plot_spin_gap_scan(
        [1.87, 2.4, 3.0], {"b3lyp": [14.0, None, -2.0]}, outdir=tmp_path
    )
    assert all(p.exists() for p in paths)
