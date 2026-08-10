"""Smoke test for :mod:`nimrod.excited`.

Every system here is tiny (four atoms or twelve, small Pople basis) and every
number is checked against a value that is known independently of this code, so
a regression shows up as a failed assertion rather than as a plausible-looking
wrong number.

What is checked, and why that reference value:

``H2CO`` n->pi*
    The textbook dark transition.  Vertical experiment is 4.07 eV; TD-B3LYP in
    a small Pople basis lands a little low, ~3.8-4.0 eV.  Its oscillator
    strength must be ~0 (the transition is symmetry-forbidden in C2v), which is
    also the check that ``f`` parses as a real float and not ``None``.

``H2CO`` singlet/triplet
    With ``triplets="ALSO"`` the lowest root must be the *triplet* n->pi*, well
    below the singlet.  This is the test that spin labels are read from Psi4
    rather than guessed.

``benzene``
    Lowest root ~5.0-5.5 eV (the forbidden 1B2u band; experiment 4.90 eV).
    Also exercises a 12-atom molecule and the dominant-orbital-pair analysis.

``H2O`` EOM-CCSD
    Lowest root ~7.5-8.0 eV (experiment 7.4 eV).

``CH3`` radical
    A doublet, standing in for the colour centre: exercises the UKS path,
    :func:`occ_absorption`, and the spin-contamination report.  Never run the
    54-atom sensor from a smoke test.

Run with::

    cd /home/user/NiMROD && MAMBA_ROOT_PREFIX=/home/user/NiMROD/.mamba \\
      NIMROD_THREADS=1 ./bin/micromamba run -n nimrod python scripts/smoke_excited.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nimrod.excited import (  # noqa: E402
    absorption_spectrum,
    colour_centre_shift,
    eom_ccsd_states,
    lowest_bright,
    occ_absorption,
    probe_tdscf_variables,
    run_tddft,
    tdscf_capability,
    tddft_ladder,
    verify_root_stability,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


# Geometries: experimental / near-experimental, angstrom.
H2CO = """
C   0.000000   0.000000  -0.529500
O   0.000000   0.000000   0.673500
H   0.000000   0.936300  -1.107700
H   0.000000  -0.936300  -1.107700
"""

H2O = """
O   0.000000   0.000000  -0.068516
H   0.000000  -0.790689   0.543701
H   0.000000   0.790689   0.543701
"""

CH3 = """
C   0.000000   0.000000   0.000000
H   1.079000   0.000000   0.000000
H  -0.539500   0.934440   0.000000
H  -0.539500  -0.934440   0.000000
"""


def benzene_geometry() -> str:
    from nimrod.geometry import named_pah

    return named_pah("benzene").to_psi4()


# ---------------------------------------------------------------------------
# 0. Which functionals can TDSCF actually run?
# ---------------------------------------------------------------------------

banner("0. TDSCF functional capability across the NiMROD ladder")
from nimrod.config import FUNCTIONAL_LADDER, REFERENCE_FUNCTIONAL  # noqa: E402

for f in FUNCTIONAL_LADDER:
    cap = tdscf_capability(f.name)
    flag = "ok " if cap.supported else "NO "
    mangle = "" if cap.driver_string_safe else f"  <- 'td-{f.name}' MANGLES TO '{cap.mangled_to}'"
    print(f"  {flag} {f.name:10s} meta={cap.is_meta!s:5s} x_alpha={cap.hf_exchange:.3f}"
          f"  {cap.reason}{mangle}")

usable = tddft_ladder()
print(f"  usable TDDFT ladder: {usable}")
check("some functionals are usable", len(usable) >= 4, str(usable))
check(
    f"project REFERENCE_FUNCTIONAL ({REFERENCE_FUNCTIONAL}) correctly reported unusable",
    not tdscf_capability(REFERENCE_FUNCTIONAL).supported,
    "it is a meta-GGA; TDSCF cannot run it",
)
check(
    "driver-string mangling of wb97x-d is detected",
    not tdscf_capability("wb97x-d").driver_string_safe,
    f"td-wb97x-d -> {tdscf_capability('wb97x-d').mangled_to}",
)


# ---------------------------------------------------------------------------
# 1. Which QCVariables does this Psi4 actually publish?
# ---------------------------------------------------------------------------

banner("1. Probe: real Psi4 QCVariable names for TDDFT roots")
probe = probe_tdscf_variables(H2CO, functional="b3lyp", basis="6-31G", n_states=2)
print(f"  psi4 {probe['psi4_version']}")
print(f"  ROOT variables on the wavefunction: {probe['n_root_variables_on_wfn']}")
print(f"  ROOT variables in the global namespace: {probe['n_root_variables_global']}")
print(f"  parseable prefixes: {probe['prefixes']}")
for prefix, roots in probe["root_variables"].items():
    for root, props in roots.items():
        print(f"    {prefix} ROOT 0 -> ROOT {root}: {props}")
print(f"  tdscf_excitations() dict keys: {probe['api_keys']}")
print(f"  tdscf_excitations() SPIN tags: {probe['api_spins']}")
check("TD-DFT prefix present", "TD-DFT" in probe["prefixes"])
check("TD-B3LYP prefix present", "TD-B3LYP" in probe["prefixes"])
check("API exposes a SPIN key", "SPIN" in probe["api_keys"])


# ---------------------------------------------------------------------------
# 2. Formaldehyde: the n->pi* dark state
# ---------------------------------------------------------------------------

banner("2. H2CO  TD-B3LYP/6-31G  (RPA, 4 singlets)")
res = run_tddft(H2CO, functional="b3lyp", basis="6-31G", n_states=4)
print(res.summary())
check("converged", res.ok, res.error or "")
s1 = res.states[0]
check("S1 in 3.5-4.3 eV (n->pi*)", 3.5 <= s1.energy_ev <= 4.3, f"{s1.energy_ev:.4f} eV")
check(
    "oscillator strength is a real float, not None",
    isinstance(s1.oscillator_strength, float),
    f"type={type(s1.oscillator_strength).__name__} value={s1.oscillator_strength!r}",
)
check("S1 is dark (symmetry-forbidden n->pi*)", s1.oscillator_strength < 1e-4,
      f"f={s1.oscillator_strength:.3e}")
bright = lowest_bright(res.states)
check("a bright state exists higher up", bright is not None,
      "" if bright is None else f"S{bright.index} {bright.energy_ev:.3f} eV f={bright.oscillator_strength:.4f}")
check("dominant orbital pair identified", s1.dominant_pair is not None,
      f"{s1.dominant_pair} w={s1.dominant_weight}")

banner("2b. H2CO  same job via harvest='driver' (psi4.energy('td-b3lyp'))")
res_drv = run_tddft(H2CO, functional="b3lyp", basis="6-31G", n_states=4, harvest="driver")
print(res_drv.summary())
check("driver route converged", res_drv.ok, res_drv.error or "")
if res_drv.ok:
    delta = abs(res_drv.states[0].energy_ev - s1.energy_ev)
    check("driver route agrees with API route to 1e-6 eV", delta < 1e-6, f"delta={delta:.2e} eV")

banner("2c. H2CO  triplets='ALSO' — spin labels read from Psi4")
res_t = run_tddft(H2CO, functional="b3lyp", basis="6-31G", n_states=4, triplets="ALSO")
print(res_t.summary())
check("converged", res_t.ok, res_t.error or "")
if res_t.ok:
    labels = [s.spin_label for s in res_t.states]
    check("both singlets and triplets present", {"singlet", "triplet"} <= set(labels), str(labels))
    check("lowest root is the triplet n->pi*", res_t.states[0].spin_label == "triplet",
          f"{res_t.states[0].energy_ev:.4f} eV")
    t1 = res_t.states[0]
    s1t = next(s for s in res_t.states if s.spin_label == "singlet")
    check("singlet-triplet gap of the n->pi* pair is 0.4-1.2 eV",
          0.4 <= s1t.energy_ev - t1.energy_ev <= 1.2,
          f"{s1t.energy_ev - t1.energy_ev:.4f} eV")


# ---------------------------------------------------------------------------
# 3. Benzene
# ---------------------------------------------------------------------------

banner("3. benzene  TD-B3LYP/6-31G  (RPA, 14 singlets)")
bz = benzene_geometry()
res_bz = run_tddft(bz, functional="b3lyp", basis="6-31G", n_states=14, label="benzene")
check("converged", res_bz.ok, res_bz.error or "")
for s in res_bz.states:
    print(f"    S{s.index:2d} {s.energy_ev:7.3f} eV  {s.energy_nm:6.1f} nm  "
          f"f={s.oscillator_strength:.3e}  {s.dominant_pair}")
if res_bz.ok:
    b1 = res_bz.states[0]
    check("lowest state in 5.0-5.5 eV", 5.0 <= b1.energy_ev <= 5.5, f"{b1.energy_ev:.4f} eV")
    check("lowest state is dark (1B2u is forbidden)", b1.oscillator_strength < 1e-3,
          f"f={b1.oscillator_strength:.3e}")
    check("second state is the dark 1B1u near 6.2 eV",
          6.0 <= res_bz.states[1].energy_ev <= 6.5,
          f"{res_bz.states[1].energy_ev:.4f} eV")
    bz_bright = lowest_bright(res_bz.states)
    check("bright E1u band found", bz_bright is not None,
          "" if bz_bright is None else
          f"S{bz_bright.index} {bz_bright.energy_ev:.3f} eV f={bz_bright.oscillator_strength:.4f}")
    if bz_bright is not None:
        check("E1u near 7.3 eV and strongly allowed (expt 6.94 eV, f~0.9)",
              6.9 <= bz_bright.energy_ev <= 7.6 and bz_bright.oscillator_strength > 0.3,
              f"f={bz_bright.oscillator_strength:.4f} at {bz_bright.energy_ev:.3f} eV")
        degenerate = [s for s in res_bz.states if abs(s.energy_ev - bz_bright.energy_ev) < 1e-4]
        check("E1u is the expected doubly degenerate pair", len(degenerate) == 2,
              f"{len(degenerate)} state(s) at that energy")
    grid, spectrum = absorption_spectrum(res_bz.states)
    peak_ev = float(grid[int(spectrum.argmax())])
    check("broadened spectrum peaks at the E1u band", abs(peak_ev - 7.27) < 0.2,
          f"peak {peak_ev:.2f} eV")

banner("3b. root-window stability: does an 8-root benzene run miss anything?")
stab = verify_root_stability(bz, n_states=8, n_reference=14, functional="b3lyp",
                             basis="6-31G", label="benzene")
print("    " + stab.summary().replace("\n", "\n    "))
check("the 8-root window is correctly diagnosed as UNSTABLE", not stab.stable,
      f"{len(stab.missing)} missed state(s)")
check("the missed set is the bright E1u pair",
      len(stab.missing) == 2
      and stab.brightest_missing is not None
      and stab.brightest_missing.oscillator_strength > 0.3,
      "" if stab.brightest_missing is None else str(stab.brightest_missing))


# ---------------------------------------------------------------------------
# 4. Water EOM-CCSD
# ---------------------------------------------------------------------------

banner("4. H2O  EOM-CCSD/6-31G  (3 roots)")
try:
    eom = eom_ccsd_states(H2O, basis="6-31G", n_states=3)
    for s in eom:
        print(f"    {s}")
    check("3 roots returned", len(eom) == 3, f"got {len(eom)}")
    check("lowest EOM-CCSD root in 7.0-8.5 eV", 7.0 <= eom[0].energy_ev <= 8.5,
          f"{eom[0].energy_ev:.4f} eV")
    check("oscillator strengths honestly reported as unavailable",
          eom[0].oscillator_strength is None and not eom[0].oscillator_strength_available)
except Exception as exc:  # noqa: BLE001
    check("EOM-CCSD ran", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 5. Open-shell doublet: the colour-centre path
# ---------------------------------------------------------------------------

banner("5. CH3 radical (doublet)  UKS TD-B3LYP/6-31G  — colour-centre path")
occ = occ_absorption(CH3, functional="b3lyp", basis="6-31G", multiplicity=2, n_states=4)
for s in occ.states:
    print(f"    {s}")
print(f"    <S^2> = {occ.s_squared!r}  ideal = {occ.s_squared_ideal!r}  "
      f"contamination = {occ.spin_contamination!r}  reliable = {occ.reliable}")
for w in occ.warnings:
    print(f"    ! {w}")
check("UKS TDDFT with a hybrid functional ran", bool(occ.states), occ.error or "")
check("spin contamination measured", occ.spin_contamination is not None,
      f"{occ.spin_contamination!r}")
if occ.spin_contamination is not None:
    check("CH3 UB3LYP is nearly spin-pure", abs(occ.spin_contamination) < 0.02,
          f"{occ.spin_contamination:+.5f}")
check("roots labelled 'doublet', not 'singlet'",
      all(s.spin_label == "doublet" for s in occ.states),
      str([s.spin_label for s in occ.states]))
check("a transition was selected", occ.lowest_bright is not None,
      "" if occ.lowest_bright is None else str(occ.lowest_bright))

banner("5b. triplets='ALSO' on an unrestricted reference must be refused")
bad = run_tddft(CH3, functional="b3lyp", basis="6-31G", multiplicity=2,
                n_states=2, triplets="ALSO")
print(f"    ok={bad.ok}  error={bad.error}")
check("rejected before any SCF ran", not bad.ok and "unrestricted" in (bad.error or ""))

banner("5c. a meta-GGA must be refused with a clear reason, not a KeyError")
bad_meta = run_tddft(H2CO, functional="tpssh", basis="6-31G", n_states=2)
print(f"    ok={bad_meta.ok}  error={bad_meta.error}")
check("meta-GGA refused with an explanation", not bad_meta.ok and "meta" in (bad_meta.error or ""))


# ---------------------------------------------------------------------------
# 6. The optical readout
# ---------------------------------------------------------------------------

banner("6. colour_centre_shift: benzene -> naphthalene (extended conjugation)")
# Stand-in for pristine-pyrene vs defect-pyrene, using only geometries the
# verified PAH builder produces.  Same physics as the real readout: enlarging
# the pi system lowers the lowest bright transition, so red_shift_ev must be
# positive.  Naphthalene's bright 1B2u band is at 5.89 eV / 210 nm
# experimentally, benzene's 1E1u at 6.94 eV / 179 nm.
from nimrod.geometry import named_pah  # noqa: E402

res_naph = run_tddft(named_pah("naphthalene").to_psi4(), functional="b3lyp",
                     basis="6-31G", n_states=6, label="naphthalene")
check("naphthalene converged", res_naph.ok, res_naph.error or "")
for s in res_naph.states:
    print(f"    naph S{s.index} {s.energy_ev:7.3f} eV  {s.energy_nm:6.1f} nm  "
          f"f={s.oscillator_strength:.3e}  {s.dominant_pair}")

shift = colour_centre_shift(res_bz.states, res_naph.states)
for k, v in shift.items():
    print(f"    {k:22s} {v}")
check("shift computed", shift.get("error") is None, str(shift.get("error")))
if shift.get("error") is None:
    check("red_shift_ev and red_shift_nm are floats",
          isinstance(shift["red_shift_ev"], float) and isinstance(shift["red_shift_nm"], float))
    check("sign convention: positive red_shift_ev <=> positive red_shift_nm",
          (shift["red_shift_ev"] > 0) == (shift["red_shift_nm"] > 0),
          f"{shift['red_shift_ev']:+.4f} eV / {shift['red_shift_nm']:+.2f} nm")
    check("naphthalene's lowest bright band is red-shifted from benzene's",
          shift["direction"] == "red" and 1.0 <= shift["red_shift_ev"] <= 4.5,
          f"{shift['red_shift_ev']:+.3f} eV / {shift['red_shift_nm']:+.1f} nm")
    # Main-band comparison: benzene 1E1u -> naphthalene 1Bb.  Experimentally
    # 6.94 -> 5.62 eV, a 1.3 eV red shift.
    check("main absorption band red-shifts by ~1.0-1.6 eV (expt 1.32 eV)",
          1.0 <= shift["strongest_red_shift_ev"] <= 1.6,
          f"{shift['pristine_strongest_ev']:.3f} -> {shift['defect_strongest_ev']:.3f} eV "
          f"= {shift['strongest_red_shift_ev']:+.3f} eV")

banner("6b. colour_centre_shift with a dark-only 'defect' set (CH3)")
dark_case = colour_centre_shift(res_bz.states, occ.states)
print(f"    {dark_case.get('error')}")
check("dark defect set reported, not crashed", dark_case.get("error") is not None)

banner("6c. colour_centre_shift when a band list is empty")
empty = colour_centre_shift([], res_naph.states)
print(f"    {empty.get('error')}")
check("missing band reported, not crashed", empty.get("error") is not None)


# ---------------------------------------------------------------------------

banner("SUMMARY")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
