"""Smoke test for :mod:`nimrod.casscf`.

Everything checked here has a known answer, so a broken active space or a
silently-failing detci run cannot pass.

* ``cas_space``      -- pure arithmetic and validation; no Psi4 involved.
* PSIO scratch       -- the bug that motivated this module.  ``PSI_SCRATCH``
                        being unset made detci abort the whole process with
                        ``PSIO_ERROR: 18 (Incorrect block end address)``.  We
                        assert that Psi4's IO manager is pointing at the
                        project scratch directory and that a CASSCF actually
                        completes.
* N2 CASSCF(6,6)     -- the textbook active space (three bonding and three
                        antibonding orbitals of the triple bond) at
                        equilibrium.  Must converge without drama and must be
                        strongly single-reference: N2 at r_e is the standard
                        example of a molecule that is *not* multireference.
* O2 CASSCF(8,6)     -- the physics test.  O2 has two electrons in a degenerate
                        pi* pair, so its ground state is the X 3Sg- triplet and
                        the lowest singlet a 1Dg lies 0.98 eV above it
                        experimentally.  The triplet MUST come out lower.  A
                        small-basis CASSCF has no dynamic correlation and is
                        expected to land somewhere in 0.7-1.6 eV.
* Diagnostics        -- H2O CISD (single-reference, C0^2 should be ~0.96) and
                        stretched H2 CASSCF(2,2) (a textbook diradical, C0^2
                        should collapse towards 0.5 and the excess unpaired
                        count towards 1).  These two bracket the verdict, so a
                        diagnostic that always says "fine" fails here.

Run with:

    cd /home/user/NiMROD && MAMBA_ROOT_PREFIX=/home/user/NiMROD/.mamba \\
        NIMROD_THREADS=1 ./bin/micromamba run -n nimrod \\
        python scripts/smoke_casscf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nimrod.casscf import (  # noqa: E402
    cas_space,
    casscf_energy,
    casscf_spin_gap,
    count_electrons,
    multireference_diagnostic,
    parse_leading_determinants,
)
from nimrod.config import DEFAULT_COMPUTE, HARTREE_TO_EV  # noqa: E402
from nimrod.psi4_driver import ensure_initialised  # noqa: E402

USE_CACHE = False  # a smoke test that reads its own cache proves nothing

# Experimental / reference equilibrium geometries (angstrom).
N2 = "N 0.00000000 0.00000000 0.00000000\nN 0.00000000 0.00000000 1.09768000\n"
O2 = "O 0.00000000 0.00000000 0.00000000\nO 0.00000000 0.00000000 1.20752000\n"
H2O = (
    "O 0.00000000  0.00000000  0.11779000\n"
    "H 0.00000000  0.75545000 -0.47116000\n"
    "H 0.00000000 -0.75545000 -0.47116000\n"
)
H2_STRETCHED = "H 0.00000000 0.00000000 0.00000000\nH 0.00000000 0.00000000 2.50000000\n"

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        _failures.append(message)


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
rule("1. cas_space arithmetic and validation (no Psi4)")

space = cas_space(14, 6, 6)
print(f"  N2 CAS(6,6):  {space}")
check(space == {"RESTRICTED_DOCC": [4], "ACTIVE": [6]}, "N2 CAS(6,6) -> docc 4, active 6")

space = cas_space(16, 8, 6)
print(f"  O2 CAS(8,6):  {space}")
check(space == {"RESTRICTED_DOCC": [4], "ACTIVE": [6]}, "O2 CAS(8,6) -> docc 4, active 6")

space = cas_space(2, 2, 2)
check(space == {"RESTRICTED_DOCC": [0], "ACTIVE": [2]}, "H2 CAS(2,2) -> docc 0, active 2")

for args, needle in (
    ((14, 5, 6), "parity"),
    ((14, 16, 6), "only has"),
    ((14, 6, 2), "overfull"),
    ((14, 6, 0), "at least 1"),
    ((-2, 2, 2), "non-negative"),
):
    try:
        cas_space(*args)
    except (ValueError, TypeError) as exc:
        check(needle in str(exc), f"cas_space{args} raises with {needle!r}: {str(exc)[:60]}...")
    else:
        check(False, f"cas_space{args} should have raised")

check(count_electrons(N2) == 14, f"count_electrons(N2) == 14 (got {count_electrons(N2)})")
check(count_electrons(O2) == 16, f"count_electrons(O2) == 16 (got {count_electrons(O2)})")
check(count_electrons(H2O, 1) == 9, f"count_electrons(H2O, +1) == 9 (got {count_electrons(H2O, 1)})")


# --------------------------------------------------------------------------
rule("2. PSI_SCRATCH is pinned (the PSIO_ERROR: 18 bug)")

import os  # noqa: E402

ensure_initialised()
import psi4  # noqa: E402

io_path = psi4.core.IOManager.shared_object().get_default_path()
print(f"  PSI_SCRATCH env      : {os.environ.get('PSI_SCRATCH')}")
print(f"  psi4 IOManager path  : {io_path}")
print(f"  config scratch dir   : {DEFAULT_COMPUTE.scratch}")
check(os.environ.get("PSI_SCRATCH") == str(DEFAULT_COMPUTE.scratch), "PSI_SCRATCH env var is pinned")
check(
    Path(io_path).resolve() == DEFAULT_COMPUTE.scratch.resolve(),
    "Psi4 IOManager default path is the project scratch directory",
)


# --------------------------------------------------------------------------
rule("3. N2 CASSCF(6,6)/6-31G at equilibrium")

n2 = casscf_energy(N2, 0, 1, 6, 6, "6-31G", label="N2-cas66", use_cache=USE_CACHE)
print(f"  converged : {n2.converged}   preset: {n2.preset}   {n2.wall_seconds:.1f} s")
print(f"  E(CASSCF) : {n2.energy!r}")
print(f"  E(RHF)    : {n2.properties.get('reference_energy')!r}")
print(f"  NOON      : {np.round(n2.properties.get('natural_occupations', []), 5)}")
print(f"  C0        : {n2.properties.get('largest_ci_coefficient')!r}")
print(f"  C0^2      : {n2.properties.get('reference_weight')!r}")

check(n2.ok, "N2 CASSCF converged")
if n2.ok:
    check(-109.2 < n2.energy < -108.9, f"N2 CASSCF energy is physical ({n2.energy:.6f} Eh)")
    occ = n2.properties.get("natural_occupations", [])
    check(len(occ) == 6, f"active-space NOONs has 6 entries (got {len(occ)})")
    check(
        abs(sum(occ) - 6.0) < 1e-6,
        f"active NOONs sum to the 6 active electrons (got {sum(occ):.8f})",
    )
    check(
        n2.properties.get("reference_weight", 0.0) > 0.90,
        f"N2 at r_e is single-reference, C0^2 = {n2.properties.get('reference_weight')}",
    )


# --------------------------------------------------------------------------
rule("4. O2 CASSCF(8,6)/6-31G triplet vs singlet")

gap = casscf_spin_gap(
    O2,
    low_multiplicity=1,
    high_multiplicity=3,
    n_active_electrons=8,
    n_active_orbitals=6,
    basis="6-31G",
    label="O2-cas86",
    use_cache=USE_CACHE,
)
print(f"  singlet (m=1) : {gap.low_energy!r}  preset {gap.low.preset}")
print(f"  triplet (m=3) : {gap.high_energy!r}  preset {gap.high.preset}")
print(f"  {gap.summary()}")
if gap.ok:
    print(f"  singlet NOON  : {np.round(gap.low.properties.get('natural_occupations', []), 4)}")
    print(f"  triplet NOON  : {np.round(gap.high.properties.get('natural_occupations', []), 4)}")

check(gap.ok, "both O2 CASSCF states converged")
if gap.ok:
    # gap = E(high) - E(low) = E(triplet) - E(singlet); triplet ground state
    # means that is NEGATIVE.
    triplet_below_singlet = gap.gap_hartree < 0.0
    check(triplet_below_singlet, f"triplet is the ground state (gap {gap.gap_ev:+.4f} eV)")
    check(gap.ground_multiplicity == 3, f"ground_multiplicity == 3 (got {gap.ground_multiplicity})")
    excitation_ev = -gap.gap_ev  # X 3Sg- -> a 1Dg
    print(f"\n  X 3Sg- -> a 1Dg excitation = {excitation_ev:.4f} eV "
          f"({-gap.gap_kcal:.2f} kcal/mol);  experiment 0.98 eV")
    check(
        0.7 <= excitation_ev <= 1.6,
        f"singlet-triplet excitation in the expected 0.7-1.6 eV window "
        f"(got {excitation_ev:.4f} eV)",
    )


# --------------------------------------------------------------------------
rule("5. Multireference diagnostic -- H2O CISD (should be single-reference)")

water = multireference_diagnostic(
    H2O,
    charge=0,
    multiplicity=1,
    method="cisd",
    basis="6-31G",
    label="H2O-cisd",
    use_cache=USE_CACHE,
)
print(f"  {water.summary()}")
print(f"  sources   : {water.sources}")
print(f"  notes     : {water.notes}")
for det in water.leading_determinants[:3]:
    print(f"    det {det[0]:>2d}  C = {det[1]:+.6f}  {det[4]}")

check(water.converged, "H2O CISD converged")
if water.converged:
    check(
        water.reference_weight is not None and water.reference_weight > 0.93,
        f"H2O CISD reference weight is high (C0^2 = {water.reference_weight})",
    )
    check(
        water.occupation_space == "full-mo",
        f"CISD occupations span the full MO space (got {water.occupation_space!r})",
    )
    check(water.single_reference_ok is True, "verdict: single-reference DFT defensible")


# --------------------------------------------------------------------------
rule("6. Multireference diagnostic -- H2 at 2.5 A, CASSCF(2,2) (a diradical)")

stretched = multireference_diagnostic(
    H2_STRETCHED,
    charge=0,
    multiplicity=1,
    method="casscf",
    n_active_electrons=2,
    n_active_orbitals=2,
    basis="6-31G",
    label="H2-2.5A-cas22",
    use_cache=USE_CACHE,
)
print(f"  {stretched.summary()}")
print(f"  NOON      : {np.round(stretched.natural_occupations, 5)}")
for det in stretched.leading_determinants[:3]:
    print(f"    det {det[0]:>2d}  C = {det[1]:+.6f}  {det[4]}")

check(stretched.converged, "stretched-H2 CASSCF converged")
if stretched.converged:
    check(
        stretched.reference_weight is not None and stretched.reference_weight < 0.75,
        f"reference weight has collapsed (C0^2 = {stretched.reference_weight})",
    )
    check(
        stretched.excess_unpaired is not None and stretched.excess_unpaired > 0.8,
        f"about one effectively unpaired electron pair "
        f"(excess N_U = {stretched.excess_unpaired})",
    )
    check(
        stretched.single_reference_ok is False,
        "verdict: MULTIREFERENCE -- exactly the case DFT must not be trusted on",
    )


# --------------------------------------------------------------------------
rule("7. Determinant-table parser (unit test, no Psi4)")

SAMPLE = """
   The 3 most important determinants:

    *   1   -0.963838  (    0,    0)  5AX 6AX 7AX
    *   2    0.075576  (    5,    5)  5AX 7AX 9AX
    *   3   -0.055257  (    1,    4)  5AX 6AX 8AA 9AB


Properties will be evaluated
"""
dets = parse_leading_determinants(SAMPLE)
check(len(dets) == 3, f"parsed 3 determinants (got {len(dets)})")
if dets:
    check(abs(dets[0].coefficient + 0.963838) < 1e-9, "leading coefficient parsed")
    check(abs(dets[0].weight - 0.963838**2) < 1e-9, "weight is C^2")
    check(dets[0].occupation == "5AX 6AX 7AX", f"occupation string: {dets[0].occupation!r}")
check(parse_leading_determinants("no table here") == [], "missing table -> empty list")


# --------------------------------------------------------------------------
rule("RESULT")
if _failures:
    print(f"{len(_failures)} check(s) FAILED:")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
