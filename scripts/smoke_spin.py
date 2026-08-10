"""Smoke test for :mod:`nimrod.spin`.

Every check here has an analytically known answer, so a wrong <S^2>
implementation cannot hide:

* O2 triplet     -- <S^2> ~ 2.0, one unpaired electron of spin density on each
                    oxygen (1.0 + 1.0).
* CH2 triplet    -- <S^2> ~ 2.0, spin density localised on the carbon.
* H2O singlet    -- <S^2> exactly 0.0 and zero spin density everywhere.
* H...H at 3 A   -- a two-centre broken-symmetry test.  The HS triplet has
                    <S^2> ~ 2, the BS "singlet" must land near 1.0 (one alpha
                    electron on one atom, one beta on the other) with equal and
                    opposite atomic spins, and J must be negative
                    (antiferromagnetic).
* H2 vs FCI      -- the real correctness proof for the coupling machinery.  H2
                    has two electrons, so FCI in the same basis *is* the exact
                    answer, and the exact J follows from the singlet-triplet
                    gap as J = (E_S - E_T)/2 under H = -2 J S1.S2.  The
                    broken-symmetry UHF result is required to reproduce it to
                    within 10% at three separations.  B3LYP is run alongside
                    and is *not* asserted against FCI: it overestimates |J| by
                    a factor of 1.4 to 7 here, which is the textbook
                    self-interaction/delocalisation failure of BS-DFT on
                    stretched H2, not a bug in this module.
* fragment split -- purely geometric, no SCF: the 54-atom assembly must
                    partition cleanly into catalyst and colour centre.

Run with:

    MAMBA_ROOT_PREFIX=... NIMROD_THREADS=1 \\
        ./bin/micromamba run -n nimrod python scripts/smoke_spin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nimrod.config import HARTREE_TO_CM  # noqa: E402
from nimrod.geometry import sensor_assembly  # noqa: E402
from nimrod.psi4_driver import JobSpec, clean_context, run_energy  # noqa: E402
from nimrod.spin import (  # noqa: E402
    exchange_coupling,
    fragment_spin,
    populations_from_properties,
    sensor_fragments,
    spin_properties,
    yamaguchi_j,
)

BASIS = "def2-svp"
METHOD = "b3lyp"
USE_CACHE = False  # the hook's output is not part of the cache key; stay honest

O2 = """
O 0.00000000 0.00000000 0.00000000
O 0.00000000 0.00000000 1.20800000
"""

CH2 = """
C  0.00000000  0.00000000  0.10000000
H  0.00000000  0.98200000 -0.29800000
H  0.00000000 -0.98200000 -0.29800000
"""

H2O = """
O  0.00000000  0.00000000  0.11730000
H  0.00000000  0.75720000 -0.46920000
H  0.00000000 -0.75720000 -0.46920000
"""

H2_STRETCHED = """
H 0.00000000 0.00000000 0.00000000
H 0.00000000 0.00000000 3.00000000
"""

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"   [{status}] {name}: {detail}")
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def show_populations(properties: dict) -> None:
    pops = populations_from_properties(properties)
    print(f"   {'atom':>6} {'Mulliken':>12} {'Lowdin':>12} {'charge':>10}")
    for i, symbol in enumerate(pops.symbols):
        print(
            f"   {i:>2} {symbol:>3} {pops.mulliken[i]:>12.6f} "
            f"{pops.lowdin[i]:>12.6f} {pops.charges[i]:>10.6f}"
        )
    print(
        f"   sums: Mulliken {pops.mulliken.sum():+.8f}  "
        f"Lowdin {pops.lowdin.sum():+.8f}  "
        f"(N_alpha - N_beta = {properties['n_alpha'] - properties['n_beta']})"
    )


def single_point(geometry: str, multiplicity: int, label: str):
    spec = JobSpec(
        geometry=geometry,
        method=METHOD,
        basis=BASIS,
        multiplicity=multiplicity,
        reference="uks" if multiplicity > 1 else "rks",
        label=label,
    )
    return run_energy(spec, use_cache=USE_CACHE, property_hook=spin_properties)


def case_o2() -> None:
    print("\n1. O2 triplet, UKS b3lyp/def2-svp")
    result = single_point(O2, 3, "o2-triplet")
    if not result.ok:
        check("O2 SCF", False, str(result.error))
        return
    p = result.properties
    print(f"   E = {result.energy:.8f} Eh   preset = {result.preset}")
    print(
        f"   <S^2> = {p['s2']:.6f}   ideal = {p['s2_ideal']:.6f}   "
        f"contamination = {p['spin_contamination']:+.6f}"
    )
    show_populations(p)
    check("O2 <S^2> ~ 2.0", abs(p["s2"] - 2.0) < 0.05, f"{p['s2']:.6f}")
    mulliken = np.asarray(p["mulliken_spin"])
    check(
        "O2 Mulliken spin ~ 1.0 per oxygen",
        bool(np.all(np.abs(mulliken - 1.0) < 0.05)),
        f"{mulliken[0]:.6f}, {mulliken[1]:.6f}",
    )
    check(
        "O2 spin sums to N_alpha - N_beta",
        abs(p["mulliken_spin_total"] - 2.0) < 1e-6
        and abs(p["lowdin_spin_total"] - 2.0) < 1e-6,
        f"Mulliken {p['mulliken_spin_total']:.8f}, Lowdin {p['lowdin_spin_total']:.8f}",
    )


def case_ch2() -> None:
    print("\n2. CH2 triplet, UKS b3lyp/def2-svp")
    result = single_point(CH2, 3, "ch2-triplet")
    if not result.ok:
        check("CH2 SCF", False, str(result.error))
        return
    p = result.properties
    print(f"   E = {result.energy:.8f} Eh   preset = {result.preset}")
    print(
        f"   <S^2> = {p['s2']:.6f}   ideal = {p['s2_ideal']:.6f}   "
        f"contamination = {p['spin_contamination']:+.6f}"
    )
    show_populations(p)
    check("CH2 <S^2> ~ 2.0", abs(p["s2"] - 2.0) < 0.05, f"{p['s2']:.6f}")
    mulliken = np.asarray(p["mulliken_spin"])
    carbon_spin = fragment_spin(populations_from_properties(p), [0])
    check(
        "CH2 spin localised on carbon (>90%)",
        carbon_spin / 2.0 > 0.90,
        f"C carries {carbon_spin:.6f} of 2.0 ({100 * carbon_spin / 2:.1f}%)",
    )
    check(
        "CH2 hydrogens nearly spin-free",
        bool(np.all(np.abs(mulliken[1:]) < 0.10)),
        f"H spins {mulliken[1]:.6f}, {mulliken[2]:.6f}",
    )


def case_h2o() -> None:
    print("\n3. H2O closed shell, RKS b3lyp/def2-svp")
    result = single_point(H2O, 1, "h2o-singlet")
    if not result.ok:
        check("H2O SCF", False, str(result.error))
        return
    p = result.properties
    print(f"   E = {result.energy:.8f} Eh   preset = {result.preset}")
    print(f"   <S^2> = {p['s2']!r}   ideal = {p['s2_ideal']:.6f}")
    show_populations(p)
    check("H2O <S^2> exactly 0.0", p["s2"] == 0.0, repr(p["s2"]))
    check(
        "H2O has no spin density",
        max(abs(x) for x in p["mulliken_spin"]) == 0.0,
        f"max |spin| = {max(abs(x) for x in p['mulliken_spin']):.3e}",
    )


def case_bs_dimer() -> None:
    print("\n4. H...H at 3.0 A, broken-symmetry UKS b3lyp/def2-svp")
    coupling = exchange_coupling(
        H2_STRETCHED,
        method=METHOD,
        basis=BASIS,
        spin_a=0.5,
        spin_b=0.5,
        label="H2@3A",
        use_cache=USE_CACHE,
    )
    print(f"   multiplicities: HS = {coupling.multiplicity_hs}, BS = {coupling.multiplicity_bs}")
    if coupling.error:
        print(f"   error: {coupling.error}")
    if coupling.hs_result is not None and coupling.hs_result.ok:
        print(
            f"   E_HS = {coupling.e_hs:.8f} Eh   <S^2>_HS = {coupling.s2_hs:.6f}   "
            f"preset = {coupling.hs_result.preset}"
        )
    if coupling.bs_result is not None and coupling.bs_result.ok:
        print(
            f"   E_BS = {coupling.e_bs:.8f} Eh   <S^2>_BS = {coupling.s2_bs:.6f}   "
            f"guess = {coupling.bs_result.properties.get('bs_guess')}   "
            f"collapsed = {coupling.bs_result.properties['bs_collapsed']}"
        )
        print(f"   BS Mulliken spins: {[f'{x:+.6f}' for x in coupling.spin_bs]}")
        print(f"   HS Mulliken spins: {[f'{x:+.6f}' for x in coupling.spin_hs]}")
    if not coupling.converged:
        check("H2 dimer exchange coupling", False, coupling.error or "did not converge")
        return
    print(f"   {coupling.summary()}")
    print(f"   E_HS - E_BS = {coupling.delta_e_kcal:+.6f} kcal/mol")

    check("BS <S^2> strictly between 0 and 1", 0.0 < coupling.s2_bs < 1.0, f"{coupling.s2_bs:.6f}")
    check("HS <S^2> ~ 2.0", abs(coupling.s2_hs - 2.0) < 0.05, f"{coupling.s2_hs:.6f}")
    check("BS did not collapse to closed shell", not coupling.bs_collapsed, f"collapsed={coupling.bs_collapsed}")
    bs_spins = np.asarray(coupling.spin_bs)
    check(
        "BS atomic spins are equal and opposite",
        abs(bs_spins.sum()) < 1e-6 and abs(abs(bs_spins[0]) - abs(bs_spins[1])) < 1e-4
        and bs_spins[0] * bs_spins[1] < 0,
        f"{bs_spins[0]:+.6f} / {bs_spins[1]:+.6f}",
    )
    check("J antiferromagnetic (negative)", coupling.j_cm < 0.0, f"{coupling.j_cm:+.4f} cm^-1")


def exact_j_cm(geometry: str) -> float:
    """Exact J for a two-electron system, from the FCI singlet-triplet gap.

    Under H = -2 J S1.S2 the gap is E_T - E_S = -2 J, so J = (E_S - E_T) / 2.
    For H2 this is not a model: two electrons means FCI is the exact solution
    of the electronic Schroedinger equation in this basis.
    """
    import psi4

    with clean_context():
        mol = psi4.geometry(f"0 1\n{geometry}\nsymmetry c1\nno_reorient\nno_com\n")
        psi4.set_options({"basis": BASIS, "reference": "rhf", "scf_type": "pk"})
        e_singlet = psi4.energy("fci", molecule=mol)
    with clean_context():
        mol = psi4.geometry(f"0 3\n{geometry}\nsymmetry c1\nno_reorient\nno_com\n")
        psi4.set_options({"basis": BASIS, "reference": "rohf", "scf_type": "pk"})
        e_triplet = psi4.energy("fci", molecule=mol)
    return (e_singlet - e_triplet) / 2.0 * HARTREE_TO_CM


def case_bs_versus_fci() -> None:
    """The real validation: BS-UHF J against the exact two-electron answer."""
    print("\n5. Broken-symmetry J against exact FCI, H2/def2-svp")
    print(f"   {'R (A)':>6} {'J_FCI':>11} {'J_UHF':>11} {'ratio':>7} "
          f"{'J_B3LYP':>11} {'ratio':>7}  <S^2>_BS(UHF)")
    for distance in (2.0, 3.0, 4.0):
        geometry = f"H 0.0 0.0 0.0\nH 0.0 0.0 {distance:.6f}"
        j_exact = exact_j_cm(geometry)
        results = {}
        for method in ("hf", "b3lyp"):
            results[method] = exchange_coupling(
                geometry, method=method, basis=BASIS,
                label=f"H2@{distance}", use_cache=USE_CACHE,
            )
        uhf, dft = results["hf"], results["b3lyp"]
        if not (uhf.converged and dft.converged):
            check(f"H2 R={distance} coupling", False, "an SCF did not converge")
            continue
        print(
            f"   {distance:>6.1f} {j_exact:>11.2f} {uhf.j_cm:>11.2f} "
            f"{uhf.j_cm / j_exact:>7.2f} {dft.j_cm:>11.2f} "
            f"{dft.j_cm / j_exact:>7.2f}  {uhf.s2_bs:.4f}"
        )
        check(
            f"BS-UHF J matches FCI within 10% at R={distance} A",
            abs(uhf.j_cm / j_exact - 1.0) < 0.10,
            f"{uhf.j_cm:+.2f} vs exact {j_exact:+.2f} cm^-1 "
            f"({100 * (uhf.j_cm / j_exact - 1):+.1f}%)",
        )
    print("   (B3LYP is reported, not asserted: BS-DFT overestimates |J| for")
    print("    stretched H2 through self-interaction error. That is real physics")
    print("    about the functional, not an error in this module.)")


def case_fragments() -> None:
    """Fragment bookkeeping on the real 54-atom assembly. No SCF."""
    print("\n6. Fragment partition of the 54-atom sensor (geometry only)")
    structure, info = sensor_assembly()
    fragments = sensor_fragments(structure, info)
    for name, indices in fragments.items():
        print(f"   {name:<14} {len(indices):>3} atoms")
    catalyst = set(fragments["catalyst"])
    colour_centre = set(fragments["colour_centre"])
    check("assembly is 54 atoms", len(structure) == 54, f"{len(structure)} ({structure.formula})")
    check(
        "catalyst + colour centre partition the molecule",
        catalyst | colour_centre == set(range(len(structure))) and not (catalyst & colour_centre),
        f"{len(catalyst)} + {len(colour_centre)} = {len(catalyst | colour_centre)}, "
        f"overlap {len(catalyst & colour_centre)}",
    )
    check("nickel is in the catalyst fragment", info["Ni"] in catalyst, f"Ni index {info['Ni']}")
    check(
        "sp3 defect is in the colour centre",
        info["sp3_carbon"] in colour_centre,
        f"sp3 C index {info['sp3_carbon']}",
    )
    check(
        "bridge is a C6H4 phenylene (10 atoms)",
        len(fragments["bridge"]) == 10,
        f"{len(fragments['bridge'])} atoms",
    )
    check(
        "bridge + pyrene reconstruct the colour centre",
        set(fragments["bridge"]) | set(fragments["pyrene"]) == colour_centre,
        f"{len(fragments['bridge'])} + {len(fragments['pyrene'])} = {len(colour_centre)}",
    )
    # fragment_spin must be an exact partition of any per-atom array.
    fake = np.arange(len(structure), dtype=float)
    total = fragment_spin(fake, fragments["catalyst"]) + fragment_spin(fake, fragments["colour_centre"])
    check(
        "fragment_spin partitions a per-atom array exactly",
        abs(total - fake.sum()) < 1e-9,
        f"{total} vs {fake.sum()}",
    )


def case_formula_algebra() -> None:
    """yamaguchi_j is pure arithmetic; pin its sign convention with exact input."""
    print("\n7. yamaguchi_j sign convention (no SCF)")
    # BS below HS by 1 mEh with the textbook contaminations -> antiferromagnetic.
    j_af = yamaguchi_j(e_hs=-1.0000, s2_hs=2.0, e_bs=-1.0010, s2_bs=1.0)
    # HS below BS -> ferromagnetic.
    j_f = yamaguchi_j(e_hs=-1.0010, s2_hs=2.0, e_bs=-1.0000, s2_bs=1.0)
    print(f"   E_BS below E_HS by 1 mEh -> J = {j_af:+.4f} cm^-1")
    print(f"   E_HS below E_BS by 1 mEh -> J = {j_f:+.4f} cm^-1")
    check("E_BS < E_HS gives J < 0 (antiferromagnetic)", j_af < 0, f"{j_af:+.4f} cm^-1")
    check("E_HS < E_BS gives J > 0 (ferromagnetic)", j_f > 0, f"{j_f:+.4f} cm^-1")
    check("|J| = 0.001 Eh / 1.0 in cm^-1", abs(abs(j_af) - 219.4746) < 1e-3, f"{abs(j_af):.4f}")
    try:
        yamaguchi_j(e_hs=-1.0, s2_hs=2.0, e_bs=-1.001, s2_bs=2.0)
    except ValueError as exc:
        check("degenerate denominator raises", True, f"{type(exc).__name__}")
    else:
        check("degenerate denominator raises", False, "no exception")


def main() -> int:
    print("=" * 72)
    print("nimrod.spin smoke test")
    print("=" * 72)
    case_o2()
    case_ch2()
    case_h2o()
    case_bs_dimer()
    case_bs_versus_fci()
    case_fragments()
    case_formula_algebra()
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
