#!/usr/bin/env python
"""The discriminating experiment, on CPU: does J follow the sign of rho?

The claim under test is this project's own current account of the nitronyl
nitroxide node -- that kinetic exchange is switched off by the symmetry node at
the tether carbon, leaving a weak polarisation-mediated coupling.  A rival
account, held earlier in this same project and withdrawn, was that the node
reverses the SIGN of J because the local spin density there is negative.

A rigid tether rotation separates them.  Rotating the reporter about the biaryl
bond cannot change the sign of the spin density at the tether carbon: that sign
is fixed by the reporter's own electronic structure, which a rotation does not
touch.  So if J changes sign across the scan, J is not a function of sign(rho)
and the rival account is dead.  If J keeps one sign and merely changes
magnitude, the rival account survives this test.

WHAT THIS SCAN DOES NOT DO, learned by running it
=================================================

It does not isolate the torsion.  Rotating the reporter rigidly swings the
nitroxide's N-O oxygen through an arc that passes close to the nickel: Ni...O
goes 3.96 A at 0 degrees, 3.00 A at 45, back to 3.66 A at 90.  J follows that
distance and not the angle -- ln J is linear in Ni...O with R^2 = 0.998, while
corr(J, cos^2 torsion) is -0.13.  The first pass reported the resulting 6-fold
swing in J as a torsion dependence.  It was a distance dependence wearing a
torsion scan's label.

The close approach is also not free: four of the six geometries push atoms below
the intact complex's own 2.267 A contact floor, down to 1.84 A.  Those are not
conformers, they are overlapping atoms, and STERIC_FLOOR now rejects them.

The sign test survives all of this, because it never depended on the torsion
meaning anything: rho holds its sign across every geometry regardless of what
else moved, so "did J change sign" is still answerable.  What does NOT survive
is any claim about how J depends on the tether angle.  Measuring that needs the
Ni...O distance held fixed while the angle varies, which a rigid rotation of
this fragment cannot do.

Deliberately unrelaxed.  Two things force it and one justifies it:

    forced      two Colab A100s were reclaimed inside ten minutes each, and the
                geometry optimisation is the step that does not fit in that
                window

    forced      the optimisation FAILED on its own before the second VM died --
                "Nuclear gradients of DFUKS_Scanner not converged" -- so a
                relaxed geometry is not currently available at any price

    justified   the scan is rigid by construction, so every point shares one
                structure and the comparison between points is clean.  What is
                NOT available from an unrelaxed geometry is the absolute value
                of J.  Only the sign pattern across the scan is being claimed.

The SCF is given a real convergence ladder rather than one attempt, since the
failure above shows this system does not converge on the first try.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nimrod.catalysts import _non_bonded_pairs, reporter_on_salen
from nimrod.config import DATA_DIR
from nimrod.geometry import _rotate_about

XC = "HYB_GGA_XC_B3LYP"
BASIS = "def2-svp"
HARTREE_CM = 219474.6313632
TORSIONS = (0.0, 30.0, 45.0, 60.0, 75.0, 90.0)

#: Closest non-bonded contact in the intact complex.  A rigidly
#: rotated geometry that goes below it is not a conformer, it is
#: atoms overlapping, and its energy is not a coupling.
STERIC_FLOOR = 2.2


def converge(mol, dm0=None, tag=""):
    """Try progressively harder to converge, and say which attempt worked.

    The GPU run died with a non-converged gradient, so a single kernel() call
    is demonstrably not enough for this system.  Each rung is recorded, because
    "converged after a level shift and a second-order restart" is a materially
    weaker result than "converged directly" and the difference should not be
    invisible in the output.
    """
    from pyscf import dft

    attempts = []
    for label, setup in (
        ("plain", dict()),
        ("level_shift", dict(level_shift=0.2)),
        ("damped", dict(level_shift=0.5, diis_start_cycle=8)),
    ):
        mf = dft.UKS(mol, xc=XC).density_fit()
        mf.grids.level = 3
        mf.conv_tol = 1e-8
        mf.max_cycle = 400
        for key, value in setup.items():
            setattr(mf, key, value)
        energy = float(mf.kernel(dm0=dm0))
        attempts.append(label)
        if mf.converged:
            return mf, energy, label, attempts
        # Second-order fallback from wherever the first-order run reached.
        second = mf.newton()
        second.max_cycle = 200
        energy = float(second.kernel(dm0=mf.make_rdm1()))
        attempts.append(f"{label}+newton")
        if second.converged:
            return second, energy, f"{label}+newton", attempts
    raise RuntimeError(f"{tag}: no SCF converged after {attempts}")


def atom_spin(mf, mol) -> np.ndarray:
    dm = mf.make_rdm1()
    spin = (np.asarray(dm[0]) - np.asarray(dm[1])) @ mol.intor_symmetric("int1e_ovlp")
    per_atom = np.zeros(mol.natm)
    diagonal = spin.diagonal()
    for mu, (atom, *_rest) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom] += float(diagonal[mu])
    return per_atom


def flip_fragment_spin(dm, ao_index: np.ndarray):
    dm_a, dm_b = np.asarray(dm[0]).copy(), np.asarray(dm[1]).copy()
    block = np.ix_(ao_index, ao_index)
    keep = dm_a[block].copy()
    dm_a[block] = dm_b[block]
    dm_b[block] = keep
    return np.array([dm_a, dm_b])


def rotate_tether(structure, info, degrees: float):
    moved = structure.copy()
    a, b = info["reporter_carbon"], info["ipso_carbon"]
    block = list(range(info["n_host_atoms"]))
    moved.coords[block] = _rotate_about(
        moved.coords[block], structure.coords[a],
        structure.coords[b] - structure.coords[a], math.radians(degrees))
    return moved


def main() -> int:
    from pyscf import gto

    out_path = DATA_DIR / "results" / "torsion_scan_cpu.jsonl"
    handle = out_path.open("w", buffering=1)

    def emit(record):
        handle.write(json.dumps(record) + "\n")

    built, info = reporter_on_salen("nitronyl-nitroxide", site="C3",
                                    methylated=False)
    ni = info["Ni"]
    fragment = list(range(info["n_host_atoms"]))

    emit({"event": "start", "xc": XC, "basis": BASIS, "natoms": len(built),
          "formula": built.formula, "geometry": "as built, NOT relaxed",
          "torsions": list(TORSIONS),
          "tether_carbon": info["reporter_carbon"],
          "ni_tether_carbon": built.distance(ni, info["reporter_carbon"])})

    print(f"Rigid tether-torsion scan, {built.formula}, {len(built)} atoms")
    print(f"  {XC}/{BASIS}, unrelaxed geometry -- sign pattern only\n")
    print(f"  {'torsion':>8s} {'J (cm-1)':>10s} {'<S2>hs':>7s} {'<S2>bs':>7s} "
          f"{'Ni(hs)':>7s} {'Ni(bs)':>7s} {'rho(teth)':>9s}  scf")

    results = []
    # Carry the converged density from one torsion to the next.  Starting each
    # point from the default guess is what let 60 degrees collapse to a
    # metal-quenched solution while its neighbours at 30 and 90 kept an S=1
    # metal: the guess, not the geometry, decided which SCF solution was found.
    # A scan is a continuation problem and should be treated as one.
    previous_quartet_dm = None
    for degrees in TORSIONS:
        record = {"event": "point", "torsion": degrees}
        started = time.time()
        try:
            structure = rotate_tether(built, info, degrees)
            text = structure.to_psi4()

            mol_q = gto.M(atom=text, basis=BASIS, charge=0, spin=3, verbose=0)
            mf_q, e_hs, how_q, tried_q = converge(
                mol_q, dm0=previous_quartet_dm, tag=f"quartet {degrees}")
            s2_hs = float(mf_q.spin_square()[0])
            pops_q = atom_spin(mf_q, mol_q)

            ao_fragment = np.array(
                [mu for mu, (a, *_r) in enumerate(mol_q.ao_labels(fmt=None))
                 if a in set(fragment)], dtype=int)
            dm_bs = flip_fragment_spin(mf_q.make_rdm1(), ao_fragment)
            mol_d = gto.M(atom=text, basis=BASIS, charge=0, spin=1, verbose=0)
            mf_bs, e_bs, how_bs, tried_bs = converge(
                mol_d, dm0=dm_bs, tag=f"broken-symmetry {degrees}")
            s2_bs = float(mf_bs.spin_square()[0])
            pops_bs = atom_spin(mf_bs, mol_d)

            record.update(
                e_hs=e_hs, e_bs=e_bs, s2_hs=s2_hs, s2_bs=s2_bs,
                scf_quartet=how_q, scf_bs=how_bs,
                hs_spin_Ni=float(pops_q[ni]), bs_spin_Ni=float(pops_bs[ni]),
                bs_spin_fragment=float(pops_bs[fragment].sum()),
                bs_spin_tether_carbon=float(pops_bs[info["reporter_carbon"]]))

            # Both determinants have to be the ones being described before the
            # difference between them is an exchange coupling.  This project
            # has now produced seven clean, converged calculations that
            # answered a different question than the one asked, and every one
            # of them was caught by a spin population rather than by an energy.
            # So J is gated on the populations, not merely annotated with them.
            hs_magnetic = abs(record["hs_spin_Ni"]) > 0.5
            bs_magnetic = abs(record["bs_spin_Ni"]) > 0.5
            antiparallel = record["bs_spin_Ni"] * record["bs_spin_fragment"] < 0
            record.update(hs_metal_magnetic=bool(hs_magnetic),
                          bs_metal_magnetic=bool(bs_magnetic),
                          fragments_antiparallel=bool(antiparallel))

            # A rigid rotation about the tether does NOT hold the rest of the
            # molecule still.  It swings the nitroxide's N-O oxygen through an
            # arc that passes close to the nickel: Ni...O runs 3.96 A at 0
            # degrees down to 3.00 A at 45.  The first version of this scan
            # reported the resulting 6-fold change in J as a torsion
            # dependence.  It is not one -- ln J is linear in Ni...O with
            # R^2 = 0.998, while its correlation with cos^2(torsion) is -0.13.
            # Worse, the close approach is bought by pushing atoms through each
            # other: four of six points sat below the intact complex's own
            # 2.267 A contact floor.  Both quantities are therefore recorded,
            # and clashing geometries are rejected rather than plotted.
            pairs = _non_bonded_pairs(structure)
            closest = float(np.linalg.norm(
                structure.coords[pairs[:, 0]] - structure.coords[pairs[:, 1]],
                axis=1).min())
            oxygens = [i for i in fragment if structure.symbols[i] == "O"]
            record["closest_contact"] = closest
            record["ni_nitroxide_oxygen"] = min(
                structure.distance(ni, o) for o in oxygens)

            problems = []
            if closest < STERIC_FLOOR:
                problems.append(
                    f"closest non-bonded contact {closest:.3f} A is below the "
                    f"intact complex's own {STERIC_FLOOR:.3f} A floor, so this "
                    f"geometry is sterically impossible")
            if not hs_magnetic:
                problems.append(
                    f"quartet metal carries only {record['hs_spin_Ni']:.2f} spin, "
                    f"so its three unpaired electrons are not on the metal")
            if not bs_magnetic:
                problems.append(
                    f"broken-symmetry metal carries {record['bs_spin_Ni']:.2f} "
                    f"spin, so it is a closed-shell-metal doublet, not a "
                    f"broken-symmetry state")
            if not antiparallel:
                problems.append("metal and reporter spins are not antiparallel")

            denominator = s2_hs - s2_bs
            if problems:
                # Recorded but NOT exposed as J: differencing these two gives a
                # spin-state gap of thousands of wavenumbers that looks like an
                # enormous exchange coupling.
                record["J_invalid"] = "; ".join(problems)
                record["J_cm_uninterpretable"] = (
                    (e_bs - e_hs) / denominator * HARTREE_CM
                    if abs(denominator) > 1e-4 else None)
            elif abs(denominator) > 1e-4:
                record["J_cm"] = (e_bs - e_hs) / denominator * HARTREE_CM
            else:
                record["J_error"] = f"degenerate <S^2> gap {denominator:.2e}"
            shown = (f"{record['J_cm']:10.2f}" if "J_cm" in record
                     else f"{'INVALID':>10s}")
            print(f"  {degrees:8.0f} {shown} {s2_hs:7.3f} "
                  f"{s2_bs:7.3f} {record['hs_spin_Ni']:7.2f} "
                  f"{record['bs_spin_Ni']:7.2f} "
                  f"{record['bs_spin_tether_carbon']:+9.3f}  {how_q}/{how_bs}")
            if "J_invalid" in record:
                print(f"           {record['J_invalid'][:96]}")
            # Only hand on a density that found the intended state; seeding the
            # next torsion from a collapsed one would propagate the collapse.
            if abs(record["hs_spin_Ni"]) > 0.5:
                previous_quartet_dm = mf_q.make_rdm1()
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-800:]
            print(f"  {degrees:8.0f}  FAILED {record['error'][:70]}")
        record["seconds"] = time.time() - started
        emit(record)
        results.append(record)

    # "J_cm" is only ever set on points whose determinants passed the
    # population gate, so this selection is the gate rather than a restatement
    # of it.  The first version of this line took every point that had a J at
    # all, which admitted a -4017 cm^-1 spin-state gap into the verdict and
    # flipped its conclusion.
    good = [r for r in results if "J_cm" in r]
    rejected = [r for r in results if "J_invalid" in r]
    verdict = {"event": "verdict", "points": len(good),
               "rejected": [{"torsion": r["torsion"], "why": r["J_invalid"]}
                            for r in rejected]}
    if rejected:
        print(f"\n  {len(rejected)} of {len(results)} points rejected: their "
              f"determinants were not\n  the states being compared, so their "
              f"energy difference is not an exchange coupling.")
    if good:
        j_values = [r["J_cm"] for r in good]
        rho = [r["bs_spin_tether_carbon"] for r in good]
        verdict.update(
            J_range=[min(j_values), max(j_values)],
            J_changes_sign=bool(min(j_values) < 0 < max(j_values)),
            rho_changes_sign=bool(min(rho) < 0 < max(rho)),
            rho_range=[min(rho), max(rho)])
        # The whole point: rho holds its sign by construction, so a sign change
        # in J cannot be attributed to one in rho.
        verdict["sign_of_J_follows_sign_of_rho"] = not (
            verdict["J_changes_sign"] and not verdict["rho_changes_sign"])
        # What is J actually a function of?  If it tracks the Ni...O distance
        # rather than the torsion, this is a distance scan wearing a torsion
        # scan's label, and saying so is the difference between a result and a
        # mislabelled one.
        distances = [r["ni_nitroxide_oxygen"] for r in good]
        if len(good) >= 3:
            import math as _m
            def _corr(x, y):
                mx, my = sum(x) / len(x), sum(y) / len(y)
                cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
                sx = _m.sqrt(sum((a - mx) ** 2 for a in x))
                sy = _m.sqrt(sum((b - my) ** 2 for b in y))
                return cov / (sx * sy) if sx and sy else float("nan")
            verdict["corr_J_vs_ni_oxygen"] = _corr(j_values, distances)
            verdict["corr_J_vs_cos2_torsion"] = _corr(
                j_values, [_m.cos(_m.radians(r["torsion"])) ** 2 for r in good])
            verdict["corr_lnJ_vs_ni_oxygen"] = _corr(
                [_m.log(abs(v)) for v in j_values], distances)
            print(f"\n  corr(J, Ni...O)  {verdict['corr_J_vs_ni_oxygen']:+.3f}"
                  f"   corr(J, cos^2 torsion) {verdict['corr_J_vs_cos2_torsion']:+.3f}"
                  f"   corr(ln J, Ni...O) {verdict['corr_lnJ_vs_ni_oxygen']:+.3f}")
        print(f"\n  J spans {min(j_values):+.2f} to {max(j_values):+.2f} cm-1")
        print(f"  rho at the tether carbon spans {min(rho):+.3f} to {max(rho):+.3f} e")
        if verdict["J_changes_sign"] and not verdict["rho_changes_sign"]:
            print("  -> J CHANGES SIGN while rho does not: sign(J) is not a "
                  "function of sign(rho)")
        elif not verdict["J_changes_sign"]:
            print(f"  -> J keeps one sign across {len(good)} valid points; the "
                  f"test is INCONCLUSIVE,\n     not confirmatory -- it can "
                  f"refute 'sign(J) follows sign(rho)' but never establish it")
    emit(verdict)
    emit({"event": "done"})
    handle.close()
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
