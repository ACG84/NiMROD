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

from nimrod.catalysts import reporter_on_salen
from nimrod.config import DATA_DIR
from nimrod.geometry import _rotate_about

XC = "HYB_GGA_XC_B3LYP"
BASIS = "def2-svp"
HARTREE_CM = 219474.6313632
TORSIONS = (0.0, 30.0, 60.0, 90.0)


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
    for degrees in TORSIONS:
        record = {"event": "point", "torsion": degrees}
        started = time.time()
        try:
            structure = rotate_tether(built, info, degrees)
            text = structure.to_psi4()

            mol_q = gto.M(atom=text, basis=BASIS, charge=0, spin=3, verbose=0)
            mf_q, e_hs, how_q, tried_q = converge(mol_q, tag=f"quartet {degrees}")
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

            denominator = s2_hs - s2_bs
            if abs(denominator) > 1e-4:
                record["J_cm"] = (e_bs - e_hs) / denominator * HARTREE_CM
            else:
                record["J_error"] = f"degenerate <S^2> gap {denominator:.2e}"

            record["metal_is_magnetic"] = bool(abs(record["bs_spin_Ni"]) > 0.5)
            print(f"  {degrees:8.0f} "
                  f"{record.get('J_cm', float('nan')):10.2f} {s2_hs:7.3f} "
                  f"{s2_bs:7.3f} {record['hs_spin_Ni']:7.2f} "
                  f"{record['bs_spin_Ni']:7.2f} "
                  f"{record['bs_spin_tether_carbon']:+9.3f}  {how_q}/{how_bs}")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-800:]
            print(f"  {degrees:8.0f}  FAILED {record['error'][:70]}")
        record["seconds"] = time.time() - started
        emit(record)
        results.append(record)

    good = [r for r in results if "J_cm" in r]
    verdict = {"event": "verdict", "points": len(good)}
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
        print(f"\n  J spans {min(j_values):+.2f} to {max(j_values):+.2f} cm-1")
        print(f"  rho at the tether carbon spans {min(rho):+.3f} to {max(rho):+.3f} e")
        if verdict["J_changes_sign"] and not verdict["rho_changes_sign"]:
            print("  -> J CHANGES SIGN while rho does not: sign(J) is not a "
                  "function of sign(rho)")
        elif not verdict["J_changes_sign"]:
            print("  -> J keeps one sign across the scan; this test does not "
                  "separate the two accounts")
    emit(verdict)
    emit({"event": "done"})
    handle.close()
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
