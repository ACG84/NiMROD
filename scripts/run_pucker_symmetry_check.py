#!/usr/bin/env python
"""Did the pucker bug actually change the spin density, or only the argument?

The des-methyl nitronyl nitroxide was built with a 0.35 A ring pucker carried
over from the tetramethyl model, which has no business being there -- there are
no methyls to unclash -- and which destroys both mirror planes.  The molecule
that got computed was C2, not C2v.

That matters for the argument: under C2v the SOMO is A2 and no atomic orbital on
the C2 axis can contribute, so the node is exact over every basis function on
that carbon.  Under C2 only the pi component is protected; s and p_y amplitude on
C2 becomes symmetry-allowed.  So the earlier claim of an "exact symmetry node"
was false for the structure it was made about.

Whether it matters for the *numbers* is a separate question, and this answers it
by computing both geometries at the same levels.  Two outcomes, both worth
knowing:

    the populations barely move   the sigma leakage C2 allowed was negligible,
                                  the earlier spin densities stand as numbers,
                                  and only the symmetry claim was wrong

    the populations move          the bug contaminated the measurements too, and
                                  anything derived from them has to be recomputed

The imino nitroxide is built by deleting an oxygen from the nitronyl one, so it
inherits whichever geometry that used; it is included for the same reason.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg

import nimrod.reporters as reporters
from nimrod.config import DATA_DIR

FUNCTIONALS = (("pbe", "pbe"), ("HYB_GGA_XC_B3LYP", "b3lyp"), ("pbe0", "pbe0"))
BASES = ("def2-svp", "def2-tzvp")


def make_scf(mol, xc):
    from pyscf import dft

    mf = dft.UKS(mol, xc=xc).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    energy = float(mf.kernel())
    if not mf.converged:
        mf = mf.newton()
        mf.max_cycle = 150
        energy = float(mf.kernel(dm0=mf.make_rdm1()))
    return mf, energy


def _per_atom(mol, diagonal):
    per_atom = np.zeros(mol.natm)
    for mu, (atom, *_rest) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom] += float(diagonal[mu])
    return per_atom


def populations(mf, mol):
    dm = mf.make_rdm1()
    spin = np.asarray(dm[0]) - np.asarray(dm[1])
    overlap = mol.intor_symmetric("int1e_ovlp")
    root = scipy.linalg.sqrtm(overlap).real
    return (_per_atom(mol, (spin @ overlap).diagonal()),
            _per_atom(mol, (root @ spin @ root).diagonal()))


def sigma_pi_split(mf, mol, atom: int) -> tuple[float, float]:
    """Mulliken spin on one atom, split by whether the AO is out of plane.

    With the ring in the xy plane, the pi system is exactly the p_z manifold.
    The split is the whole point of the symmetry question: C2v protects every
    orbital on the axis, C2 protects only p_z, so a difference between the two
    geometries should appear in the sigma part and not the pi part.
    """
    dm = mf.make_rdm1()
    spin = np.asarray(dm[0]) - np.asarray(dm[1])
    diagonal = (spin @ mol.intor_symmetric("int1e_ovlp")).diagonal()
    sigma = pi = 0.0
    for mu, (index, _symbol, label, _m) in enumerate(mol.ao_labels(fmt=None)):
        if index != atom:
            continue
        if label.endswith("pz"):
            pi += float(diagonal[mu])
        else:
            sigma += float(diagonal[mu])
    return sigma, pi


def main() -> int:
    from pyscf import gto
    import pyscf

    out_path = DATA_DIR / "results" / "pucker_symmetry_check.jsonl"
    handle = out_path.open("w", buffering=1)

    def emit(record):
        handle.write(json.dumps(record) + "\n")

    emit({"event": "start", "pyscf": pyscf.__version__,
          "question": "does the C2v/C2 pucker bug change the populations, "
                      "or only the symmetry claim?"})

    print("Flat (C2v, node exact over all AOs) vs puckered (C2, only pi "
          "protected)\n")

    original = reporters.NN_PUCKER_DESMETHYL
    collected = {}
    for name, builder in (("nitronyl-nitroxide", reporters.nitronyl_nitroxide),
                          ("imino-nitroxide", reporters.imino_nitroxide)):
        print(f"{name} (des-methyl)")
        print(f"  {'geometry':<10s} {'level':<22s} {'<S^2>':>7s} {'Mulliken':>9s} "
              f"{'Lowdin':>9s} {'sigma':>8s} {'pi':>8s}")
        for geometry, pucker in (("flat", 0.0), ("puckered", 0.35)):
            reporters.NN_PUCKER_DESMETHYL = pucker
            struct, index = builder(methylated=False)
            centre = index["C2"]
            for xc, label in FUNCTIONALS:
                for basis in BASES:
                    record = {"event": "point", "molecule": name,
                              "geometry": geometry, "pucker": pucker,
                              "functional": label, "basis": basis}
                    started = time.time()
                    try:
                        mol = gto.M(atom=struct.to_psi4(), basis=basis,
                                    charge=0, spin=1, verbose=0)
                        mf, energy = make_scf(mol, xc)
                        mul, low = populations(mf, mol)
                        sigma, pi = sigma_pi_split(mf, mol, centre)
                        record.update(
                            energy=energy, converged=bool(mf.converged),
                            s2=float(mf.spin_square()[0]),
                            mulliken_C2=float(mul[centre]),
                            lowdin_C2=float(low[centre]),
                            sigma_C2=sigma, pi_C2=pi)
                        print(f"  {geometry:<10s} {label + '/' + basis:<22s} "
                              f"{record['s2']:7.4f} {record['mulliken_C2']:+9.4f} "
                              f"{record['lowdin_C2']:+9.4f} {sigma:+8.4f} {pi:+8.4f}")
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
                        record["traceback"] = traceback.format_exc()[-600:]
                        print(f"  {geometry:<10s} {label + '/' + basis:<22s} FAILED")
                    record["seconds"] = time.time() - started
                    emit(record)
                    collected[(name, geometry, label, basis)] = record
        print()
    reporters.NN_PUCKER_DESMETHYL = original

    verdict = {"event": "verdict"}
    print("Verdict")
    for name in ("nitronyl-nitroxide", "imino-nitroxide"):
        shifts, sigma_shifts, pi_shifts = [], [], []
        for xc, label in FUNCTIONALS:
            for basis in BASES:
                flat = collected.get((name, "flat", label, basis), {})
                bent = collected.get((name, "puckered", label, basis), {})
                if "mulliken_C2" not in flat or "mulliken_C2" not in bent:
                    continue
                shifts.append(bent["mulliken_C2"] - flat["mulliken_C2"])
                sigma_shifts.append(bent["sigma_C2"] - flat["sigma_C2"])
                pi_shifts.append(bent["pi_C2"] - flat["pi_C2"])
        if not shifts:
            continue
        verdict[name] = {
            "max_abs_population_shift": float(np.max(np.abs(shifts))),
            "max_abs_sigma_shift": float(np.max(np.abs(sigma_shifts))),
            "max_abs_pi_shift": float(np.max(np.abs(pi_shifts))),
            "bug_changed_the_numbers": bool(np.max(np.abs(shifts)) > 0.02),
        }
        v = verdict[name]
        print(f"  {name}: puckering moves rho(C2) by at most "
              f"{v['max_abs_population_shift']:.4f} e "
              f"(sigma {v['max_abs_sigma_shift']:.4f}, pi {v['max_abs_pi_shift']:.4f})")
        print(f"    -> the bug {'CHANGED THE NUMBERS' if v['bug_changed_the_numbers'] else 'affected only the symmetry claim'}")

    emit(verdict)
    emit({"event": "done"})
    handle.close()
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
