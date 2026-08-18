#!/usr/bin/env python3
"""Isolate the closed-shell-metal doublet, to finish the perturbation measurement.

The tethered construct has (at least) three doublet solutions at Ms = 1/2:

    magnetic, antiparallel   metal S=1 coupled to the defect S=1/2 -- reached by
                             flipping the defect's spin blocks (broken symmetry)
    magnetic, from the       what the default guess finds; on this system it is
    default guess            *also* a magnetic metal (Ni spin 0.92-1.18)
    closed-shell metal       metal S=0, spin entirely on the defect -- the state
                             the perturbation comparison needs, and the one
                             nothing so far has actually produced

Assuming the default guess gives the third is what invalidated the first attempt
at this measurement: E(quartet) - E(default) turned out to be a quartet/
broken-symmetry exchange splitting worth tenths of a kcal/mol, and differencing
it against a genuine ~20 kcal/mol bare-catalyst gap manufactured an apparent
20 kcal/mol "perturbation" that was purely the mismatch.

The construction here is the mirror of the broken-symmetry one.  Rather than
flipping the defect's alpha/beta blocks, it *quenches the metal's*: average the
two spin densities over the nickel's atomic orbitals so the guess carries no
metal spin, keep the defect's, and re-converge.  Whether the SCF stays there is
an empirical question -- at long Ni-N the magnetic metal is the ground state, so
the closed-shell solution is excited and may relax away.  The metal's spin
population is recorded after convergence so that can be seen rather than
assumed.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

HARTREE_KCAL = 627.5094740631
XC = "HYB_GGA_XC_B3LYP"


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def make_scf(mol, dm0=None, level_shift=0.0):
    from gpu4pyscf import dft

    mf = dft.UKS(mol, xc=XC).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    if level_shift:
        mf.level_shift = level_shift
    energy = float(mf.kernel(dm0=dm0))
    if not mf.converged:
        mf = mf.newton()
        mf.max_cycle = 150
        energy = float(mf.kernel(dm0=mf.make_rdm1()))
    return mf, energy


def atom_spin(mf, mol):
    dm = mf.make_rdm1()
    spin_ao = (to_numpy(dm[0]) - to_numpy(dm[1])) @ mol.intor_symmetric("int1e_ovlp")
    per_atom = np.zeros(mol.natm)
    diag = spin_ao.diagonal()
    for mu, (atom_index, *_r) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom_index] += float(diag[mu])
    return per_atom


def ao_slice_for_atoms(mol, atom_indices) -> np.ndarray:
    wanted = set(atom_indices)
    return np.array([mu for mu, (a, *_r) in enumerate(mol.ao_labels(fmt=None))
                     if a in wanted], dtype=int)


def quench_fragment_spin(dm, ao_index: np.ndarray):
    """Average alpha and beta density on a fragment: the mirror of flipping it.

    Leaves the rest of the molecule's spin polarisation intact, so the guess is
    "this fragment closed-shell, everything else as it was".
    """
    dm_a, dm_b = to_numpy(dm[0]).copy(), to_numpy(dm[1]).copy()
    block = np.ix_(ao_index, ao_index)
    average = 0.5 * (dm_a[block] + dm_b[block])
    dm_a[block] = average
    dm_b[block] = average
    return np.array([dm_a, dm_b])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--site", default="C3")
    parser.add_argument("--host", default="pyrene")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--maxsteps", type=int, default=15)
    parser.add_argument("--distances", default="2.90,3.60,4.50")
    args = parser.parse_args()

    import geometry as geo
    import catalysts as cat
    from pyscf import gto
    from pyscf.geomopt.geometric_solver import optimize
    import cupy

    built, info = cat.sensor_on_salen(site=args.site, host_name=args.host)
    ni, n_labile = info["Ni"], info["N_labile"]
    metal_only = [ni]

    out = open(args.out, "w", buffering=1)
    out.write(json.dumps({
        "event": "start",
        "device": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "host": args.host, "site": args.site, "natoms": len(built),
        "formula": built.formula, "basis": args.basis,
    }) + "\n")

    t0 = time.time()
    mol0 = gto.M(atom=built.to_psi4(), basis=args.basis, charge=0, spin=1, verbose=0)
    mf0, _ = make_scf(mol0)
    mol_eq = optimize(mf0, maxsteps=args.maxsteps)
    relaxed = geo.Structure.from_psi4("\n".join(
        f"{mol_eq.atom_symbol(i):<3s} " +
        " ".join(f"{c * 0.529177210903:14.8f}" for c in mol_eq.atom_coord(i))
        for i in range(mol_eq.natm)))
    out.write(json.dumps({"event": "optimised", "seconds": time.time() - t0,
                          "xyz": relaxed.to_psi4()}) + "\n")

    # The intact structure matters most: it is the sensor's resting state, and
    # the one geometry where the closed-shell metal is unambiguously the ground
    # state, so the comparison there is the cleanest.
    targets = [("intact", None)] + [
        (f"{float(x):.2f}", float(x)) for x in args.distances.split(",")]

    for label, distance in targets:
        record = {"event": "point", "label": label, "distance": distance}
        started = time.time()
        try:
            structure = (relaxed if distance is None
                         else cat.open_salen_arm(relaxed, ni, n_labile, distance))
            record["ni_n"] = structure.distance(ni, n_labile)
            text = structure.to_psi4()

            mol_q = gto.M(atom=text, basis=args.basis, charge=0, spin=3, verbose=0)
            mf_q, e_hs = make_scf(mol_q)
            record.update(e_quartet=e_hs, s2_quartet=float(mf_q.spin_square()[0]),
                          quartet_spin_Ni=float(atom_spin(mf_q, mol_q)[ni]))

            mol_d = gto.M(atom=text, basis=args.basis, charge=0, spin=1, verbose=0)
            ao_metal = ao_slice_for_atoms(mol_q, metal_only)
            dm_cs = quench_fragment_spin(mf_q.make_rdm1(), ao_metal)

            # A level shift discourages the SCF from sliding straight back into
            # the magnetic solution, which is lower in energy once the arm is out.
            mf_cs, e_cs = make_scf(mol_d, dm0=dm_cs, level_shift=0.25)
            pops = atom_spin(mf_cs, mol_d)
            metal_spin = float(pops[ni])
            record.update(e_closed_shell=e_cs,
                          s2_closed_shell=float(mf_cs.spin_square()[0]),
                          cs_spin_Ni=metal_spin,
                          cs_spin_colour_centre=float(
                              pops[:info["n_host_atoms"]].sum()))

            # Did the quench hold?  If the metal re-polarised, this is not the
            # closed-shell state and the gap below is not a spin-state gap.
            record["closed_shell_held"] = bool(abs(metal_spin) < 0.3)
            if record["closed_shell_held"]:
                record["metal_gap_kcal"] = (e_hs - e_cs) * HARTREE_KCAL
            else:
                record["metal_gap_invalid"] = (
                    f"metal re-polarised to {metal_spin:.2f} spin; the quench "
                    f"did not hold and this is not the closed-shell state")

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-700:]

        record["seconds"] = time.time() - started
        out.write(json.dumps(record) + "\n")

    out.write(json.dumps({"event": "done"}) + "\n")
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
