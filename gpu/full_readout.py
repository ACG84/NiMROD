#!/usr/bin/env python3
"""All three sensor readouts on the 54-atom assembly, on a Colab GPU.

Self-contained and detached: it relaxes the intact assembly, walks the
degradation coordinate with the same hinge rotation the CPU path uses, and at
each point computes the optical and magnetic readouts.  Only ``geometry.py``
comes along from the repo, so the structure logic is never duplicated.

Written as one detached job because ``colab exec`` imposes a client-side
execution timeout: a geometry optimisation that outlives it is killed
mid-flight even though the kernel is still working.

    python full_readout.py --out results.jsonl --states 12
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

HARTREE_EV = 27.211386245988
HARTREE_CM = 219474.6313632
BOHR = 0.529177210903

#: Psi4's b3lyp is VWN_RPA; PySCF's bare 'b3lyp' has meant VWN5 in some
#: releases.  Name it explicitly so both halves of the project agree.
XC = "HYB_GGA_XC_B3LYP"


def molecule_to_xyz(mol) -> str:
    lines = []
    for i in range(mol.natm):
        c = mol.atom_coord(i) * BOHR
        lines.append(f"{mol.atom_symbol(i):<3s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")
    return "\n".join(lines)


def spin_populations(mf, mol):
    import cupy
    import numpy as np

    dm = mf.make_rdm1()
    to_np = lambda a: cupy.asnumpy(a) if hasattr(a, "get") else np.asarray(a)  # noqa: E731
    spin_ao = (to_np(dm[0]) - to_np(dm[1])) @ mol.intor_symmetric("int1e_ovlp")
    per_atom = [0.0] * mol.natm
    for mu, (atom_index, *_r) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom_index] += float(spin_ao.diagonal()[mu])
    return per_atom


def run_scf(mol, xc=XC, level_shift=0.0):
    from gpu4pyscf import dft

    mf = dft.UKS(mol, xc=xc).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    if level_shift:
        mf.level_shift = level_shift
    energy = float(mf.kernel())
    if not mf.converged:
        mf = mf.newton()
        mf.max_cycle = 120
        energy = float(mf.kernel())
    return mf, energy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--maxsteps", type=int, default=80)
    parser.add_argument("--distances", default="1.87,2.35,2.90,3.30,4.50")
    args = parser.parse_args()

    import geometry as geo
    from pyscf import gto
    from pyscf.geomopt.geometric_solver import optimize
    from gpu4pyscf.tdscf import uks as tduks
    import cupy

    device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    targets = [float(x) for x in args.distances.split(",")]

    assembly, info = geo.sensor_assembly()
    ni, n_labile = info["Ni"], info["N_labile"]
    n_cat = info["n_catalyst_atoms"]

    out = open(args.out, "w", buffering=1)
    out.write(json.dumps({"event": "start", "device": device, "basis": args.basis,
                          "natoms": len(assembly), "info": info,
                          "distances": targets}) + "\n")

    # ---- 1. relax the intact assembly ------------------------------------
    t0 = time.time()
    try:
        mol = gto.M(atom=assembly.to_psi4(), basis=args.basis, charge=0, spin=1, verbose=0)
        mf, _ = run_scf(mol)
        mol_eq = optimize(mf, maxsteps=args.maxsteps)
        relaxed_xyz = molecule_to_xyz(mol_eq)
        relaxed = geo.Structure.from_psi4(relaxed_xyz)
        out.write(json.dumps({"event": "optimised", "seconds": time.time() - t0,
                              "xyz": relaxed_xyz,
                              "ni_n": relaxed.distance(ni, n_labile),
                              "min_contact": relaxed.min_interatomic_distance()}) + "\n")
    except Exception as exc:
        out.write(json.dumps({"event": "optimise_failed",
                              "error": f"{type(exc).__name__}: {exc}"[:400],
                              "traceback": traceback.format_exc()[-800:]}) + "\n")
        relaxed = assembly       # fall back to the builder geometry, flagged above

    hinge = geo.chelate_hinge(relaxed, ni, n_labile)

    # ---- 2. the readouts along the coordinate ----------------------------
    for target in targets:
        record = {"event": "point", "distance": target}
        started = time.time()
        try:
            structure = geo.swing_arm(relaxed, ni, n_labile, target, hinge=hinge)
            record["min_contact"] = structure.min_interatomic_distance()
            geometry_text = structure.to_psi4()

            # -- doublet: optical readout + where the spin sits ------------
            mol_d = gto.M(atom=geometry_text, basis=args.basis, charge=0, spin=1, verbose=0)
            mf_d, e_doublet = run_scf(mol_d)
            record["nao"] = int(mol_d.nao)
            record["e_doublet"] = e_doublet
            record["doublet_converged"] = bool(mf_d.converged)
            s2_d, _ = mf_d.spin_square()
            record["s2_doublet"] = float(s2_d)

            pops = spin_populations(mf_d, mol_d)
            record["spin_on_Ni"] = pops[ni]
            record["spin_catalyst"] = sum(pops[:n_cat])
            record["spin_colour_centre"] = sum(pops[n_cat:])

            td = tduks.TDA(mf_d)
            td.nstates = args.states
            td.kernel()
            energies = [float(e) * HARTREE_EV for e in td.e]
            try:
                strengths = [float(f) for f in td.oscillator_strength()]
            except Exception as exc:
                record["oscillator_error"] = f"{type(exc).__name__}: {exc}"[:200]
                # Fall back to computing f from the transition dipoles directly:
                # f = (2/3) * dE(au) * |mu|^2
                try:
                    dips = td.transition_dipole()
                    strengths = [float(2.0 / 3.0 * float(e) * float((d ** 2).sum()))
                                 for e, d in zip(td.e, dips)]
                    record["oscillator_source"] = "transition_dipole"
                except Exception:
                    strengths = [None] * len(energies)
            record["states"] = [
                {"index": i + 1, "energy_ev": e,
                 "energy_nm": (1239.841984 / e) if e > 0 else None,
                 "oscillator_strength": f}
                for i, (e, f) in enumerate(zip(energies, strengths))
            ]
            bright = [s for s in record["states"]
                      if s["oscillator_strength"] and s["oscillator_strength"] >= 0.01]
            if bright:
                record["lowest_bright_ev"] = bright[0]["energy_ev"]
                record["lowest_bright_nm"] = bright[0]["energy_nm"]
                record["lowest_bright_f"] = bright[0]["oscillator_strength"]

            # -- quartet: the high-spin partner for the exchange coupling --
            mol_q = gto.M(atom=geometry_text, basis=args.basis, charge=0, spin=3, verbose=0)
            mf_q, e_quartet = run_scf(mol_q)
            record["e_quartet"] = e_quartet
            record["quartet_converged"] = bool(mf_q.converged)
            s2_q, _ = mf_q.spin_square()
            record["s2_quartet"] = float(s2_q)
            record["spin_on_Ni_quartet"] = spin_populations(mf_q, mol_q)[ni]

            # Yamaguchi, H = -2 J S1.S2, J < 0 antiferromagnetic.  The doublet
            # is the broken-symmetry partner of the quartet here.
            denominator = float(s2_q) - float(s2_d)
            if abs(denominator) > 1e-4:
                j_hartree = (e_doublet - e_quartet) / denominator
                record["J_cm"] = j_hartree * HARTREE_CM
            else:
                record["J_error"] = f"degenerate <S^2> difference {denominator:.2e}"

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-800:]

        record["seconds"] = time.time() - started
        out.write(json.dumps(record) + "\n")

    out.write(json.dumps({"event": "done"}) + "\n")
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
