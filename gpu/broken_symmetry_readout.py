#!/usr/bin/env python3
"""Broken-symmetry exchange coupling and the optical readout, on a Colab GPU.

Two things went wrong in the previous GPU pass and both are addressed here.

**The doublet SCF kept finding the wrong state.**  A doublet assembly has two
solutions at the same Ms: a closed-shell metal carrying the spin on the defect,
and an S = 1 metal antiferromagnetically coupled to the defect.  The default
guess lands on the first at every geometry, so no exchange pathway is ever
sampled and the resulting "J" is meaningless.  The fix is to *construct* the
broken-symmetry determinant instead of hoping for it: converge the quartet,
swap the alpha and beta density blocks on the colour-centre fragment, and
re-converge at Ms = 1/2.  That starts the SCF with the metal spin-up and the
defect spin-down, which is the state we actually want.

**Oscillator strengths never computed.**  Both ``oscillator_strength()`` and the
``transition_dipole()`` fallback raised, so every state came back dark and no
optical readout existed.  They are computed here directly from the TDA
amplitudes and the dipole integrals, which needs nothing from the library
beyond the eigenvectors.

Whether J is *meaningful* at a given geometry is a separate question from
whether it can be computed, so the script records the metal's spin population in
both determinants and flags points where the metal is not actually magnetic.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

HARTREE_EV = 27.211386245988
HARTREE_CM = 219474.6313632
XC = "HYB_GGA_XC_B3LYP"


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def make_scf(mol, dm0=None):
    from gpu4pyscf import dft

    mf = dft.UKS(mol, xc=XC).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
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


def flip_fragment_spin(dm, ao_index: np.ndarray):
    """Swap alpha/beta density on a fragment -> the broken-symmetry guess."""
    dm_a, dm_b = to_numpy(dm[0]).copy(), to_numpy(dm[1]).copy()
    block = np.ix_(ao_index, ao_index)
    a_block = dm_a[block].copy()
    dm_a[block] = dm_b[block]
    dm_b[block] = a_block
    return np.array([dm_a, dm_b])


def oscillator_strengths(td, mf, mol, energies_hartree):
    """f = (2/3) dE |mu|^2 from the TDA amplitudes and dipole integrals.

    Computed directly because gpu4pyscf's own routines raised on this system.
    """
    dip_ao = mol.intor("int1e_r", comp=3)
    mo_a, mo_b = to_numpy(mf.mo_coeff[0]), to_numpy(mf.mo_coeff[1])
    occ_a, occ_b = to_numpy(mf.mo_occ[0]), to_numpy(mf.mo_occ[1])

    o_a, v_a = mo_a[:, occ_a > 0], mo_a[:, occ_a == 0]
    o_b, v_b = mo_b[:, occ_b > 0], mo_b[:, occ_b == 0]
    d_a = np.array([o_a.T @ dip_ao[x] @ v_a for x in range(3)])
    d_b = np.array([o_b.T @ dip_ao[x] @ v_b for x in range(3)])

    out = []
    for n, energy in enumerate(energies_hartree):
        x = td.xy[n][0]
        xa, xb = to_numpy(x[0]), to_numpy(x[1])
        mu = np.array([
            float((xa * d_a[k]).sum() + (xb * d_b[k]).sum()) for k in range(3)
        ])
        out.append(float(2.0 / 3.0 * float(energy) * (mu ** 2).sum()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--xyz", required=True, help="relaxed intact assembly")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--distances", default="1.87,2.35,2.90,3.30,3.80,4.50")
    args = parser.parse_args()

    import geometry as geo
    from pyscf import gto
    from gpu4pyscf.tdscf import uks as tduks
    import cupy

    device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    relaxed = geo.Structure.from_xyz(open(args.xyz).read())
    _, info = geo.sensor_assembly()
    ni, n_labile, n_cat = info["Ni"], info["N_labile"], info["n_catalyst_atoms"]
    hinge = geo.chelate_hinge(relaxed, ni, n_labile)
    colour_centre_atoms = list(range(n_cat, len(relaxed)))

    out = open(args.out, "w", buffering=1)
    out.write(json.dumps({"event": "start", "device": device,
                          "natoms": len(relaxed), "basis": args.basis,
                          "colour_centre_atoms": [n_cat, len(relaxed) - 1]}) + "\n")

    for target in [float(x) for x in args.distances.split(",")]:
        record = {"event": "point", "distance": target}
        started = time.time()
        try:
            structure = geo.swing_arm(relaxed, ni, n_labile, target, hinge=hinge)
            text = structure.to_psi4()

            # ---- high spin: metal S=1 parallel to the defect S=1/2 --------
            mol_q = gto.M(atom=text, basis=args.basis, charge=0, spin=3, verbose=0)
            mf_q, e_hs = make_scf(mol_q)
            s2_hs = float(mf_q.spin_square()[0])
            pops_q = atom_spin(mf_q, mol_q)
            record.update(e_hs=e_hs, s2_hs=s2_hs, hs_converged=bool(mf_q.converged),
                          hs_spin_Ni=float(pops_q[ni]),
                          hs_spin_colour_centre=float(pops_q[n_cat:].sum()))

            # ---- broken symmetry: flip the colour centre ------------------
            ao_cc = ao_slice_for_atoms(mol_q, colour_centre_atoms)
            dm_bs = flip_fragment_spin(mf_q.make_rdm1(), ao_cc)
            mol_d = gto.M(atom=text, basis=args.basis, charge=0, spin=1, verbose=0)
            mf_bs, e_bs = make_scf(mol_d, dm0=dm_bs)
            s2_bs = float(mf_bs.spin_square()[0])
            pops_bs = atom_spin(mf_bs, mol_d)
            record.update(e_bs=e_bs, s2_bs=s2_bs, bs_converged=bool(mf_bs.converged),
                          bs_spin_Ni=float(pops_bs[ni]),
                          bs_spin_colour_centre=float(pops_bs[n_cat:].sum()))

            # ---- naive doublet, for comparison ----------------------------
            mf_naive, e_naive = make_scf(mol_d)
            record["e_naive_doublet"] = e_naive
            record["naive_spin_Ni"] = float(atom_spin(mf_naive, mol_d)[ni])
            record["bs_below_naive_kcal"] = (e_bs - e_naive) * 627.5094740631

            # Is the metal actually magnetic in these determinants?  Without
            # that, J is not an exchange coupling at all.
            magnetic = abs(record["hs_spin_Ni"]) > 0.5 and abs(record["bs_spin_Ni"]) > 0.3
            antiparallel = record["bs_spin_Ni"] * record["bs_spin_colour_centre"] < 0
            record["metal_is_magnetic"] = bool(magnetic)
            record["fragments_antiparallel"] = bool(antiparallel)

            denominator = s2_hs - s2_bs
            if abs(denominator) > 1e-4:
                record["J_cm"] = (e_bs - e_hs) / denominator * HARTREE_CM
                record["J_meaningful"] = bool(magnetic and antiparallel)
            else:
                record["J_error"] = f"degenerate <S^2> difference {denominator:.2e}"

            # ---- optical readout on the lower doublet ---------------------
            mf_opt = mf_bs if e_bs <= e_naive else mf_naive
            record["optical_from"] = "broken_symmetry" if e_bs <= e_naive else "naive"
            td = tduks.TDA(mf_opt)
            td.nstates = args.states
            td.kernel()
            energies = [float(e) for e in td.e]
            try:
                strengths = oscillator_strengths(td, mf_opt, mol_d, energies)
            except Exception as exc:
                record["oscillator_error"] = f"{type(exc).__name__}: {exc}"[:200]
                strengths = [None] * len(energies)
            record["states"] = [
                {"index": i + 1, "energy_ev": e * HARTREE_EV,
                 "energy_nm": (1239.841984 / (e * HARTREE_EV)) if e > 0 else None,
                 "oscillator_strength": f}
                for i, (e, f) in enumerate(zip(energies, strengths))
            ]
            bright = [s for s in record["states"]
                      if s["oscillator_strength"] and s["oscillator_strength"] >= 0.01]
            if bright:
                record["lowest_bright_ev"] = bright[0]["energy_ev"]
                record["lowest_bright_nm"] = bright[0]["energy_nm"]
                record["lowest_bright_f"] = bright[0]["oscillator_strength"]

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
