#!/usr/bin/env python3
"""Full sensor readout for the pentacene / Ni(salen) construct, on a Colab GPU.

Runs both attachment sites side by side so the distance dependence of the
exchange coupling gets a third data point:

    C5  para to the phenolate oxygen, the standard synthetic handle, 7.01 A
    C3  ortho to the oxygen, one bond from the chelate ring,          4.62 A

For reference, the pyrene study gave J ~ 0.4 cm^-1 at 8.75 A (unmeasurable) and
~6 cm^-1 at 4.46 A, so C3 should land near the useful end and C5 between.

At every geometry three determinants are computed:

    quartet         metal S=1 parallel to the defect S=1/2 (high spin)
    broken symmetry metal S=1 antiparallel to the defect, built deliberately by
                    flipping the alpha/beta density on the colour centre --
                    the default guess never finds this state on its own
    naive doublet   whatever the default guess converges to, usually a
                    closed-shell metal with the spin on the defect

TD-DFT runs on *both* doublets at *every* point.  Running it only on whichever
is lower in energy compares different electronic states at different points on
the coordinate, which previously produced a completely spurious result.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

HARTREE_EV = 27.211386245988
HARTREE_CM = 219474.6313632
XC = "HYB_GGA_XC_B3LYP"   # matches Psi4's b3lyp (VWN_RPA), unlike bare 'b3lyp'


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
    dm_a, dm_b = to_numpy(dm[0]).copy(), to_numpy(dm[1]).copy()
    block = np.ix_(ao_index, ao_index)
    a_block = dm_a[block].copy()
    dm_a[block] = dm_b[block]
    dm_b[block] = a_block
    return np.array([dm_a, dm_b])


def oscillator_strengths(td, mf, mol, energies_hartree):
    """f = (2/3) dE |mu|^2 from the TDA amplitudes and dipole integrals.

    Computed by hand because gpu4pyscf's oscillator_strength() and
    transition_dipole() both raise on these systems.
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
        xa, xb = to_numpy(td.xy[n][0][0]), to_numpy(td.xy[n][0][1])
        mu = np.array([float((xa * d_a[k]).sum() + (xb * d_b[k]).sum())
                       for k in range(3)])
        out.append(float(2.0 / 3.0 * float(energy) * (mu ** 2).sum()))
    return out


def run_site(site, args, out, geo, cat, gto, tduks):
    built, info = cat.pentacene_sensor(site=site)
    ni, n_labile = info["Ni"], info["N1"]
    # occ_radical places the host first, so the colour centre is the leading block.
    n_host = info["n_host_atoms"]
    colour_centre = list(range(n_host))

    out.write(json.dumps({
        "event": "site_start", "site": site, "natoms": len(built),
        "formula": built.formula,
        "ni_defect_distance": built.distance(ni, info["sp3_carbon"]),
        "colour_centre": [0, n_host - 1],
    }) + "\n")

    from pyscf.geomopt.geometric_solver import optimize
    t0 = time.time()
    mol0 = gto.M(atom=built.to_psi4(), basis=args.basis, charge=0, spin=1, verbose=0)
    mf0, _ = make_scf(mol0)
    mol_eq = optimize(mf0, maxsteps=args.maxsteps)
    lines = [f"{mol_eq.atom_symbol(i):<3s} " +
             " ".join(f"{c * 0.529177210903:14.8f}" for c in mol_eq.atom_coord(i))
             for i in range(mol_eq.natm)]
    relaxed = geo.Structure.from_psi4("\n".join(lines))
    out.write(json.dumps({"event": "optimised", "site": site,
                          "seconds": time.time() - t0,
                          "xyz": relaxed.to_psi4()}) + "\n")

    for target in [float(x) for x in args.distances.split(",")]:
        record = {"event": "point", "site": site, "distance": target}
        started = time.time()
        try:
            # Salen is one chain, so the arm must be opened by rotating about
            # real torsions; a single rigid hinge leaves the bond angles at the
            # pivot free and bends the diamine sp3 carbon past linear.
            structure = cat.open_salen_arm(relaxed, ni, n_labile, target)
            record["ni_n"] = structure.distance(ni, n_labile)
            text = structure.to_psi4()

            mol_q = gto.M(atom=text, basis=args.basis, charge=0, spin=3, verbose=0)
            mf_q, e_hs = make_scf(mol_q)
            s2_hs = float(mf_q.spin_square()[0])
            pops_q = atom_spin(mf_q, mol_q)
            record.update(e_hs=e_hs, s2_hs=s2_hs, hs_spin_Ni=float(pops_q[ni]))

            ao_cc = ao_slice_for_atoms(mol_q, colour_centre)
            dm_bs = flip_fragment_spin(mf_q.make_rdm1(), ao_cc)
            mol_d = gto.M(atom=text, basis=args.basis, charge=0, spin=1, verbose=0)
            mf_bs, e_bs = make_scf(mol_d, dm0=dm_bs)
            s2_bs = float(mf_bs.spin_square()[0])
            pops_bs = atom_spin(mf_bs, mol_d)
            record.update(e_bs=e_bs, s2_bs=s2_bs,
                          bs_spin_Ni=float(pops_bs[ni]),
                          bs_spin_colour_centre=float(pops_bs[colour_centre].sum()))

            mf_naive, e_naive = make_scf(mol_d)
            record["e_naive"] = e_naive
            record["naive_spin_Ni"] = float(atom_spin(mf_naive, mol_d)[ni])
            record["bs_minus_naive_kcal"] = (e_bs - e_naive) * 627.5094740631

            magnetic = abs(record["bs_spin_Ni"]) > 0.5
            antiparallel = record["bs_spin_Ni"] * record["bs_spin_colour_centre"] < 0
            record["metal_is_magnetic"] = bool(magnetic)
            record["fragments_antiparallel"] = bool(antiparallel)

            denom = s2_hs - s2_bs
            if abs(denom) > 1e-4:
                record["J_cm"] = (e_bs - e_hs) / denom * HARTREE_CM
                record["J_meaningful"] = bool(magnetic and antiparallel)
            else:
                record["J_error"] = f"degenerate <S^2> difference {denom:.2e}"

            for tag, mf_ref in (("bs", mf_bs), ("naive", mf_naive)):
                try:
                    td = tduks.TDA(mf_ref)
                    td.nstates = args.states
                    td.kernel()
                    energies = [float(e) for e in td.e]
                    strengths = oscillator_strengths(td, mf_ref, mol_d, energies)
                    states = [{"index": i + 1, "energy_ev": e * HARTREE_EV,
                               "energy_nm": (1239.841984 / (e * HARTREE_EV)) if e > 0 else None,
                               "oscillator_strength": f}
                              for i, (e, f) in enumerate(zip(energies, strengths))]
                    record[f"states_{tag}"] = states
                    record[f"max_f_{tag}"] = max(
                        (s["oscillator_strength"] or 0.0) for s in states)
                    bright = [s for s in states
                              if s["oscillator_strength"] and s["oscillator_strength"] >= 0.01]
                    if bright:
                        record[f"bright_ev_{tag}"] = bright[0]["energy_ev"]
                        record[f"bright_nm_{tag}"] = bright[0]["energy_nm"]
                        record[f"bright_f_{tag}"] = bright[0]["oscillator_strength"]
                except Exception as exc:
                    record[f"tddft_error_{tag}"] = f"{type(exc).__name__}: {exc}"[:200]

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-700:]

        record["seconds"] = time.time() - started
        out.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sites", default="C3,C5")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--states", type=int, default=14)
    parser.add_argument("--maxsteps", type=int, default=80)
    parser.add_argument("--distances", default="1.85,2.40,2.90,3.40,4.00,4.50")
    args = parser.parse_args()

    import geometry as geo          # noqa: F401  (imported by catalysts)
    import catalysts as cat
    from pyscf import gto
    from gpu4pyscf.tdscf import uks as tduks
    import cupy

    device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    with open(args.out, "w", buffering=1) as out:
        out.write(json.dumps({"event": "start", "device": device,
                              "basis": args.basis, "states": args.states,
                              "sites": args.sites.split(",")}) + "\n")
        for site in args.sites.split(","):
            try:
                run_site(site.strip(), args, out, geo, cat, gto, tduks)
            except Exception as exc:
                out.write(json.dumps({
                    "event": "site_failed", "site": site,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                    "traceback": traceback.format_exc()[-800:]}) + "\n")
        out.write(json.dumps({"event": "done"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
