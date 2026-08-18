#!/usr/bin/env python3
"""Exchange coupling for the operando reporters on Ni(salen), on a Colab GPU.

Same measurement as ``pentacene_readout.py`` and the same broken-symmetry
construction, run on the reporters that could survive a working reactor: a
phenalenyl, a nitronyl nitroxide, and an imino nitroxide bonded straight to the
catalyst with no phenylene spacer.

Three things make this run different from the colour-centre ones, and each is a
place the earlier work went wrong before it was caught:

**The reporter is not always a carbon radical.**  On a nitroxide the spin lives
on the O-N-C-N-O unit, so ``bs_spin_colour_centre`` summed over the whole
fragment is the right quantity but the per-atom breakdown is what shows whether
the SOMO stayed put.  Both are recorded.

**Nitronyl nitroxide should couple weakly through pi and the control says so.**
Its SOMO is antisymmetric about the aryl-bearing carbon, so the pi pathway is
closed by symmetry.  Imino nitroxide has the same skeleton with one oxygen
removed and no such node.  Running both means a weak J on the nitronyl
nitroxide can be attributed rather than merely observed -- if the imino control
is also weak, the node is not the explanation.

**J only means something if the metal is magnetic.**  Every point records the
metal's spin population in both determinants and flags the ones where the
"exchange coupling" is between a magnetic metal and the reporter rather than
between two closed-shell fragments that happen to differ in energy.

Usage on the VM:

    python reporter_readout.py --out out.jsonl \\
        --reporters nitronyl-nitroxide,imino-nitroxide,phenalenyl --desmethyl
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

HARTREE_EV = 27.211386245988
HARTREE_CM = 219474.6313632
HARTREE_KCAL = 627.5094740631
XC = "HYB_GGA_XC_B3LYP"   # matches Psi4's b3lyp (VWN_RPA), unlike bare 'b3lyp'


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


def atom_spin(mf, mol) -> np.ndarray:
    dm = mf.make_rdm1()
    spin_ao = (to_numpy(dm[0]) - to_numpy(dm[1])) @ mol.intor_symmetric("int1e_ovlp")
    per_atom = np.zeros(mol.natm)
    diag = spin_ao.diagonal()
    for mu, (atom_index, *_rest) in enumerate(mol.ao_labels(fmt=None)):
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


def quench_fragment_spin(dm, ao_index: np.ndarray):
    """Average alpha/beta on a fragment -> the closed-shell-metal guess."""
    dm_a, dm_b = to_numpy(dm[0]).copy(), to_numpy(dm[1]).copy()
    block = np.ix_(ao_index, ao_index)
    average = 0.5 * (dm_a[block] + dm_b[block])
    dm_a[block] = average
    dm_b[block] = average
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


def run_reporter(reporter, args, out, geo, cat, gto, tduks):
    built, info = cat.reporter_on_salen(
        reporter, site=args.site, methylated=not args.desmethyl)
    # N_labile is the arm OPPOSITE the reporter, so the signal travels through
    # the metal rather than along the tether.  Opening the tethered arm instead
    # measures a different degradation mode and was a real mistake in an
    # earlier run.
    ni, n_labile = info["Ni"], info["N_labile"]
    n_reporter = info["n_host_atoms"]
    fragment = list(range(n_reporter))

    heavy = [i for i in fragment if built.symbols[i] != "H"]
    out.write(json.dumps({
        "event": "reporter_start", "reporter": reporter, "site": args.site,
        "methylated": not args.desmethyl, "natoms": len(built),
        "formula": built.formula, "reporter_block": [0, n_reporter - 1],
        "ni_reporter_closest_heavy": min(built.distance(ni, i) for i in heavy),
        "ni_tether_carbon": built.distance(ni, info["reporter_carbon"]),
    }) + "\n")

    from pyscf.geomopt.geometric_solver import optimize
    t0 = time.time()
    mol0 = gto.M(atom=built.to_psi4(), basis=args.basis, charge=0, spin=1, verbose=0)
    mf0, _ = make_scf(mol0)
    mol_eq = optimize(mf0, maxsteps=args.maxsteps)
    relaxed = geo.Structure.from_psi4("\n".join(
        f"{mol_eq.atom_symbol(i):<3s} " +
        " ".join(f"{c * 0.529177210903:14.8f}" for c in mol_eq.atom_coord(i))
        for i in range(mol_eq.natm)))
    out.write(json.dumps({"event": "optimised", "reporter": reporter,
                          "seconds": time.time() - t0,
                          "converged_in_maxsteps": bool(args.maxsteps > 15),
                          "xyz": relaxed.to_psi4()}) + "\n")

    targets = [("intact", None)] + [(f"{float(x):.2f}", float(x))
                                    for x in args.distances.split(",")]

    for label, distance in targets:
        record = {"event": "point", "reporter": reporter, "label": label,
                  "distance": distance}
        started = time.time()
        try:
            # Salen is one chain, so the arm must be opened by rotating about
            # real torsions; a single rigid hinge leaves the bond angles at the
            # pivot free and bends the diamine sp3 carbon past linear.
            structure = (relaxed if distance is None
                         else cat.open_salen_arm(relaxed, ni, n_labile, distance))
            record["ni_n"] = structure.distance(ni, n_labile)
            text = structure.to_psi4()

            # ---- quartet: metal S=1 parallel to the reporter S=1/2 ---------
            mol_q = gto.M(atom=text, basis=args.basis, charge=0, spin=3, verbose=0)
            mf_q, e_hs = make_scf(mol_q)
            s2_hs = float(mf_q.spin_square()[0])
            pops_q = atom_spin(mf_q, mol_q)
            record.update(e_hs=e_hs, s2_hs=s2_hs, hs_converged=bool(mf_q.converged),
                          hs_spin_Ni=float(pops_q[ni]),
                          hs_spin_reporter=float(pops_q[fragment].sum()))

            ao_fragment = ao_slice_for_atoms(mol_q, fragment)
            mol_d = gto.M(atom=text, basis=args.basis, charge=0, spin=1, verbose=0)

            # ---- broken symmetry: flip the reporter ------------------------
            mf_bs, e_bs = make_scf(
                mol_d, dm0=flip_fragment_spin(mf_q.make_rdm1(), ao_fragment))
            s2_bs = float(mf_bs.spin_square()[0])
            pops_bs = atom_spin(mf_bs, mol_d)
            record.update(e_bs=e_bs, s2_bs=s2_bs, bs_converged=bool(mf_bs.converged),
                          bs_spin_Ni=float(pops_bs[ni]),
                          bs_spin_reporter=float(pops_bs[fragment].sum()),
                          bs_spin_tether_carbon=float(pops_bs[info["reporter_carbon"]]))

            # ---- closed-shell metal: quench the metal, keep the reporter ---
            # A level shift discourages the SCF from sliding back into the
            # magnetic solution, which is lower once the arm is out.
            mf_cs, e_cs = make_scf(
                mol_d, dm0=quench_fragment_spin(mf_q.make_rdm1(),
                                                ao_slice_for_atoms(mol_q, [ni])),
                level_shift=0.25)
            cs_metal = float(atom_spin(mf_cs, mol_d)[ni])
            record.update(e_closed_shell=e_cs, cs_spin_Ni=cs_metal,
                          closed_shell_held=bool(abs(cs_metal) < 0.3))
            if record["closed_shell_held"]:
                record["metal_gap_kcal"] = (e_hs - e_cs) * HARTREE_KCAL

            # ---- is J an exchange coupling at all? -------------------------
            magnetic = abs(record["bs_spin_Ni"]) > 0.5
            antiparallel = record["bs_spin_Ni"] * record["bs_spin_reporter"] < 0
            record["metal_is_magnetic"] = bool(magnetic)
            record["fragments_antiparallel"] = bool(antiparallel)

            denominator = s2_hs - s2_bs
            if abs(denominator) > 1e-4:
                record["J_cm"] = (e_bs - e_hs) / denominator * HARTREE_CM
                record["J_meaningful"] = bool(magnetic and antiparallel)
            else:
                record["J_error"] = f"degenerate <S^2> difference {denominator:.2e}"

            # ---- optical readout on both doublets --------------------------
            for tag, mf_ref in (("bs", mf_bs), ("cs", mf_cs)):
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
    parser.add_argument("--reporters",
                        default="nitronyl-nitroxide,imino-nitroxide,phenalenyl")
    parser.add_argument("--site", default="C3")
    parser.add_argument("--desmethyl", action="store_true",
                        help="drop the nitroxide methyls: same SOMO, 12 fewer atoms")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--states", type=int, default=14)
    parser.add_argument("--maxsteps", type=int, default=60)
    parser.add_argument("--distances", default="2.40,2.90,3.60,4.50")
    args = parser.parse_args()

    import geometry as geo          # noqa: F401  (imported by catalysts)
    import catalysts as cat
    from pyscf import gto
    from gpu4pyscf.tdscf import uks as tduks
    import cupy

    device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    with open(args.out, "w", buffering=1) as out:
        out.write(json.dumps({
            "event": "start", "device": device, "basis": args.basis,
            "states": args.states, "site": args.site,
            "desmethyl": args.desmethyl,
            "reporters": args.reporters.split(",")}) + "\n")
        for reporter in args.reporters.split(","):
            try:
                run_reporter(reporter.strip(), args, out, geo, cat, gto, tduks)
            except Exception as exc:
                out.write(json.dumps({
                    "event": "reporter_failed", "reporter": reporter,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                    "traceback": traceback.format_exc()[-800:]}) + "\n")
        out.write(json.dumps({"event": "done"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
