#!/usr/bin/env python3
"""TD-DFT on the 54-atom sensor assembly, for the Colab GPU runtime.

This is the calculation that does not fit on the CPU budget: the coupled
optical readout, where the colour centre's transition is computed with the
nickel catalyst actually attached, at several points along the ligand-loss
coordinate.  On four cores a single point is hours; on an A100 it is minutes.

Runs detached and writes newline-delimited JSON so progress survives the
client-side execution timeouts of ``colab exec``.  Geometries come in from a
JSON file so all structure logic stays in the repo rather than being duplicated
on the VM.

    python assembly_tddft.py --geoms scan_geoms.json --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

BOHR = 0.529177210903
HARTREE_EV = 27.211386245988

#: Psi4's "b3lyp" uses VWN_RPA correlation.  PySCF's bare 'b3lyp' string has
#: meant the VWN5 variant in some releases, which would put the CPU and GPU
#: halves of this project on subtly different functionals.  Name it explicitly.
XC_ALIASES = {
    "b3lyp": "HYB_GGA_XC_B3LYP",
    "pbe0": "PBE0",
    "bp86": "BP86",
    "wb97x-d": "wb97x-d",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geoms", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--functional", default="b3lyp")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=1, help="n_alpha - n_beta")
    parser.add_argument("--states", type=int, default=10)
    args = parser.parse_args()

    payload = json.load(open(args.geoms))
    geometries = payload["geometries"]
    info = payload.get("info", {})

    from pyscf import gto
    from pyscf.lib import logger  # noqa: F401
    import cupy
    from gpu4pyscf import dft
    from gpu4pyscf.tdscf import uks as tduks

    device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    xc = XC_ALIASES.get(args.functional, args.functional)

    with open(args.out, "w", buffering=1) as fh:
        fh.write(json.dumps({
            "event": "start", "device": device, "functional": args.functional,
            "xc": xc, "basis": args.basis, "states": args.states,
            "points": sorted(geometries),
        }) + "\n")

        for label in sorted(geometries, key=float):
            record: dict = {"event": "point", "distance": float(label)}
            started = time.time()
            try:
                mol = gto.M(atom=geometries[label], basis=args.basis,
                            charge=args.charge, spin=args.spin, verbose=0)
                record["natm"] = mol.natm
                record["nao"] = int(mol.nao)

                mf = dft.UKS(mol, xc=xc).density_fit()
                mf.grids.level = 3
                mf.conv_tol = 1e-9
                mf.max_cycle = 300
                energy = float(mf.kernel())
                if not mf.converged:
                    # Near the spin crossing the first-order SCF stalls; the
                    # second-order (Newton) solver handles the near-degeneracy.
                    record["scf_retry"] = "newton"
                    mf = mf.newton()
                    mf.max_cycle = 100
                    energy = float(mf.kernel())
                record["scf_energy"] = energy
                record["scf_converged"] = bool(mf.converged)
                record["scf_seconds"] = time.time() - started

                # <S^2> and multiplicity, the spin diagnostic the CPU path also
                # reports, so the two halves are directly comparable.
                try:
                    s2, mult = mf.spin_square()
                    record["s2"] = float(s2)
                    record["multiplicity"] = float(mult)
                except Exception as exc:
                    record["s2_error"] = str(exc)[:200]

                # Mulliken spin populations -> where the unpaired electron sits.
                try:
                    dm = mf.make_rdm1()
                    dm_a = cupy.asnumpy(dm[0]) if hasattr(dm[0], "get") else dm[0]
                    dm_b = cupy.asnumpy(dm[1]) if hasattr(dm[1], "get") else dm[1]
                    ovlp = mol.intor_symmetric("int1e_ovlp")
                    spin_pop = ((dm_a - dm_b) @ ovlp).diagonal()
                    per_atom = [0.0] * mol.natm
                    for mu, (atom_index, *_rest) in enumerate(mol.ao_labels(fmt=None)):
                        per_atom[atom_index] += float(spin_pop[mu])
                    record["mulliken_spin"] = per_atom
                    for key in ("Ni", "sp3_carbon", "N_labile"):
                        if key in info:
                            record[f"spin_on_{key}"] = per_atom[info[key]]
                    n_cat = info.get("n_catalyst_atoms")
                    if n_cat:
                        record["spin_catalyst"] = sum(per_atom[:n_cat])
                        record["spin_colour_centre"] = sum(per_atom[n_cat:])
                except Exception as exc:
                    record["population_error"] = f"{type(exc).__name__}: {exc}"[:200]

                # TD-DFT: the optical readout.
                t_td = time.time()
                td = tduks.TDA(mf)
                td.nstates = args.states
                td.kernel()
                energies_ev = [float(e) * HARTREE_EV for e in td.e]
                try:
                    strengths = [float(f) for f in td.oscillator_strength()]
                except Exception:
                    strengths = [None] * len(energies_ev)
                record["states"] = [
                    {"index": i + 1, "energy_ev": e,
                     "energy_nm": (1239.841984 / e) if e > 0 else None,
                     "oscillator_strength": f}
                    for i, (e, f) in enumerate(zip(energies_ev, strengths))
                ]
                bright = [s for s in record["states"]
                          if s["oscillator_strength"] and s["oscillator_strength"] >= 0.01]
                if bright:
                    record["lowest_bright_ev"] = bright[0]["energy_ev"]
                    record["lowest_bright_nm"] = bright[0]["energy_nm"]
                    record["lowest_bright_f"] = bright[0]["oscillator_strength"]
                record["tddft_seconds"] = time.time() - t_td

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"[:400]
                record["traceback"] = traceback.format_exc()[-800:]

            record["total_seconds"] = time.time() - started
            fh.write(json.dumps(record) + "\n")

        fh.write(json.dumps({"event": "done"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
