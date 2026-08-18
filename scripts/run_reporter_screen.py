#!/usr/bin/env python
"""Where does each reporter actually keep its spin?

The case for these reporters rests on a claim about spin density, and so far
that claim rests on Huckel theory and on symmetry arguments.  Both are cheap to
believe and cheap to check, and the difference between them decides which
molecule is worth an expensive exchange-coupling calculation:

    phenalenyl          Huckel says the non-bonding orbital has zero amplitude
                        at positions 2, 5 and 8 and equal amplitude at the other
                        six.  If DFT agrees, the tether position rule is real and
                        the reporter is usable; if the "nodes" carry a tenth of
                        an electron, the rule is a cartoon.

    nitronyl nitroxide  C2v symmetry says the SOMO is antisymmetric about C2, so
                        the aryl-bearing carbon is a node and the pi pathway to
                        the metal is closed.  That is a strong claim -- it means
                        this reporter cannot couple the way the colour centre
                        did -- and it should be believed only if the computed
                        spin population on C2 is near zero.

    imino nitroxide     the control.  Removing one oxygen breaks the symmetry,
                        so if the node story is right this molecule must show
                        real spin density on C2 while its parent does not.

Each is run as the free radical, which is where the argument lives: a fragment's
SOMO is a property of the fragment, and if the node survives attachment it is
because it was there to begin with.  The assembled couplings are a separate and
much more expensive question.

Mulliken populations are basis-set-dependent in magnitude but their *pattern* --
which atoms are near zero and which are not -- is what is being tested here, and
that is robust.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nimrod.config import DATA_DIR
from nimrod.geometry import Structure
from nimrod.reporters import (
    imino_nitroxide,
    nitronyl_nitroxide,
    phenalenyl,
    phenalenyl_nbmo,
    phenalenyl_sites,
)

XC = "HYB_GGA_XC_B3LYP"
BASIS = "def2-svp"


def make_scf(mol):
    from pyscf import dft

    mf = dft.UKS(mol, xc=XC).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.max_cycle = 300
    energy = float(mf.kernel())
    if not mf.converged:
        mf = mf.newton()
        mf.max_cycle = 150
        energy = float(mf.kernel(dm0=mf.make_rdm1()))
    return mf, energy


def atom_spin(mf, mol) -> np.ndarray:
    dm = mf.make_rdm1()
    spin_ao = (np.asarray(dm[0]) - np.asarray(dm[1])) @ mol.intor_symmetric("int1e_ovlp")
    per_atom = np.zeros(mol.natm)
    diag = spin_ao.diagonal()
    for mu, (atom_index, *_rest) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom_index] += float(diag[mu])
    return per_atom


def run_one(struct: Structure, labels: dict[str, int]) -> dict:
    from pyscf import gto

    mol = gto.M(atom=struct.to_psi4(), basis=BASIS, charge=0, spin=1, verbose=0)
    started = time.time()
    mf, energy = make_scf(mol)
    spin = atom_spin(mf, mol)
    return {
        "formula": struct.formula, "natoms": len(struct), "energy": energy,
        "converged": bool(mf.converged), "s2": float(mf.spin_square()[0]),
        "seconds": time.time() - started,
        "total_spin": float(spin.sum()),
        "spin": {label: float(spin[index]) for label, index in labels.items()},
        "spin_all": [float(x) for x in spin],
    }


def main() -> int:
    out_path = DATA_DIR / "results" / "reporter_spin_screen.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", buffering=1)

    def emit(record: dict) -> None:
        handle.write(json.dumps(record) + "\n")

    import pyscf

    emit({"event": "start", "pyscf": pyscf.__version__, "xc": XC, "basis": BASIS})

    jobs = []

    ply = phenalenyl()
    sites = phenalenyl_sites(ply)
    huckel = phenalenyl_nbmo(ply)
    ply_labels = {f"C{i}": i for i in sites["somo_bearing"] + sites["nodal"]}
    jobs.append(("phenalenyl", ply, ply_labels,
                 {"somo_bearing": sites["somo_bearing"], "nodal": sites["nodal"],
                  "huckel_squared": {str(i): huckel[i] ** 2 for i in huckel}}))

    for name, builder in (("nitronyl-nitroxide", nitronyl_nitroxide),
                          ("imino-nitroxide", imino_nitroxide)):
        for methylated in (True, False):
            struct, index = builder(methylated=methylated)
            tag = name if methylated else f"{name}-desmethyl"
            jobs.append((tag, struct, dict(index), {}))

    summary = {}
    for tag, struct, labels, extra in jobs:
        record = {"event": "reporter", "name": tag, **extra}
        try:
            record.update(run_one(struct, labels))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["traceback"] = traceback.format_exc()[-800:]
        emit(record)
        summary[tag] = record
        print(f"  {tag:<32s} {record.get('formula', '?'):<12s} "
              f"{record.get('seconds', 0):6.0f} s  "
              f"{'ok' if 'error' not in record else record['error'][:60]}")

    # --- the two claims, stated as numbers -------------------------------
    verdict: dict = {"event": "verdict"}

    ply_record = summary.get("phenalenyl", {})
    if "spin" in ply_record:
        bearing = [ply_record["spin"][f"C{i}"] for i in sites["somo_bearing"]]
        nodal = [ply_record["spin"][f"C{i}"] for i in sites["nodal"]]
        verdict["phenalenyl"] = {
            "mean_spin_on_somo_positions": float(np.mean(bearing)),
            "mean_spin_on_nodal_positions": float(np.mean(nodal)),
            "max_abs_spin_on_nodal": float(np.max(np.abs(nodal))),
            "contrast": (float(np.mean(bearing) / np.mean(np.abs(nodal)))
                         if np.mean(np.abs(nodal)) > 1e-9 else None),
        }

    for tag in ("nitronyl-nitroxide", "imino-nitroxide",
                "nitronyl-nitroxide-desmethyl", "imino-nitroxide-desmethyl"):
        record = summary.get(tag, {})
        if "spin" not in record:
            continue
        oxygens = [v for k, v in record["spin"].items() if k.startswith("O_")]
        nitrogens = [record["spin"][k] for k in ("N1", "N3") if k in record["spin"]]
        verdict[tag] = {
            "spin_on_C2": record["spin"].get("C2"),
            "spin_on_nitrogens": nitrogens,
            "spin_on_oxygens": oxygens,
            "fraction_on_NO_unit": (
                (sum(nitrogens) + sum(oxygens)) / record["total_spin"]
                if record.get("total_spin") else None),
        }

    emit(verdict)
    emit({"event": "done"})
    handle.close()

    print()
    if "phenalenyl" in verdict:
        v = verdict["phenalenyl"]
        print(f"phenalenyl   SOMO positions carry {v['mean_spin_on_somo_positions']:+.4f} "
              f"e each, nodal positions {v['mean_spin_on_nodal_positions']:+.4f}")
        if v["contrast"]:
            print(f"             contrast {v['contrast']:.0f}x  -- the tether rule "
                  f"{'holds' if v['contrast'] > 5 else 'DOES NOT hold'}")
    for tag in ("nitronyl-nitroxide-desmethyl", "imino-nitroxide-desmethyl"):
        if tag in verdict:
            v = verdict[tag]
            print(f"{tag:<30s} spin on C2 {v['spin_on_C2']:+.4f} e, "
                  f"{v['fraction_on_NO_unit']:.0%} of the spin on the N-O unit")
    node = verdict.get("nitronyl-nitroxide-desmethyl", {}).get("spin_on_C2")
    control = verdict.get("imino-nitroxide-desmethyl", {}).get("spin_on_C2")
    if node is not None and control is not None:
        print(f"\nnode test: |C2| is {abs(node):.4f} on nitronyl nitroxide against "
              f"{abs(control):.4f} on the\n           symmetry-broken control "
              f"-- {'node confirmed' if abs(node) < abs(control) / 3 else 'NO node'}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
