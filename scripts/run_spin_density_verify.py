#!/usr/bin/env python
"""Is the negative spin density at the tether carbon real, or a partitioning artefact?

The screen produced a result that reverses the design argument: the carbons the
symmetry analysis calls nodes carry *negative* spin density rather than none, and
on nitronyl nitroxide the "node" carbon C2 carries more spin (-0.272 e) than the
same carbon on the symmetry-broken imino nitroxide control (-0.144 e).  That is
the opposite of what a node is supposed to do, and if it is right it changes
which reporter is worth building.

Before it changes anything it has to survive three attacks, because every one of
them could produce that number on its own:

**Mulliken populations are basis-set dependent.**  They divide the overlap
density evenly between two atoms regardless of electronegativity, and the split
gets worse as the basis grows more diffuse.  A small negative population on a
carbon flanked by two nitrogens is exactly the shape of a Mulliken artefact.
Lowdin populations, which symmetrically orthogonalise first, are computed
alongside as an independent partitioning.

**Spin polarisation is functional dependent.**  The negative density at a nodal
position comes from alpha/beta core polarisation, and how much of it a functional
produces scales with its fraction of exact exchange -- more Hartree-Fock exchange
means more spin polarisation and more spin contamination.  If the sign flips
across the ladder, it is a functional artefact; if only the magnitude moves, it
is physics.

**The result could be spin contamination.**  A UKS doublet with <S^2> well above
0.75 is not a doublet, and its spin density is not the doublet's spin density.
<S^2> is recorded at every level so a contaminated point can be seen rather than
averaged in.

The prediction, if the effect is real spin polarisation: the sign is negative
everywhere, the magnitude grows with exact exchange, and the McConnell ratio
-rho(node) / sum(rho(neighbours)) is roughly constant across functionals and
bases even though the individual populations move.
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

from nimrod.config import DATA_DIR
from nimrod.reporters import (
    imino_nitroxide,
    nitronyl_nitroxide,
    phenalenyl,
    phenalenyl_sites,
)

#: Ordered by fraction of exact exchange, which is the axis spin polarisation
#: is expected to move along.  PBE has none, BHandHLYP has half.
#: B3LYP is spelled out as the libxc name because PySCF's bare 'b3lyp' selects
#: VWN5 while Psi4's selects VWN_RPA, and every other number in this project
#: came from the VWN_RPA form.  The rest have no such ambiguity.
FUNCTIONALS = (
    ("pbe", "pbe", 0.00),
    ("HYB_GGA_XC_B3LYP", "b3lyp", 0.20),
    ("pbe0", "pbe0", 0.25),
    ("bhandhlyp", "bhandhlyp", 0.50),
)
BASES = ("def2-svp", "def2-tzvp")


def make_scf(mol, xc: str):
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


def _per_atom(mol, diagonal: np.ndarray) -> np.ndarray:
    per_atom = np.zeros(mol.natm)
    for mu, (atom_index, *_rest) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom_index] += float(diagonal[mu])
    return per_atom


def mulliken_spin(mf, mol) -> np.ndarray:
    dm = mf.make_rdm1()
    spin = np.asarray(dm[0]) - np.asarray(dm[1])
    return _per_atom(mol, (spin @ mol.intor_symmetric("int1e_ovlp")).diagonal())


def lowdin_spin(mf, mol) -> np.ndarray:
    """Symmetrically orthogonalise first, so no overlap density is split by fiat.

    S^(1/2) rho S^(1/2) rather than rho S.  The two agree when the basis is
    small and well balanced and diverge when Mulliken's even split of the
    overlap population stops being defensible, which is the failure mode being
    tested for.
    """
    dm = mf.make_rdm1()
    spin = np.asarray(dm[0]) - np.asarray(dm[1])
    overlap = mol.intor_symmetric("int1e_ovlp")
    root = scipy.linalg.sqrtm(overlap).real
    return _per_atom(mol, (root @ spin @ root).diagonal())


def mcconnell(spin: np.ndarray, centre: int, neighbours: list[int]) -> float | None:
    """-rho(centre) / sum rho(neighbours): the polarisation constant Q.

    Roughly 0.25-0.5 for pi radicals and, crucially, roughly *constant* across
    methods even when the populations themselves move.  A stable Q says the
    negative density is polarisation; an erratic one says it is noise.
    """
    total = float(sum(spin[j] for j in neighbours))
    return None if abs(total) < 1e-6 else float(-spin[centre] / total)


def targets():
    """(name, structure, centre atom, its neighbours, what the centre means)."""
    ply = phenalenyl()
    sites = phenalenyl_sites(ply)
    nodal = sites["nodal"][0]
    yield ("phenalenyl", ply, nodal,
           [j for j in ply.neighbours(nodal) if ply.symbols[j] == "C"],
           "Huckel node, flanked by two SOMO-bearing carbons")

    for name, builder in (("nitronyl-nitroxide", nitronyl_nitroxide),
                          ("imino-nitroxide", imino_nitroxide)):
        struct, index = builder(methylated=False)
        yield (f"{name}-desmethyl", struct, index["C2"],
               [index["N1"], index["N3"]],
               "aryl-bearing carbon; symmetry node on nitronyl nitroxide only")


def main() -> int:
    from pyscf import gto
    import pyscf

    out_path = DATA_DIR / "results" / "spin_density_verify.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", buffering=1)

    def emit(record: dict) -> None:
        handle.write(json.dumps(record) + "\n")

    emit({"event": "start", "pyscf": pyscf.__version__,
          "functionals": [f[1] for f in FUNCTIONALS], "bases": list(BASES)})

    print("Does the negative spin density at the tether carbon survive?\n")
    collected: dict[str, list[dict]] = {}

    for name, struct, centre, neighbours, meaning in targets():
        collected[name] = []
        print(f"{name}  (atom {centre}: {meaning})")
        print(f"  {'functional':<12s} {'%HF':>4s} {'basis':<11s} {'<S^2>':>7s} "
              f"{'Mulliken':>9s} {'Lowdin':>9s} {'Q':>6s}")
        for xc, label, exact_exchange in FUNCTIONALS:
            for basis in BASES:
                record = {"event": "point", "molecule": name, "centre": centre,
                          "neighbours": neighbours, "functional": label,
                          "exact_exchange": exact_exchange, "basis": basis}
                started = time.time()
                try:
                    mol = gto.M(atom=struct.to_psi4(), basis=basis, charge=0,
                                spin=1, verbose=0)
                    mf, energy = make_scf(mol, xc)
                    mul, low = mulliken_spin(mf, mol), lowdin_spin(mf, mol)
                    record.update(
                        energy=energy, converged=bool(mf.converged),
                        s2=float(mf.spin_square()[0]),
                        mulliken_centre=float(mul[centre]),
                        lowdin_centre=float(low[centre]),
                        mulliken_neighbours=[float(mul[j]) for j in neighbours],
                        lowdin_neighbours=[float(low[j]) for j in neighbours],
                        mcconnell_mulliken=mcconnell(mul, centre, neighbours),
                        mcconnell_lowdin=mcconnell(low, centre, neighbours))
                    q = record["mcconnell_mulliken"]
                    print(f"  {label:<12s} {exact_exchange * 100:4.0f} {basis:<11s} "
                          f"{record['s2']:7.4f} {record['mulliken_centre']:+9.4f} "
                          f"{record['lowdin_centre']:+9.4f} "
                          f"{'    --' if q is None else f'{q:6.3f}'}")
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    record["traceback"] = traceback.format_exc()[-600:]
                    print(f"  {label:<12s} {basis:<11s} FAILED "
                          f"{record['error'][:60]}")
                record["seconds"] = time.time() - started
                emit(record)
                collected[name].append(record)
        print()

    # --- did it survive? -------------------------------------------------
    verdict: dict = {"event": "verdict"}
    for name, records in collected.items():
        good = [r for r in records if "error" not in r]
        if not good:
            verdict[name] = {"error": "every level failed"}
            continue
        mulliken = [r["mulliken_centre"] for r in good]
        lowdin = [r["lowdin_centre"] for r in good]
        q_values = [r["mcconnell_mulliken"] for r in good
                    if r["mcconnell_mulliken"] is not None]
        verdict[name] = {
            "levels": len(good),
            "sign_is_negative_everywhere": bool(max(mulliken) < 0 and max(lowdin) < 0),
            "mulliken_range": [min(mulliken), max(mulliken)],
            "lowdin_range": [min(lowdin), max(lowdin)],
            "partitionings_agree_in_sign": bool(
                all(m * l > 0 for m, l in zip(mulliken, lowdin))),
            "mcconnell_mean": float(np.mean(q_values)) if q_values else None,
            "mcconnell_spread": float(np.std(q_values)) if q_values else None,
            "max_s2": max(r["s2"] for r in good),
        }

    # The one comparison the whole design argument turns on.
    node = verdict.get("nitronyl-nitroxide-desmethyl", {})
    control = verdict.get("imino-nitroxide-desmethyl", {})
    if "mulliken_range" in node and "mulliken_range" in control:
        node_worst = max(abs(x) for x in node["mulliken_range"])
        node_best = min(abs(x) for x in node["mulliken_range"])
        control_worst = max(abs(x) for x in control["mulliken_range"])
        verdict["node_carbon_carries_more_spin_than_control"] = bool(
            node_best > control_worst)
        verdict["node_comparison"] = (
            f"nitronyl nitroxide C2 spans {node['mulliken_range'][0]:+.3f}.."
            f"{node['mulliken_range'][1]:+.3f} e, imino control "
            f"{control['mulliken_range'][0]:+.3f}..{control['mulliken_range'][1]:+.3f}; "
            f"the symmetry node {'does not suppress' if node_best > control_worst else 'may still suppress'}"
            f" the density at the tether carbon")

    emit(verdict)
    emit({"event": "done"})
    handle.close()

    print("Verdict")
    for name in collected:
        v = verdict[name]
        if "error" in v:
            print(f"  {name}: {v['error']}")
            continue
        print(f"  {name}")
        print(f"    negative at all {v['levels']} levels: {v['sign_is_negative_everywhere']}"
              f"   Mulliken and Lowdin agree in sign: {v['partitionings_agree_in_sign']}")
        print(f"    Mulliken {v['mulliken_range'][0]:+.3f}..{v['mulliken_range'][1]:+.3f} e"
              f"   Lowdin {v['lowdin_range'][0]:+.3f}..{v['lowdin_range'][1]:+.3f} e")
        if v["mcconnell_mean"] is not None:
            print(f"    McConnell Q = {v['mcconnell_mean']:.3f} "
                  f"+/- {v['mcconnell_spread']:.3f}   max <S^2> {v['max_s2']:.3f}")
    if "node_comparison" in verdict:
        print(f"\n  {verdict['node_comparison']}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
