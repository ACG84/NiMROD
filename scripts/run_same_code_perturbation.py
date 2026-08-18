#!/usr/bin/env python
"""The observer effect, measured with one code on both sides.

The tether perturbation came out at -0.04 kcal/mol against a 20 kcal/mol signal,
0.2%.  That number had a defect: the bare catalyst was computed in Psi4 and the
tethered construct in gpu4pyscf.  A difference of four hundredths of a kcal/mol
is well inside the spread two independent DFT implementations show on the same
functional and basis -- different grids, different density-fitting auxiliaries,
different libxc call paths -- so the measurement could not distinguish the
perturbation from the codes disagreeing, and the two could partly cancel.

This removes that ambiguity by computing both sides in PySCF 2.14.0, the same
version gpu4pyscf 1.8.1 is built on.  gpu4pyscf is a GPU backend for exactly
this molecular and DFT infrastructure, so a CPU PySCF run is the same code path
with a different linear-algebra device.  Three numbers come out:

    same-code shift     tethered - bare, both in PySCF.  This is the
                        perturbation, with the cross-code term removed.

    tethered agreement  PySCF/CPU against gpu4pyscf/A100 on the identical
                        construction.  Checks the device, not the physics.

    bare agreement      PySCF against Psi4 on the identical geometry.  This is
                        the cross-code term the original number was confounded
                        with, measured directly instead of assumed small.

Everything runs at the relaxed intact geometry, which is where the
closed-shell-metal doublet exists at all -- past ~2.9 A the quench does not hold
because the magnetic metal has become the ground state.

Settings mirror the GPU run exactly: HYB_GGA_XC_B3LYP (Psi4's b3lyp, VWN_RPA
rather than the VWN5 that bare 'b3lyp' selects in PySCF), def2-svp,
density fitting, grids level 3, conv_tol 1e-9, and the same 0.25 level shift on
the quenched doublet.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nimrod.catalysts import sensor_on_salen
from nimrod.config import DATA_DIR, HARTREE_TO_KCAL
from nimrod.geometry import CH_AROMATIC, Structure

XC = "HYB_GGA_XC_B3LYP"
BASIS = "def2-svp"

# What the earlier runs produced, for the agreement checks.
GPU_TETHERED_GAP = 20.024585890762882      # gpu4pyscf 1.8.1, A100
PSI4_BARE_GAP = 20.069072632830803         # Psi4 1.11, CPU


def make_scf(mol, spin_restricted=False, dm0=None, level_shift=0.0):
    from pyscf import dft

    mf = (dft.RKS(mol, xc=XC) if spin_restricted else dft.UKS(mol, xc=XC))
    mf = mf.density_fit()
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
    """Mulliken spin population per atom, from the AO overlap."""
    dm = mf.make_rdm1()
    if np.asarray(dm).ndim == 2:          # restricted: no spin density at all
        return np.zeros(mol.natm)
    spin_ao = (np.asarray(dm[0]) - np.asarray(dm[1])) @ mol.intor_symmetric("int1e_ovlp")
    per_atom = np.zeros(mol.natm)
    diag = spin_ao.diagonal()
    for mu, (atom_index, *_rest) in enumerate(mol.ao_labels(fmt=None)):
        per_atom[atom_index] += float(diag[mu])
    return per_atom


def ao_slice_for_atoms(mol, atom_indices) -> np.ndarray:
    wanted = set(atom_indices)
    return np.array([mu for mu, (a, *_r) in enumerate(mol.ao_labels(fmt=None))
                     if a in wanted], dtype=int)


def quench_fragment_spin(dm, ao_index: np.ndarray):
    """Average alpha and beta density on a fragment -> closed-shell there."""
    dm_a, dm_b = np.asarray(dm[0]).copy(), np.asarray(dm[1]).copy()
    block = np.ix_(ao_index, ao_index)
    average = 0.5 * (dm_a[block] + dm_b[block])
    dm_a[block] = average
    dm_b[block] = average
    return np.array([dm_a, dm_b])


def strip_colour_centre(struct: Structure, info: dict) -> Structure:
    """Delete the colour centre, re-hydrogenate the attachment carbon.

    The catalyst keeps exactly the coordinates it had in the tethered construct,
    so the comparison isolates the electronic perturbation from a structural one.
    """
    host = set(range(info["n_host_atoms"]))
    attach = info["ipso_carbon"]
    direction = struct.coords[info["sp3_carbon"]] - struct.coords[attach]
    direction = direction / np.linalg.norm(direction)
    cap = struct.coords[attach] + CH_AROMATIC * direction

    kept = [i for i in range(len(struct)) if i not in host]
    symbols = [struct.symbols[i] for i in kept] + ["H"]
    coords = np.vstack([struct.coords[kept], cap])
    return Structure(symbols, coords, "ni-salen-stripped")


def load_relaxed_intact() -> Structure:
    """The relaxed tethered geometry the gpu4pyscf number was computed at."""
    source = DATA_DIR / "results" / "closed_shell_metal.jsonl"
    for line in source.open():
        row = json.loads(line)
        if row.get("event") == "optimised" and row.get("xyz"):
            return Structure.from_psi4(row["xyz"])
    raise SystemExit(f"no optimised geometry with coordinates in {source}")


def run_bare(structure: Structure, out) -> dict:
    """Closed-shell singlet (RKS) and triplet (UKS) for the naked catalyst."""
    from pyscf import gto

    text = structure.to_psi4()
    record = {"event": "bare", "natoms": len(structure), "formula": structure.formula}

    t0 = time.time()
    mol_s = gto.M(atom=text, basis=BASIS, charge=0, spin=0, verbose=0)
    mf_s, e_singlet = make_scf(mol_s, spin_restricted=True)
    record.update(e_singlet=e_singlet, singlet_converged=bool(mf_s.converged),
                  singlet_seconds=time.time() - t0)
    out(record)

    t0 = time.time()
    mol_t = gto.M(atom=text, basis=BASIS, charge=0, spin=2, verbose=0)
    mf_t, e_triplet = make_scf(mol_t)
    record.update(e_triplet=e_triplet, triplet_converged=bool(mf_t.converged),
                  s2_triplet=float(mf_t.spin_square()[0]),
                  triplet_spin_Ni=float(atom_spin(mf_t, mol_t)[
                      [i for i, s in enumerate(structure.symbols) if s == "Ni"][0]]),
                  triplet_seconds=time.time() - t0)
    record["bare_gap_kcal"] = (e_triplet - e_singlet) * HARTREE_TO_KCAL
    return record


def run_tethered(structure: Structure, info: dict, out) -> dict:
    """Quartet, then the closed-shell-metal doublet built by quenching it.

    Identical construction to the GPU run: converge the quartet, average the
    metal's alpha and beta density blocks, re-converge at Ms = 1/2 with a level
    shift so the SCF does not slide back into the magnetic solution.
    """
    from pyscf import gto

    ni = info["Ni"]
    text = structure.to_psi4()
    record = {"event": "tethered", "natoms": len(structure),
              "formula": structure.formula}

    t0 = time.time()
    mol_q = gto.M(atom=text, basis=BASIS, charge=0, spin=3, verbose=0)
    mf_q, e_quartet = make_scf(mol_q)
    record.update(e_quartet=e_quartet, quartet_converged=bool(mf_q.converged),
                  s2_quartet=float(mf_q.spin_square()[0]),
                  quartet_spin_Ni=float(atom_spin(mf_q, mol_q)[ni]),
                  quartet_seconds=time.time() - t0)
    out(record)

    t0 = time.time()
    mol_d = gto.M(atom=text, basis=BASIS, charge=0, spin=1, verbose=0)
    dm_cs = quench_fragment_spin(mf_q.make_rdm1(), ao_slice_for_atoms(mol_q, [ni]))
    mf_cs, e_closed = make_scf(mol_d, dm0=dm_cs, level_shift=0.25)
    pops = atom_spin(mf_cs, mol_d)
    metal_spin = float(pops[ni])
    record.update(e_closed_shell=e_closed, cs_converged=bool(mf_cs.converged),
                  s2_closed_shell=float(mf_cs.spin_square()[0]),
                  cs_spin_Ni=metal_spin,
                  cs_spin_colour_centre=float(pops[:info["n_host_atoms"]].sum()),
                  closed_shell_seconds=time.time() - t0)

    # The gap is a spin-state gap only if the metal actually stayed diamagnetic.
    record["closed_shell_held"] = bool(abs(metal_spin) < 0.3)
    if record["closed_shell_held"]:
        record["tethered_gap_kcal"] = (e_quartet - e_closed) * HARTREE_TO_KCAL
    else:
        record["tethered_gap_invalid"] = (
            f"metal re-polarised to {metal_spin:.2f} spin; this is not the "
            f"closed-shell state and the difference is not a spin-state gap")
    return record


def main() -> int:
    out_path = DATA_DIR / "results" / "same_code_perturbation.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", buffering=1)

    def emit(record: dict) -> None:
        handle.write(json.dumps(record) + "\n")

    import pyscf

    _, info = sensor_on_salen(site="C3", host_name="pyrene")
    tethered = load_relaxed_intact()
    bare = strip_colour_centre(tethered, info)

    emit({"event": "start", "pyscf": pyscf.__version__, "xc": XC, "basis": BASIS,
          "geometry": "relaxed intact tethered construct, coordinates matched",
          "tethered_natoms": len(tethered), "bare_natoms": len(bare),
          "reference_gpu_tethered_kcal": GPU_TETHERED_GAP,
          "reference_psi4_bare_kcal": PSI4_BARE_GAP})

    summary: dict = {"event": "summary"}
    # Bare first: it is the smaller system, so a failure surfaces in minutes
    # rather than after the 60-atom pair.
    for name, runner in (("bare", lambda: run_bare(bare, emit)),
                         ("tethered", lambda: run_tethered(tethered, info, emit))):
        try:
            record = runner()
        except Exception as exc:
            record = {"event": name, "error": f"{type(exc).__name__}: {exc}"[:400],
                      "traceback": traceback.format_exc()[-900:]}
        emit(record)
        summary[name] = record

    bare_gap = summary.get("bare", {}).get("bare_gap_kcal")
    teth_gap = summary.get("tethered", {}).get("tethered_gap_kcal")

    if bare_gap is not None:
        summary["bare_vs_psi4_kcal"] = bare_gap - PSI4_BARE_GAP
    if teth_gap is not None:
        summary["tethered_vs_gpu4pyscf_kcal"] = teth_gap - GPU_TETHERED_GAP
    if bare_gap is not None and teth_gap is not None:
        shift = teth_gap - bare_gap
        summary["same_code_shift_kcal"] = shift
        summary["same_code_fraction_of_gap"] = abs(shift) / abs(bare_gap)
        summary["cross_code_shift_kcal"] = GPU_TETHERED_GAP - PSI4_BARE_GAP

    emit(summary)
    emit({"event": "done"})
    handle.close()

    print(f"PySCF {pyscf.__version__}  {XC}/{BASIS}  matched intact geometry\n")
    if bare_gap is not None:
        print(f"  bare      {bare_gap:+8.3f} kcal/mol   "
              f"(Psi4 {PSI4_BARE_GAP:+.3f}, differs by "
              f"{bare_gap - PSI4_BARE_GAP:+.3f})")
    if teth_gap is not None:
        print(f"  tethered  {teth_gap:+8.3f} kcal/mol   "
              f"(gpu4pyscf {GPU_TETHERED_GAP:+.3f}, differs by "
              f"{teth_gap - GPU_TETHERED_GAP:+.3f})")
    if bare_gap is not None and teth_gap is not None:
        shift = teth_gap - bare_gap
        print(f"\n  same-code shift  {shift:+.3f} kcal/mol = "
              f"{abs(shift) / abs(bare_gap):.2%} of the signal")
        print(f"  cross-code shift {GPU_TETHERED_GAP - PSI4_BARE_GAP:+.3f} kcal/mol "
              f"(the number this replaces)")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
