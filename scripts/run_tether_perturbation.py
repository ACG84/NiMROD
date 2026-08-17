#!/usr/bin/env python
"""How much does the sensor perturb the catalyst it is watching?

The observer-effect objection is that bolting a pyrene onto a salicylidene ring
changes the chemistry you are trying to observe.  That is not a yes/no question,
and it is not settled by calling the perturbation "minor".  The useful criterion
is a ratio: the tether must shift the catalyst's energetics by much less than
the difference the sensor is meant to resolve.

So this measures the metal's S=0 -> S=1 gap two ways at *identical* catalyst
geometries:

    tethered   E(quartet) - E(closed-shell-metal doublet).  The defect's own
               S=1/2 is a spectator in both terms, so the difference is the
               metal's spin-state gap plus an exchange term of order J -- tens
               of cm^-1, negligible at this scale.

               NOT YET AVAILABLE.  The GPU run's "naive" doublet was assumed to
               be the closed-shell-metal state and is not: it carries 0.92-1.18
               spin on the metal at every point, i.e. the default guess found
               the magnetic solution.  E(quartet) - E(naive) is then a
               quartet/broken-symmetry exchange splitting of a few tenths of a
               kcal/mol, and differencing it against a real ~20 kcal/mol gap
               manufactured an apparent 20 kcal/mol perturbation that was purely
               the mismatch.  The guard below now refuses that comparison.
               Getting the tethered side needs a run that constructs the
               closed-shell-metal doublet explicitly -- the mirror of the
               broken-symmetry construction, constraining the metal closed-shell
               rather than flipping the defect.

    bare       E(triplet) - E(singlet) for Ni(salen) alone, with the colour
               centre deleted and the attachment carbon re-hydrogenated.

Using matched geometries isolates the *electronic* perturbation from any
structural one.  A separate relaxed-vs-relaxed comparison would capture both,
and is the obvious follow-up.

The bare calculation runs on Psi4 while the tethered numbers came from
gpu4pyscf, so agreement on the intact-complex gap also cross-checks two
independent codes on the same functional and basis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nimrod.catalysts import open_salen_arm, sensor_on_salen
from nimrod.config import DATA_DIR, HARTREE_TO_KCAL
from nimrod.geometry import CH_AROMATIC, Structure
from nimrod.psi4_driver import JobSpec, run_energy_multiguess

FUNCTIONAL = "b3lyp"
BASIS = "def2-svp"


def strip_colour_centre(struct: Structure, info: dict) -> Structure:
    """Delete the colour centre and re-hydrogenate the attachment carbon.

    Leaves the catalyst at exactly the geometry it had in the tethered
    construct, so the comparison isolates the electronic perturbation.
    """
    host = list(range(info["n_host_atoms"]))
    attach = info["meso_carbon"] if "meso_carbon" in info else info["ipso_carbon"]
    sp3 = info["sp3_carbon"]

    direction = struct.coords[sp3] - struct.coords[attach]
    direction = direction / np.linalg.norm(direction)
    cap = struct.coords[attach] + CH_AROMATIC * direction

    kept = [i for i in range(len(struct)) if i not in host]
    symbols = [struct.symbols[i] for i in kept] + ["H"]
    coords = np.vstack([struct.coords[kept], cap])
    return Structure(symbols, coords, "ni-salen-stripped")


def main() -> int:
    sensor, info = sensor_on_salen(site="C3", host_name="pyrene")
    ni, n_labile = info["Ni"], info["N_labile"]

    source = DATA_DIR / "results" / "pyrene_salen_C3.jsonl"
    rows = [json.loads(line) for line in source.open()]
    optimised = next(r for r in rows if r.get("event") == "optimised")
    relaxed = Structure.from_psi4(optimised["xyz"])
    tethered = {r["distance"]: r for r in rows if r.get("event") == "point"
                and not r.get("error")}

    print(f"Tether perturbation on the metal's spin-state gap\n"
          f"  {FUNCTIONAL}/{BASIS}, matched geometries, colour centre deleted\n")

    results = []
    for distance in sorted(tethered):
        opened = open_salen_arm(relaxed, ni, n_labile, distance)
        bare = strip_colour_centre(opened, info)

        energies = {}
        for multiplicity, reference in ((1, "rks"), (3, "uks")):
            spec = JobSpec(
                geometry=bare.to_psi4(), method=FUNCTIONAL, basis=BASIS,
                charge=0, multiplicity=multiplicity, reference=reference,
                label=f"bare-salen-m{multiplicity}-{distance:.2f}",
            )
            result = run_energy_multiguess(spec)
            if result.ok:
                energies[multiplicity] = result.energy
            else:
                print(f"  {distance:.2f} A  m={multiplicity} FAILED: "
                      f"{str(result.error)[:110]}")

        record = {"distance": distance, "formula": bare.formula,
                  "natoms": len(bare)}
        if 1 in energies and 3 in energies:
            record["bare_gap_kcal"] = (energies[3] - energies[1]) * HARTREE_TO_KCAL

        point = tethered[distance]
        # E(quartet) - E(naive) is only the metal's spin-state gap if the naive
        # solution really has a CLOSED-SHELL metal.  It usually does not: on this
        # construct the default guess converged to a magnetic metal at every
        # point (Ni spin 0.92-1.18), which makes the difference a quartet/
        # broken-symmetry exchange splitting of a few tenths of a kcal/mol.
        # Comparing that against a real ~20 kcal/mol spin-state gap compares two
        # different quantities and manufactures a spurious 20 kcal/mol
        # "perturbation".  Refuse rather than let the assumption ride.
        naive_metal_spin = abs(point.get("naive_spin_Ni", float("nan")))
        if point.get("e_hs") is not None and point.get("e_naive") is not None:
            if naive_metal_spin < 0.3:
                record["tethered_gap_kcal"] = (
                    (point["e_hs"] - point["e_naive"]) * HARTREE_TO_KCAL)
            else:
                record["tethered_gap_invalid"] = (
                    f"naive reference carries {naive_metal_spin:.2f} spin on the "
                    f"metal, so it is not the closed-shell state; a deliberately "
                    f"constructed closed-shell-metal doublet is required")

        if "bare_gap_kcal" in record and "tethered_gap_kcal" in record:
            record["shift_kcal"] = record["tethered_gap_kcal"] - record["bare_gap_kcal"]
        results.append(record)

        bare_s = record.get("bare_gap_kcal")
        teth_s = record.get("tethered_gap_kcal")
        shift = record.get("shift_kcal")
        print(f"  Ni-N {distance:5.2f} A   bare "
              f"{'--' if bare_s is None else f'{bare_s:+8.2f}'}   tethered "
              f"{'--' if teth_s is None else f'{teth_s:+8.2f}'}   shift "
              f"{'--' if shift is None else f'{shift:+7.2f}'} kcal/mol")

    out = DATA_DIR / "results" / "tether_perturbation.json"
    out.write_text(json.dumps({
        "functional": FUNCTIONAL, "basis": BASIS,
        "note": "bare gap from Psi4; tethered gap from gpu4pyscf, same "
                "functional and basis, matched geometries",
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")

    invalid = [r for r in results if "tethered_gap_invalid" in r]
    if invalid:
        print("\n  !! No valid comparison at "
              f"{len(invalid)}/{len(results)} points:")
        print(f"     {invalid[0]['tethered_gap_invalid']}")
        print("     The bare gaps above stand; the tethered side needs a rerun "
              "that\n     constructs the closed-shell-metal doublet explicitly, "
              "the mirror of\n     the broken-symmetry construction.")

    shifts = [abs(r["shift_kcal"]) for r in results if "shift_kcal" in r]
    if shifts:
        signal = 19.4   # intact-complex S=0/S=1 gap the sensor is built to detect
        print(f"\n  largest |shift| {max(shifts):.2f} kcal/mol against a "
              f"{signal:.1f} kcal/mol signal  ->  {max(shifts) / signal:.1%}")
        print("  The observer effect is acceptable when this ratio is small "
              "compared with\n  the differences being discriminated, not when "
              "the shift is small in absolute terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
