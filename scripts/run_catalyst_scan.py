#!/usr/bin/env python
"""Degradation scan of the bare catalyst: does the spin manifold actually flip?

This is the central physical claim of the project, isolated from the colour
centre so it can be tested on its own.  Square-planar d8 Ni(II) is a
closed-shell S = 0 ion; if one imine arm dissociates, the ligand field drops
from four-coordinate to three-coordinate and the S = 1 state should fall below
S = 0.  If that crossing does not happen there is nothing for a spin-based
sensor to detect, and the rest of the project is moot.

    micromamba run -n nimrod python scripts/run_catalyst_scan.py [--quick] [--rigid]

Two ways of walking the coordinate, with an honest trade-off between them.

**Relaxed** (default).  Only the Ni-N distance is frozen; everything else is
optimised, starting each point from the previous converged geometry.  This is
the physically right answer, and it is expensive: on four cores each point costs
minutes, and past about 2.3 A optking begins fighting its own internal
coordinates — the Ni-N stretch stops being recognised as a bond, the coordinate
system goes redundant, and the optimiser stalls rather than failing cleanly.

**Rigid** (``--rigid``).  The imine arm is swung open about the chelate hinge as
a rigid body, from the relaxed intact structure.  Each point is then two
single-point energies instead of an optimisation, so the whole eight-point scan
across four functionals finishes in the time one relaxed point takes.

It has to be a *rotation*, not a translation.  Translating the nitrogen along
the Ni-N axis is the obvious move and it is wrong: the nitrogen sits inside a
six-membered chelate ring, so pushing it radially outward compresses the imine
N=C bond, which falls to 0.99 A between roughly 2.3 and 3.3 A of Ni-N
separation.  No C-N bond is that short.  Since the ligand field at the metal is
precisely what drives the spin flip we are trying to measure, those geometries
would have poisoned the result while looking perfectly reasonable in a plot.
The hinge rotation preserves every internal bond and angle in the moving
fragment exactly, and the rotation sense is chosen by maximising the closest
contact so the arm swings into free space rather than through the trans ligand.

The rigid profile still overestimates the energy of every stretched structure,
because the rest of the complex is denied the chance to reorganise around the
vacated site.  But *both spin states see the identical geometry*, so that error
largely cancels in the gap, which is the quantity of interest.  What the rigid
scan can be trusted for is the sign of the gap, the direction it moves, and the
rough location of the crossing; what it cannot be trusted for is the crossing
distance to better than a few tenths of an angstrom, or any dissociation energy.

Either way, both spin states are evaluated at every geometry across a ladder of
functionals, because a single functional's answer to a 3d spin-state question is
not evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nimrod.config import (
    BASIS_TIERS,
    DATA_DIR,
    FUNCTIONAL_SCREEN,
    HARTREE_TO_KCAL,
)
from nimrod.geometry import arm_atoms, chelate_hinge, ni_salen_model, swing_arm
from nimrod.degradation import relax_at_distance
from nimrod.psi4_driver import JobSpec, run_energy_multiguess, run_optimize
from nimrod.spin import populations_from_properties, spin_properties

GEOMETRY_FUNCTIONAL = "b3lyp"   # best mean absolute error in the validation tier
COARSE = (1.87, 2.30, 2.80, 3.40, 4.20)
FULL = (1.87, 2.10, 2.35, 2.60, 2.90, 3.30, 3.80, 4.50)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="5 points, one functional")
    parser.add_argument("--rigid", action="store_true",
                        help="displace the imine arm without relaxing the rest")
    parser.add_argument("--basis", default=BASIS_TIERS["screen"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    distances = FULL if args.rigid else (COARSE if args.quick else FULL)
    functionals = (GEOMETRY_FUNCTIONAL,) if args.quick else tuple(FUNCTIONAL_SCREEN)
    basis = args.basis

    catalyst, index = ni_salen_model()
    ni, n_labile = index["Ni"], index["N1"]
    carry = arm_atoms(catalyst, ni, n_labile)

    print(f"Catalyst: {catalyst.formula}, {len(catalyst)} atoms")
    print(f"Basis {basis}; geometries at {GEOMETRY_FUNCTIONAL}; "
          f"energetics across {', '.join(functionals)}")
    print(f"Coordinate: Ni-N {distances[0]:.2f} -> {distances[-1]:.2f} A "
          f"({len(distances)} points)\n")

    # ---- Step 1: relax the intact complex -------------------------------
    print("Optimising the intact square-planar complex (S=0) ...", flush=True)
    started = time.time()
    intact_spec = JobSpec(
        geometry=catalyst.to_psi4(),
        method=GEOMETRY_FUNCTIONAL,
        basis=basis,
        charge=0,
        multiplicity=1,
        reference="rks",
        label="catalyst-intact",
    )
    intact_result, intact_geom = run_optimize(intact_spec, max_iter=80)
    if not intact_result.ok or intact_geom is None:
        print(f"  FAILED: {intact_result.error}")
        return 1
    from nimrod.geometry import Structure

    current = Structure.from_psi4(intact_geom)
    current_intact = current  # rigid scan always displaces from here
    hinge = chelate_hinge(current_intact, ni, n_labile)
    print(f"  E = {intact_result.energy:.8f} Eh  ({time.time()-started:.0f} s)")
    print(f"  Ni-N {current.distance(ni, n_labile):.3f} A, "
          f"Ni-O {current.distance(ni, index['O1']):.3f} A\n")

    # ---- Step 2: relaxed scan with sequential continuation ---------------
    records = []
    for distance in distances:
        print(f"Ni-N = {distance:.2f} A", flush=True)
        started = time.time()

        if args.rigid:
            # Swing the imine arm open about the chelate hinge.  A rigid-body
            # rotation preserves every internal bond exactly; translating the
            # nitrogen along the Ni-N axis instead compresses the imine N=C
            # bond to 0.99 A at intermediate separations, which is not
            # chemistry and badly distorts the ligand field we are measuring.
            relaxed = swing_arm(current_intact, ni, n_labile, distance, hinge=hinge)
        else:
            job, relaxed = relax_at_distance(
                current, ni, n_labile, distance,
                functional=GEOMETRY_FUNCTIONAL, basis=basis,
                charge=0, multiplicity=1, carry=carry,
                label=f"catalyst-relax-{distance:.2f}", max_iter=60,
            )
            if not job.ok or relaxed is None:
                print(f"  relaxation FAILED: {str(job.error)[:160]}")
                records.append({"distance": distance, "error": str(job.error)[:400]})
                continue
            current = relaxed  # sequential continuation

        record: dict[str, object] = {
            "distance": distance,
            "geometry": relaxed.to_psi4(),
            "relax_seconds": time.time() - started,
            "rigid": bool(args.rigid),
        }
        if not args.rigid and "constraint_slip" in job.properties:
            record["constraint_slip"] = job.properties["constraint_slip"]

        # ---- Step 3: both spin states, every functional ------------------
        gaps: dict[str, float] = {}
        for functional in functionals:
            energies = {}
            for multiplicity, reference in ((1, "rks"), (3, "uks")):
                spec = JobSpec(
                    geometry=relaxed.to_psi4(),
                    method=functional,
                    basis=basis,
                    charge=0,
                    multiplicity=multiplicity,
                    reference=reference,
                    label=f"catalyst-{functional}-m{multiplicity}-{distance:.2f}",
                )
                result = run_energy_multiguess(spec, property_hook=spin_properties)
                if not result.ok:
                    continue
                energies[multiplicity] = result.energy
                record[f"{functional}_m{multiplicity}"] = result.energy
                if "guess_spread_kcal" in result.properties:
                    record[f"{functional}_m{multiplicity}_guess_spread"] = (
                        result.properties["guess_spread_kcal"])
                if multiplicity == 3:
                    record[f"{functional}_s2_triplet"] = result.properties.get("s2")
                    try:
                        populations = populations_from_properties(result.properties)
                        record[f"{functional}_spin_on_Ni"] = float(
                            populations.mulliken[ni]
                        )
                    except Exception:
                        pass
            if 1 in energies and 3 in energies:
                gap = (energies[3] - energies[1]) * HARTREE_TO_KCAL
                gaps[functional] = gap
                record[f"gap_{functional}"] = gap

        if gaps:
            summary = "  ".join(f"{k}:{v:+7.2f}" for k, v in gaps.items())
            print(f"  S=1 minus S=0 (kcal/mol)   {summary}")
            values = np.array(list(gaps.values()))
            print(f"  mean {values.mean():+.2f}, spread {values.max()-values.min():.2f}"
                  f"   [{time.time()-started:.0f} s]")
        records.append(record)

    # ---- Step 4: persist --------------------------------------------------
    out = Path(args.out) if args.out else DATA_DIR / "results" / "catalyst_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "system": catalyst.formula,
        "basis": basis,
        "geometry_functional": GEOMETRY_FUNCTIONAL,
        "functionals": list(functionals),
        "ni_index": ni,
        "n_index": n_labile,
        "records": records,
    }, indent=2, default=str))
    print(f"\nWrote {out}")

    # ---- Step 5: the headline ---------------------------------------------
    print("\n=== Spin-state crossing ===")
    for functional in functionals:
        series = [
            (r["distance"], r.get(f"gap_{functional}"))
            for r in records if isinstance(r.get(f"gap_{functional}"), (int, float))
        ]
        if len(series) < 2:
            print(f"  {functional:10s} insufficient data")
            continue
        crossing = None
        for (d0, g0), (d1, g1) in zip(series, series[1:]):
            if g0 is not None and g1 is not None and g0 * g1 < 0:
                crossing = d0 + (d1 - d0) * abs(g0) / (abs(g0) + abs(g1))
                break
        first, last = series[0][1], series[-1][1]
        note = (f"crosses at Ni-N = {crossing:.2f} A" if crossing
                else "no sign change over the scanned range")
        print(f"  {functional:10s} {first:+7.2f} -> {last:+7.2f} kcal/mol   {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
