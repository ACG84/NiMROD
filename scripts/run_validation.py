#!/usr/bin/env python
"""Run the spin-state validation tier and write data/results/validation.json.

    micromamba run -n nimrod python scripts/run_validation.py [--quick]

``--quick`` restricts the run to the four screening functionals and a smaller
basis, which is enough to check the machinery but not to quote numbers from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nimrod.config import BASIS_TIERS, FUNCTIONAL_SCREEN
from nimrod.validation import (
    BENCHMARKS,
    default_report_path,
    run_spin_gap_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="screening functionals and a smaller basis")
    parser.add_argument("--no-optimise", action="store_true",
                        help="vertical gaps at the reference geometry instead of adiabatic")
    parser.add_argument("--basis", default=None)
    args = parser.parse_args()

    basis = args.basis or (BASIS_TIERS["screen"] if args.quick else BASIS_TIERS["production"])
    functionals = list(FUNCTIONAL_SCREEN) if args.quick else None

    print(f"Validation tier: basis {basis}, "
          f"{'screening' if args.quick else 'full'} functional ladder, "
          f"{'vertical' if args.no_optimise else 'adiabatic'} gaps\n")
    for benchmark in BENCHMARKS:
        line = (f"  {benchmark.name}: {benchmark.transition}, "
                f"experiment {benchmark.experiment_kcal:.2f} kcal/mol")
        if benchmark.degenerate_pair:
            line += (f"; single-determinant reference "
                     f"{benchmark.sd_reference_kcal:.2f} kcal/mol")
        print(line)
        if benchmark.note:
            print(f"      {benchmark.note}")
    print()

    report = run_spin_gap_validation(
        functionals=functionals,
        basis=basis,
        optimise=not args.no_optimise,
    )

    path = report.save(default_report_path())
    print(f"\nWrote {path}")

    print("\n=== Spread across functionals (the honest error bar) ===")
    for benchmark in BENCHMARKS:
        spread = report.spread(benchmark.name)
        if spread is None:
            print(f"  {benchmark.name:16s} no converged results")
            continue
        print(f"  {benchmark.name:16s} ref {benchmark.sd_reference_kcal:6.2f} | "
              f"computed {spread['min']:6.2f} to {spread['max']:6.2f} "
              f"(mean {spread['mean']:6.2f}, range {spread['range']:5.2f}, n={spread['n']})")

    print("\n=== Mean absolute error per functional ===")
    mae = report.mean_absolute_error()
    for name, value in sorted(mae.items(), key=lambda kv: kv[1]):
        print(f"  {name:12s} {value:6.2f} kcal/mol")

    print("\n=== Spin purity of the high-spin reference (<S^2>, ideal 2.000) ===")
    contaminated = []
    for r in report.results:
        if r.s2_high is None:
            continue
        if abs(r.s2_high - 2.0) > 0.05:
            contaminated.append(r)
    if contaminated:
        for r in contaminated:
            print(f"  {r.system:16s} {r.functional:10s} <S^2> = {r.s2_high:.4f}")
    else:
        print("  all triplet references within 0.05 of 2.000")

    wrong = [r for r in report.results if r.ordering_correct is False]
    if wrong:
        print("\n!! Functionals that got the ground-state ordering WRONG:")
        for r in wrong:
            print(f"   {r.system} / {r.functional}: gap {r.gap_kcal:+.2f} kcal/mol")

    failures = [r for r in report.results if r.failure]
    if failures:
        print(f"\n!! {len(failures)} calculation(s) failed:")
        for r in failures[:10]:
            print(f"   {r.system} / {r.functional}: {str(r.failure)[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
