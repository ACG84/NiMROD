"""Command-line entry point for NiMROD.

    nimrod info                 what the model system actually is
    nimrod validate             spin-state validation against experiment
    nimrod scan                 catalyst degradation scan
    nimrod occ                  colour-centre photophysics
    nimrod figures              regenerate figures from stored results
    nimrod xyz <what>           dump a structure for viewing

Every subcommand is a thin wrapper: the science lives in the modules, and the
long-running ones write their results into ``data/results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import BASIS_TIERS, DATA_DIR, FIGURE_DIR, FUNCTIONAL_LADDER


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    from .geometry import ni_salen_model, occ_radical, pyrene, sensor_assembly

    host = pyrene()
    occ, occ_info = occ_radical()
    catalyst, cat_index = ni_salen_model()
    sensor, sensor_info = sensor_assembly()

    print("NiMROD model system\n")
    print(f"  host PAH        {host.formula:14s} {len(host):3d} atoms  "
          f"closed shell")
    print(f"  colour centre   {occ.formula:14s} {len(occ):3d} atoms  "
          f"multiplicity {occ.multiplicity_for(0)} (sp3 aryl defect, unpaired spin)")
    print(f"  catalyst        {catalyst.formula:14s} {len(catalyst):3d} atoms  "
          f"multiplicity {catalyst.multiplicity_for(0)} (square-planar d8 Ni(II))")
    print(f"  assembly        {sensor.formula:14s} {len(sensor):3d} atoms  "
          f"multiplicity {sensor.multiplicity_for(0)}")

    print("\n  catalyst coordination:")
    for label in ("O1", "N1", "O2", "N2"):
        print(f"    Ni-{label}  {catalyst.distance(cat_index['Ni'], cat_index[label]):.3f} A")

    print("\n  degradation coordinate: elongate "
          f"Ni-N (atom {sensor_info['N_labile']}) on the arm opposite the tether")
    print(f"  reporting defect: sp3 carbon at atom {sensor_info['sp3_carbon']}, "
          f"{sensor.distance(sensor_info['Ni'], sensor_info['sp3_carbon']):.2f} A from Ni")

    print(f"\n  functional ladder ({len(FUNCTIONAL_LADDER)}): "
          + ", ".join(f"{f.name}({f.hf_exchange:.0%})" for f in FUNCTIONAL_LADDER))
    print(f"  basis tiers: " + ", ".join(f"{k}={v}" for k, v in BASIS_TIERS.items()))
    return 0


# --------------------------------------------------------------------------
# xyz
# --------------------------------------------------------------------------


def cmd_xyz(args: argparse.Namespace) -> int:
    from .geometry import named_pah, ni_salen_model, occ_radical, sensor_assembly

    builders = {
        "pyrene": lambda: named_pah("pyrene"),
        "occ": lambda: occ_radical()[0],
        "catalyst": lambda: ni_salen_model()[0],
        "sensor": lambda: sensor_assembly()[0],
    }
    if args.what not in builders:
        print(f"unknown structure {args.what!r}; choose from {sorted(builders)}",
              file=sys.stderr)
        return 2
    struct = builders[args.what]()
    text = struct.to_xyz(args.what)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(struct)} atoms)")
    else:
        sys.stdout.write(text)
    return 0


# --------------------------------------------------------------------------
# validate / scan / occ
# --------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    from .validation import default_report_path, run_spin_gap_validation
    from .config import FUNCTIONAL_SCREEN

    report = run_spin_gap_validation(
        functionals=list(FUNCTIONAL_SCREEN) if args.quick else None,
        basis=args.basis or (BASIS_TIERS["screen"] if args.quick
                             else BASIS_TIERS["production"]),
    )
    path = report.save(default_report_path())
    print(f"\nwrote {path}")
    for name, mae in sorted(report.mean_absolute_error().items(), key=lambda kv: kv[1]):
        print(f"  {name:12s} MAE {mae:6.2f} kcal/mol")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    import runpy

    script = Path(__file__).resolve().parent.parent / "scripts" / "run_catalyst_scan.py"
    sys.argv = ["run_catalyst_scan.py"] + (["--quick"] if args.quick else [])
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_occ(args: argparse.Namespace) -> int:
    """Colour-centre photophysics: pristine host versus sp3 defect."""
    from .excited import colour_centre_shift, tddft_states
    from .geometry import occ_radical, pyrene

    basis = args.basis or BASIS_TIERS["screen"]
    host = pyrene()
    occ, _ = occ_radical()

    print(f"Pristine {host.formula} ({len(host)} atoms), TD-DFT/{basis} ...", flush=True)
    pristine = tddft_states(host.to_psi4(), basis=basis, multiplicity=1,
                            n_states=args.states)
    for state in pristine[:5]:
        print(f"  S{state.index}  {state.energy_ev:6.3f} eV  "
              f"{state.energy_nm:7.1f} nm  f={state.oscillator_strength:.4f}")

    print(f"\nDefect {occ.formula} ({len(occ)} atoms), doublet, TD-DFT/{basis} ...",
          flush=True)
    defect = tddft_states(occ.to_psi4(), basis=basis, multiplicity=2,
                          n_states=args.states)
    for state in defect[:5]:
        print(f"  D{state.index}  {state.energy_ev:6.3f} eV  "
              f"{state.energy_nm:7.1f} nm  f={state.oscillator_strength:.4f}")

    shift = colour_centre_shift(pristine, defect)
    print("\ncolour-centre shift:")
    for key, value in shift.items():
        print(f"  {key}: {value}")

    out = DATA_DIR / "results" / "occ_photophysics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "basis": basis,
        "pristine": [s.__dict__ for s in pristine],
        "defect": [s.__dict__ for s in defect],
        "shift": shift,
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def cmd_figures(args: argparse.Namespace) -> int:
    from . import figures as figure_builder

    made = figure_builder.build_all()
    if not made:
        print("no stored results found; run 'nimrod validate' and 'nimrod scan' first")
        return 1
    for path in made:
        print(f"  {path}")
    print(f"\n{len(made)} figures in {FIGURE_DIR}")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nimrod", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="describe the model system")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("xyz", help="dump a structure as XYZ")
    p.add_argument("what", choices=["pyrene", "occ", "catalyst", "sensor"])
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_xyz)

    p = sub.add_parser("validate", help="spin-state validation against experiment")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--basis")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("scan", help="catalyst degradation scan")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("occ", help="colour-centre photophysics")
    p.add_argument("--basis")
    p.add_argument("--states", type=int, default=8)
    p.set_defaults(func=cmd_occ)

    p = sub.add_parser("figures", help="regenerate figures from stored results")
    p.set_defaults(func=cmd_figures)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
