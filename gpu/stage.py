#!/usr/bin/env python3
"""Flatten the nimrod package into modules a Colab VM can import.

The GPU scripts run on a bare VM with the repository's modules copied next to
them, so they say ``import catalysts`` rather than ``from nimrod import
catalysts``.  The package itself uses relative imports (``from .geometry import
Structure``), which fail outright once the files are flat -- and they fail at
*import* time, after the VM has been allocated and the accelerator is on the
clock.

That rewrite was being done by hand.  Doing it by hand is how a run gets
launched against a stale copy of a module, which is worse than a crash because
it produces numbers.  This does it reproducibly and checks the result imports
before anything is uploaded.

    python gpu/stage.py --out /tmp/nimrod-gpu
    # then upload /tmp/nimrod-gpu/*.py alongside the readout script

Only the modules the GPU scripts actually need are staged; ``psi4_driver`` and
anything else that pulls in Psi4 is left behind, because the VM has no Psi4 and
importing it would be a crash rather than a missing feature.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Modules the GPU readout scripts import, directly or transitively.  Kept
#: explicit rather than globbed: staging the whole package would drag in
#: psi4_driver, and a VM with no Psi4 would fail on import of a module nothing
#: on the VM needs.
STAGED = ("geometry", "catalysts", "reporters")

RELATIVE_IMPORT = re.compile(r"^(\s*)from\s+\.(\w+)\s+import\s", re.MULTILINE)


def flatten(source: str) -> str:
    """Rewrite ``from .module import x`` to ``from module import x``."""
    return RELATIVE_IMPORT.sub(r"\1from \2 import ", source)


def stage(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name in STAGED:
        source = REPO / "nimrod" / f"{name}.py"
        text = flatten(source.read_text())
        if re.search(r"^\s*from\s+\.", text, re.MULTILINE):
            raise SystemExit(
                f"{source}: a relative import survived the rewrite; the VM copy "
                f"would fail on import")
        target = destination / f"{name}.py"
        target.write_text(text)
        written.append(target)
    return written


def verify(destination: Path) -> None:
    """Import every staged module in a fresh interpreter, flat on sys.path.

    Running it in a subprocess with only the staging directory on the path is
    the point: importing them here would find the real package and prove
    nothing.
    """
    probe = "; ".join(f"import {name}" for name in STAGED)
    probe += "; import catalysts as c"
    probe += "; s, i = c.reporter_on_salen('nitronyl-nitroxide', site='C3')"
    probe += "; assert len(s) == i['n_host_atoms'] + 34, len(s)"
    probe += "; print('flat import ok:', s.formula, len(s), 'atoms')"
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=destination,
        capture_output=True, text=True,
        env={"PYTHONPATH": str(destination), "PATH": "/usr/bin:/bin"})
    if result.returncode != 0:
        raise SystemExit(
            f"staged modules do not import flat:\n{result.stderr.strip()[-1500:]}")
    print(f"  {result.stdout.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/nimrod-gpu")
    parser.add_argument("--clean", action="store_true",
                        help="remove the staging directory first")
    args = parser.parse_args()

    destination = Path(args.out)
    if args.clean and destination.exists():
        shutil.rmtree(destination)

    written = stage(destination)
    verify(destination)
    print(f"Staged {len(written)} modules in {destination}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
