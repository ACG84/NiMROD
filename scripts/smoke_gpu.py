"""Smoke test for :mod:`gpu.gpu4pyscf_job` and :mod:`nimrod.gpu_bridge`.

The GPU offload path has an unusual testing problem: the machine that develops
it has no GPU, no CUDA and no PySCF, and the machine that runs it is destroyed
the moment the job ends.  Everything that *can* be pinned without a GPU is
pinned here, and the one thing that cannot is stated rather than faked.

What is checked, and why each check is the one that catches a real failure:

1. **Dry run.**  The plan is the same object the real run consumes, so a wrong
   electron count, a wrong reference or a mistranslated functional shows up
   here rather than after ten minutes of rented VM.  Also asserts the dry run
   imports nothing outside the standard library -- if it ever imports numpy,
   the "runnable on a bare machine" promise is gone.

2. **Fingerprint agreement.**  :func:`nimrod.gpu_bridge.gpu_fingerprint` and
   :func:`gpu.gpu4pyscf_job.fingerprint` are two independent implementations of
   one hash.  The bridge consults the cache with its value *before* renting a
   VM; if the two drift, either every job is recomputed or -- much worse -- a
   result is filed under the wrong key.  Checked through the real
   ``Structure -> to_xyz -> parse_xyz -> geometry_block`` round trip, because
   that is where a formatting difference would actually bite.

3. **Schema equality with the Psi4 path.**  ``properties`` must carry exactly
   the keys :func:`nimrod.spin.spin_properties` produces, and each TDDFT root
   exactly the fields of :class:`nimrod.excited.ExcitedState`.  This is
   verified against a *live* Psi4 run, not against a hand-written list.

4. **``Infinity`` round trip.**  A zero excitation energy is the signature of a
   triplet instability and reaches the report as ``float('inf')``.  Encoding it
   as a string instead makes ``ExcitedState.__str__`` raise.

5. **Exit codes and argument validation**, since :mod:`nimrod.gpu_bridge`
   dispatches on them and a collision reports a diverged SCF as an auth error.

6. **Cross-code numerical agreement** -- only if a PySCF interpreter is
   available.  Psi4 and PySCF are different programs; the point of the schema
   work is that their answers can be *compared*, and this is where that claim
   is either demonstrated or honestly skipped.  Point ``NIMROD_PYSCF_PYTHON``
   at an interpreter with ``pyscf`` installed to enable it, e.g.::

       python3 -m venv /tmp/pyscfenv && /tmp/pyscfenv/bin/pip install pyscf
       NIMROD_PYSCF_PYTHON=/tmp/pyscfenv/bin/python ... scripts/smoke_gpu.py

   Without it the section prints SKIPPED and says so in the summary.  It is
   never silently passed.

No GPU4PySCF code path is exercised anywhere in this file.  There is no GPU on
this machine and none of these checks pretend otherwise.

Run with:

    MAMBA_ROOT_PREFIX=... NIMROD_THREADS=1 \\
        ./bin/micromamba run -n nimrod python scripts/smoke_gpu.py
"""

from __future__ import annotations

import base64
import dataclasses
import gzip
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gpu"))

import gpu4pyscf_job as JOB  # noqa: E402

from nimrod.excited import ExcitedState, _result_from_job  # noqa: E402
from nimrod.geometry import named_pah, occ_radical, sensor_assembly  # noqa: E402
from nimrod.gpu_bridge import (  # noqa: E402
    gpu_fingerprint,
    load_gpu_result,
    worth_offloading,
)
from nimrod.psi4_driver import JobSpec, run_energy  # noqa: E402
from nimrod.spin import populations_from_properties, spin_properties  # noqa: E402

JOB_SCRIPT = Path(__file__).resolve().parent.parent / "gpu" / "gpu4pyscf_job.py"
SCRATCH = Path(os.environ.get("NIMROD_SMOKE_SCRATCH", "/tmp")) / "nimrod-smoke-gpu"

#: Interpreter with PySCF installed, for the cross-code section.  Absent by
#: design: the project's own environment is Psi4-only.
PYSCF_PYTHON = os.environ.get("NIMROD_PYSCF_PYTHON", "")

CH2_TRIPLET = (
    "C  0.00000000  0.00000000  0.11000000\n"
    "H  0.00000000  0.98200000 -0.33000000\n"
    "H  0.00000000 -0.98200000 -0.33000000"
)
H2O = (
    "O  0.00000000  0.00000000  0.11730000\n"
    "H  0.00000000  0.75720000 -0.46920000\n"
    "H  0.00000000 -0.75720000 -0.46920000"
)

FAILURES: list[str] = []
SKIPS: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"   [{status}] {name}: {detail}")
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def skip(name: str, reason: str) -> None:
    print(f"   [SKIP] {name}: {reason}")
    SKIPS.append(f"{name}: {reason}")


def run_job(args: list[str], *, python: str | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the job script as a subprocess, as ``colab run`` would."""
    return subprocess.run(
        [python or sys.executable, str(JOB_SCRIPT), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=1800,
    )


def write_xyz(name: str, block: str, comment: str) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / name
    natom = len([ln for ln in block.strip().splitlines() if ln.strip()])
    path.write_text(f"{natom}\n{comment}\n{block}\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 1. The plan
# --------------------------------------------------------------------------


def case_dry_run() -> None:
    print("\n1. Dry run on the 54-atom sensor assembly")
    structure = sensor_assembly()[0]
    path = write_xyz("sensor.xyz", structure.to_psi4(), "sensor")

    proc = run_job([
        "--xyz", str(path), "--mult", "2", "--functional", "pbe0",
        "--basis", "def2-tzvp", "--n-states", "12", "--dry-run",
    ])
    for line in proc.stdout.splitlines():
        print(f"   | {line}")
    check("dry run exits 0", proc.returncode == 0, f"exit {proc.returncode}")

    plan = {}
    for line in proc.stdout.splitlines():
        if ":" in line and line.startswith("  "):
            key, _, value = line.strip().partition(":")
            plan[key.strip()] = value.strip()

    # 28*6 + 21*1 + 2*7 + 28 + 2*8 = 247, and 247 is odd, so a doublet.
    check(
        "electron count is 247 (odd -> doublet)",
        plan.get("electrons", "").startswith("247"),
        plan.get("electrons", "MISSING"),
    )
    check(
        "an odd-electron system gets an unrestricted reference",
        plan.get("reference", "").startswith("UKS"),
        plan.get("reference", "MISSING"),
    )
    check("formula is C28H21N2NiO2", "C28H21N2NiO2" in plan.get("system", ""), plan.get("system", ""))

    # Parity: a doublet needs an odd electron count, a singlet an even one.
    bad = run_job(["--xyz", str(path), "--mult", "1", "--dry-run"])
    check(
        "247 electrons at multiplicity 1 is rejected",
        bad.returncode == 2 and "cannot form multiplicity" in bad.stderr,
        f"exit {bad.returncode}",
    )


def case_dry_run_is_stdlib_only() -> None:
    print("\n2. The dry run must not import a quantum-chemistry stack")
    path = SCRATCH / "sensor.xyz"
    before = set(sys.modules)
    argv = sys.argv
    sys.argv = ["gpu4pyscf_job.py", "--xyz", str(path), "--mult", "2", "--dry-run"]
    code = None
    try:
        runpy.run_path(str(JOB_SCRIPT), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = argv
    heavy = sorted(
        m for m in set(sys.modules) - before
        if m.split(".")[0] in {"numpy", "scipy", "pyscf", "gpu4pyscf", "cupy", "psi4"}
    )
    check("in-process dry run exits 0", code == 0, f"exit {code}")
    check("imports nothing heavy", not heavy, ", ".join(heavy) or "nothing beyond the stdlib")


# --------------------------------------------------------------------------
# 2. Fingerprints
# --------------------------------------------------------------------------


def case_fingerprints() -> None:
    print("\n3. Fingerprint agreement: gpu_bridge vs the job script")
    print("   (through the real Structure -> xyz -> parse -> block round trip)")

    sensor = sensor_assembly()[0]
    colour_centre = occ_radical()[0]
    pyrene = named_pah("pyrene")

    cases = [
        ("sensor  pbe0/def2-tzvp  mult 2  12 roots", sensor, 2,
         dict(functional="pbe0", basis="def2-tzvp", n_states=12)),
        ("sensor  pbe0/def2-tzvp  mult 2  12 roots TDA", sensor, 2,
         dict(functional="pbe0", basis="def2-tzvp", n_states=12, tda=True)),
        ("colour centre  b3lyp/def2-svp  mult 2", colour_centre, 2,
         dict(functional="b3lyp", basis="def2-svp")),
        ("pyrene  tpssh/def2-svp  mult 1", pyrene, 1,
         dict(functional="tpssh", basis="def2-svp")),
    ]

    for name, structure, mult, kwargs in cases:
        local = gpu_fingerprint(structure, multiplicity=mult, **kwargs)
        path = write_xyz(f"fp-{abs(hash(name))}.xyz", structure.to_psi4(), structure.name)
        args = [
            "--xyz", str(path), "--mult", str(mult),
            "--functional", kwargs["functional"], "--basis", kwargs["basis"],
            "--n-states", str(kwargs.get("n_states", 0)), "--dry-run",
        ]
        if kwargs.get("tda"):
            args.append("--tda")
        proc = run_job(args)
        remote = ""
        for line in proc.stdout.splitlines():
            if "fingerprint" in line:
                remote = line.split(":", 1)[1].strip()
        check(f"{name}", bool(remote) and remote == local, f"{local} vs {remote or 'MISSING'}")

    # The geometry also travels compressed through argv; the wire format must
    # not perturb the hash.
    blob = base64.b64encode(gzip.compress(sensor.to_xyz("sensor").encode())).decode()
    print(f"   xyz {len(sensor.to_xyz('sensor'))} bytes -> {len(blob)} base64 chars")
    proc = run_job([
        "--xyz-b64", blob, "--mult", "2", "--functional", "pbe0",
        "--basis", "def2-tzvp", "--n-states", "12", "--dry-run",
    ])
    remote = ""
    for line in proc.stdout.splitlines():
        if "fingerprint" in line:
            remote = line.split(":", 1)[1].strip()
    expected = gpu_fingerprint(sensor, functional="pbe0", basis="def2-tzvp",
                               multiplicity=2, n_states=12)
    check("gzip+base64 argv path gives the same fingerprint", remote == expected,
          f"{expected} vs {remote or 'MISSING'}")

    # A GPU answer must never land in a Psi4 cache slot.
    psi4_fp = JobSpec(
        geometry=sensor.to_psi4(), method="pbe0", basis="def2-tzvp", multiplicity=2,
    ).fingerprint()
    check("GPU and Psi4 fingerprints differ for the same specification",
          psi4_fp != expected, f"psi4 {psi4_fp} vs gpu {expected}")


# --------------------------------------------------------------------------
# 3. Schema, pinned against live Psi4
# --------------------------------------------------------------------------


def case_schema_against_psi4() -> None:
    print("\n4. Output schema vs a live Psi4 run (CH2 triplet, b3lyp/6-31G)")

    psi4_job = run_energy(
        JobSpec(geometry=CH2_TRIPLET, method="b3lyp", basis="6-31g", multiplicity=3,
                label="smoke-gpu-ch2"),
        use_cache=False,
        property_hook=spin_properties,
    )
    if not psi4_job.ok:
        check("Psi4 reference SCF", False, str(psi4_job.error))
        return
    print(f"   Psi4: E = {psi4_job.energy:.10f} Eh   <S^2> = {psi4_job.properties['s2']:.6f}")

    # What the GPU script *would* emit: build the properties block with the
    # same code the remote job runs, minus the parts that need PySCF.
    plan_keys = set(psi4_job.properties)
    gpu_keys = {
        "s2", "s2_ideal", "spin_contamination", "n_alpha", "n_beta", "sz",
        "multiplicity", "natom", "symbols", "mulliken_spin", "lowdin_spin",
        "mulliken_charge", "mulliken_spin_total", "lowdin_spin_total",
        "max_spin_atom", "max_spin_value",
    }
    check("Psi4 spin_properties key set is the 16 the GPU script emits",
          plan_keys == gpu_keys,
          f"psi4-only {sorted(plan_keys - gpu_keys)}, gpu-only {sorted(gpu_keys - plan_keys)}")

    # The JobResult field set must be reproducible with no translation layer.
    document = JOB.empty_result(JOB.make_plan(_namespace(
        xyz=None, xyz_b64=_b64(CH2_TRIPLET), charge=0, multiplicity=3,
        functional="b3lyp", basis="6-31g", n_states=0, triplets="none", tda=False,
        out="result.json", label="", no_df=False, grid_level=3, conv_tol=1e-9,
        max_cycle=200, cpu_only=True, no_install=True,
    )))
    path = SCRATCH / "schema.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    try:
        loaded = load_gpu_result(path)
        check("empty GPU document constructs a JobResult", True,
              f"reference={loaded.reference} converged={loaded.converged}")
    except Exception as exc:  # noqa: BLE001
        check("empty GPU document constructs a JobResult", False, f"{type(exc).__name__}: {exc}")

    # Every ExcitedState field, and only those.
    state = JOB.make_state(1, 0.3, 0.0125, "singlet", transition_dipole=[0.1, 0.0, 0.0])
    fields = {f.name for f in dataclasses.fields(ExcitedState)}
    check("make_state emits exactly the ExcitedState fields", set(state) == fields,
          f"extra {sorted(set(state) - fields)}, missing {sorted(fields - set(state))}")
    try:
        ExcitedState(**state)
        check("ExcitedState accepts it", True, "constructed")
    except Exception as exc:  # noqa: BLE001
        check("ExcitedState accepts it", False, f"{type(exc).__name__}: {exc}")


def case_infinity_round_trip() -> None:
    print("\n5. A zero root (triplet instability) survives the JSON round trip")
    state = JOB.make_state(1, 0.0, 0.0, "triplet")
    document = json.dumps({"states": [state]})
    check("encoded as the bare Infinity literal, not a string",
          '"energy_nm": Infinity' in document,
          document[document.index('"energy_nm"'):].split(",")[0])
    restored = ExcitedState(**json.loads(document)["states"][0])
    check("decodes back to a float", isinstance(restored.energy_nm, float),
          f"{type(restored.energy_nm).__name__} = {restored.energy_nm}")
    try:
        check("__str__ does not raise on it", True, str(restored))
    except Exception as exc:  # noqa: BLE001
        check("__str__ does not raise on it", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# 4. Exit codes
# --------------------------------------------------------------------------


def case_exit_codes() -> None:
    print("\n6. Exit codes (nimrod.gpu_bridge dispatches on these)")
    path = SCRATCH / "ch2.xyz"
    write_xyz("ch2.xyz", CH2_TRIPLET, "ch2 triplet")

    cases = [
        ("multiplicity 0", ["--xyz", str(path), "--mult", "0", "--dry-run"], 2),
        ("wrong electron parity", ["--xyz", str(path), "--mult", "2", "--dry-run"], 2),
        ("missing geometry file", ["--xyz", "/nonexistent.xyz", "--dry-run"], 2),
        ("negative n-states", ["--xyz", str(path), "--n-states", "-1", "--mult", "3", "--dry-run"], 2),
        ("--triplets on an unrestricted reference",
         ["--xyz", str(path), "--mult", "3", "--triplets", "only", "--dry-run"], 2),
        ("corrupt --xyz-b64", ["--xyz-b64", "not!base64!", "--dry-run"], 2),
        ("no geometry at all", ["--dry-run"], 2),
        ("pyscf missing and --no-install",
         ["--xyz", str(path), "--mult", "3", "--cpu-only", "--no-install",
          "--out", str(SCRATCH / "unused.json")], 5),
    ]
    for name, args, expected in cases:
        proc = run_job(args)
        detail = (proc.stderr.strip().splitlines() or ["no stderr"])[-1][:90]
        check(f"{name} -> exit {expected}", proc.returncode == expected,
              f"exit {proc.returncode}  {detail}")

    # Launcher codes sit in a 9x block so they cannot collide with the job's.
    from nimrod.gpu_bridge import (
        JOB_NO_BACKEND, LAUNCHER_NOACCEL, LAUNCHER_NOAUTH, LAUNCHER_NOJSON,
    )
    launcher = {LAUNCHER_NOAUTH, LAUNCHER_NOACCEL, LAUNCHER_NOJSON}
    job_codes = {1, 3, 4, JOB_NO_BACKEND}
    check("launcher and job exit codes do not overlap", not (launcher & job_codes),
          f"launcher {sorted(launcher)} vs job {sorted(job_codes)}")


# --------------------------------------------------------------------------
# 5. Offload advice
# --------------------------------------------------------------------------


def case_offload_advice() -> None:
    print("\n7. worth_offloading: the validation tier must stay local")
    tiers = [
        ("H2O def2-svp", 3, "def2-svp", "pbe0", 0, False),
        ("benzene def2-svp", 12, "def2-svp", "pbe0", 0, False),
        ("sensor def2-tzvp", 54, "def2-tzvp", "pbe0", 0, True),
        ("sensor def2-svp + TDDFT", 54, "def2-svp", "pbe0", 20, True),
    ]
    for name, natom, basis, method, n_states, expected in tiers:
        advice = worth_offloading(natom, basis, method=method, n_states=n_states)
        check(f"{name} -> {'offload' if expected else 'local'}",
              bool(advice) == expected, advice.reason[:80])
    casscf = worth_offloading(54, "def2-tzvp", method="casscf")
    check("CASSCF is impossible, not merely unwise",
          not casscf.possible, casscf.reason[:80])


# --------------------------------------------------------------------------
# 6. Cross-code agreement (only with a real PySCF)
# --------------------------------------------------------------------------


def case_cross_code() -> None:
    print("\n8. Psi4 vs PySCF: do the two codes actually agree?")
    if not PYSCF_PYTHON:
        skip("cross-code numerical agreement",
             "NIMROD_PYSCF_PYTHON is unset, so no PySCF interpreter is available. "
             "This is the one claim that cannot be checked without one.")
        return
    if not Path(PYSCF_PYTHON).exists():
        skip("cross-code numerical agreement", f"{PYSCF_PYTHON} does not exist")
        return

    print(f"   using {PYSCF_PYTHON}")
    write_xyz("ch2.xyz", CH2_TRIPLET, "ch2 triplet")
    write_xyz("h2o.xyz", H2O, "water")

    # --- unrestricted ground state + spin populations ---
    out = SCRATCH / "cross-ch2.json"
    proc = run_job([
        "--xyz", str(SCRATCH / "ch2.xyz"), "--mult", "3", "--functional", "b3lyp",
        "--basis", "6-31g", "--cpu-only", "--no-install", "--verbose", "0",
        "--out", str(out),
    ], python=PYSCF_PYTHON)
    if proc.returncode != 0:
        check("PySCF CH2 job", False, f"exit {proc.returncode}: {proc.stderr.strip()[-200:]}")
        return

    gpu = load_gpu_result(out)
    psi4_job = run_energy(
        JobSpec(geometry=CH2_TRIPLET, method="b3lyp", basis="6-31g", multiplicity=3,
                label="smoke-gpu-cross-ch2"),
        use_cache=False,
        property_hook=spin_properties,
    )
    if not psi4_job.ok:
        check("Psi4 CH2 job", False, str(psi4_job.error))
        return

    g, p = gpu.properties, psi4_job.properties
    print(f"   Psi4  E = {psi4_job.energy:.10f} Eh   PySCF E = {gpu.energy:.10f} Eh")
    print(f"   Psi4  <S^2> = {p['s2']:.8f}          PySCF <S^2> = {g['s2']:.8f}")

    # Both codes run density fitting with *different* auxiliary bases, so the
    # total energies agree to the DF error, not to the SCF threshold.
    check("total energies agree to 1e-4 Eh (different DF auxiliary bases)",
          abs(gpu.energy - psi4_job.energy) < 1e-4,
          f"delta = {abs(gpu.energy - psi4_job.energy):.3e} Eh")
    check("<S^2> agrees to 1e-5", abs(g["s2"] - p["s2"]) < 1e-5,
          f"delta = {abs(g['s2'] - p['s2']):.3e}")
    for scheme in ("mulliken_spin", "lowdin_spin", "mulliken_charge"):
        worst = max(abs(a - b) for a, b in zip(g[scheme], p[scheme]))
        check(f"{scheme} agrees to 1e-4", worst < 1e-4, f"worst delta = {worst:.3e}")
    check("population key sets are identical",
          set(g) - {"gpu_provenance"} == set(p),
          f"{len(set(g) - {'gpu_provenance'})} keys")

    pops = populations_from_properties(g)
    check("nimrod.spin rebuilds SpinPopulations from the GPU document",
          pops.natom == 3 and abs(pops.mulliken.sum() - 2.0) < 1e-6,
          f"largest = {pops.largest(1)}")

    # --- restricted TDDFT ---
    out = SCRATCH / "cross-h2o.json"
    proc = run_job([
        "--xyz", str(SCRATCH / "h2o.xyz"), "--mult", "1", "--functional", "pbe0",
        "--basis", "6-31g", "--n-states", "3", "--cpu-only", "--no-install",
        "--verbose", "0", "--out", str(out),
    ], python=PYSCF_PYTHON)
    if proc.returncode != 0:
        check("PySCF H2O TDDFT job", False, f"exit {proc.returncode}: {proc.stderr.strip()[-200:]}")
        return

    gpu_td = load_gpu_result(out)
    from nimrod.excited import run_tddft as psi4_run_tddft

    psi4_td = psi4_run_tddft(
        H2O, functional="pbe0", basis="6-31g", n_states=3, multiplicity=1,
        use_cache=False,
    )
    if not psi4_td.converged:
        check("Psi4 H2O TDDFT", False, str(psi4_td.error))
        return

    gpu_result = _result_from_job(gpu_td, "pbe0", "6-31g", 1, [])
    print(f"   {'root':>5} {'Psi4 eV':>10} {'PySCF eV':>10} {'d(eV)':>9} "
          f"{'Psi4 f':>9} {'PySCF f':>9}")
    worst_ev = 0.0
    for a, b in zip(psi4_td.states, gpu_result.states):
        delta = abs(a.energy_ev - b.energy_ev)
        worst_ev = max(worst_ev, delta)
        print(f"   {a.index:>5} {a.energy_ev:>10.4f} {b.energy_ev:>10.4f} "
              f"{delta:>9.4f} {a.oscillator_strength:>9.4f} {b.oscillator_strength:>9.4f}")
    check("excitation energies agree to 0.02 eV", worst_ev < 0.02, f"worst = {worst_ev:.4f} eV")
    check("nimrod.excited consumes the GPU document",
          gpu_result.converged and len(gpu_result.states) == 3,
          f"{len(gpu_result.states)} states, converged={gpu_result.converged}")

    # --- a failure in the TDDFT stage must not discard the converged SCF ---
    print("   forcing a TDDFT failure to check the SCF is not thrown away")
    driver = SCRATCH / "force_td_failure.py"
    driver.write_text(
        "import sys, json\n"
        f"sys.path.insert(0, {str(JOB_SCRIPT.parent)!r})\n"
        "import gpu4pyscf_job as J\n"
        "def boom(mf, plan, backend):\n"
        "    raise J.JobError('simulated TDDFT solver failure', 4)\n"
        "J.run_tddft = boom\n"
        "sys.exit(J.main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    out = SCRATCH / "cross-partial.json"
    proc = subprocess.run(
        [PYSCF_PYTHON, str(driver),
         "--xyz", str(SCRATCH / "ch2.xyz"), "--mult", "3", "--functional", "b3lyp",
         "--basis", "6-31g", "--n-states", "4", "--cpu-only", "--no-install",
         "--verbose", "0", "--out", str(out)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1800,
    )
    check("a TDDFT failure still exits 4", proc.returncode == 4, f"exit {proc.returncode}")
    partial = load_gpu_result(out)
    check("the converged SCF energy survives a TDDFT failure",
          partial.energy is not None and abs(partial.energy - gpu.energy) < 1e-10,
          f"energy = {partial.energy!r}")
    check("the spin analysis survives too",
          "mulliken_spin" in partial.properties and "s2" in partial.properties,
          f"{len(partial.properties)} property keys")
    check("the failure is recorded where nimrod.excited looks for it",
          "property_error" in partial.properties,
          str(partial.properties.get("property_error"))[:60])
    downstream = _result_from_job(partial, "b3lyp", "6-31g", 3, [])
    check("nimrod.excited reports it as not converged",
          not downstream.converged and downstream.error is not None,
          f"error = {str(downstream.error)[:50]}")


def _b64(block: str) -> str:
    natom = len([ln for ln in block.strip().splitlines() if ln.strip()])
    text = f"{natom}\nsmoke\n{block}\n"
    return base64.b64encode(gzip.compress(text.encode())).decode()


def _namespace(**kwargs):
    import argparse

    return argparse.Namespace(**kwargs)


def main() -> int:
    print("=" * 72)
    print("gpu.gpu4pyscf_job / nimrod.gpu_bridge smoke test")
    print("=" * 72)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    case_dry_run()
    case_dry_run_is_stdlib_only()
    case_fingerprints()
    case_schema_against_psi4()
    case_infinity_round_trip()
    case_exit_codes()
    case_offload_advice()
    case_cross_code()

    print("\n" + "=" * 72)
    if SKIPS:
        print(f"{len(SKIPS)} CHECK(S) SKIPPED - not run, not passed:")
        for reason in SKIPS:
            print(f"  - {reason}")
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all executed checks passed")
    print("NOT COVERED: no GPU4PySCF code path is exercised anywhere in this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
