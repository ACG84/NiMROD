# NiMROD GPU offload path

Renting a GPU on Google Colab to run the large open-shell DFT and TDDFT jobs
that a four-core container cannot finish in reasonable time.

This is an operator guide, not a sales pitch. Read the [Reality check](#reality-check)
first, because most of the reasons this path fails are account-level and no
amount of correct code fixes them.

---

## Reality check

**Psi4 does not use GPUs.** There is no build flag, no plugin and no option
that offloads Psi4 to CUDA. So this path is not "the same calculation, faster" —
it is a *different program* (GPU4PySCF, the CUDA port of PySCF) run on
different hardware, and the numbers it returns must be compared with the Psi4
numbers rather than substituted for them. `nimrod/gpu_bridge.py` labels every
returned result with its backend for exactly this reason.

**Accelerator availability is tier-gated.** `colab new --gpu A100` is a request,
not an allocation. Free and low-tier Colab accounts frequently get *no*
accelerator at all, and an A100 in particular is routinely unavailable even on
paid tiers. A `400` on session creation means "no quota/entitlement for this
accelerator on this account" — it is a permanent answer for that accelerator,
not a transient error worth retrying. `launch_colab.sh` recognises it and steps
down the list.

**Idle VMs burn compute units.** A session is a rented machine. Nothing
reclaims it except a 24-hour keep-alive cap, so a session left running costs
you until that cap. `colab run` (which the launcher uses) tears the VM down
even when the script fails; the launcher additionally traps `EXIT`/`INT`/`TERM`
and issues a belt-and-braces `colab stop`. If you ever interrupt a run
uncleanly, check with `colab sessions` and stop anything left behind.

**This path has never been executed end to end.** It was developed in a
container with no GPU, no CUDA, no PySCF and no Google credentials. Everything
that can be checked without those has been checked (see
[What is verified](#what-is-verified)); everything that needs them has not.

---

## One-time authentication

The Colab CLI cannot authenticate non-interactively. You need a browser once,
and the `gcloud` SDK.

```bash
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

All four scopes are mandatory, for four separate reasons:

| Scope | Why it is required |
|---|---|
| `https://www.googleapis.com/auth/userinfo.email` | The session backend at `colab.research.google.com` identifies you by email. Without this scope every session call returns **401**. |
| `https://www.googleapis.com/auth/colaboratory` | The keep-alive RPC against `colab.pa.googleapis.com` needs it. Without it, session creation appears to work and then the keep-alive daemon dies with **403**, taking your VM with it. `colab new` pre-flights this RPC and unassigns the fresh VM rather than leaking a billable assignment. |
| `openid` | Mandated by `gcloud` itself for an ADC login. |
| `https://www.googleapis.com/auth/cloud-platform` | Also mandated by `gcloud`: it rejects any scope list that omits it. |

Verify with either of:

```bash
colab --auth=adc sessions   # read-only, lists server-side assignments
colab whoami                # hidden debug command: email, scopes, audience, expiry
```

### Two auth strategies, and a documentation trap

`--auth` accepts `adc` and `oauth2`, and **it must come before the subcommand**:
`colab --auth=adc new -s foo`, never `colab new --auth=adc -s foo`.

> **Trap.** The bundled skill (`colab skill`) states that the default is `adc`.
> It is not. `colab --help` reports `[default: oauth2]`, and the observed
> behaviour agrees: with no flag, the CLI starts the interactive OAuth flow.
> **Always pass `--auth` explicitly.** `launch_colab.sh` and
> `nimrod/gpu_bridge.py` both default to `adc` and both pass it explicitly;
> override with `NIMROD_COLAB_AUTH=oauth2` if you have a client config.

The two strategies fail differently, which is how the tooling tells "not
authenticated" from "broken":

* **`adc`** raises `DefaultCredentialsError: Your default credentials were not
  found` and exits 1 immediately. Safe for unattended use.
* **`oauth2`** prints a consent URL to stderr, writes `Enter the authorization
  code:` to **stdout**, and then blocks reading stdin. With a TTY attached
  **this hangs forever**, and in an overnight scan that is indistinguishable
  from a slow calculation. Every CLI invocation in this directory therefore
  redirects stdin from `/dev/null` and wraps the call in a timeout.

---

## Layout

| File | Runs where | Purpose |
|---|---|---|
| `gpu4pyscf_job.py` | on the rented VM | Self-contained worker: UKS/RKS + TDDFT + population analysis, emits NiMROD-shaped JSON. Imports nothing from this repo. |
| `launch_colab.sh` | locally | Wraps `colab run`, handles accelerator fallback, timeouts, session teardown, and harvesting the result. |
| `../nimrod/gpu_bridge.py` | locally | Python API: `Structure` in, `JobResult` out. |

---

## What to offload, and what not to

`nimrod.gpu_bridge.worth_offloading()` encodes this table and will tell you the
answer for any given job.

### Worth offloading

* **The 54-atom sensor assembly at def2-TZVP** (1163 basis functions). An
  open-shell UKS SCF on a 3d metal, repeated at every point of the degradation
  scan. This is the job the path exists for.
* **TDDFT on the assembly or the 37-atom colour centre.** Linear response calls
  the same J/K engine as the SCF, but tens of times more often — the single
  most GPU-favourable workload in the project.
* **def2-TZVP single points generally**, once the system is past ~45 atoms.

### Not worth offloading

* **The validation tier** — H2O, O2, CH2, benzene at 6-31G or def2-SVP. These
  finish locally in seconds. Allocating a VM takes longer than the calculation,
  and you would be paying for the privilege.
* **Anything at def2-SVP below ~45 atoms.** Run it locally.

### Impossible to offload

* **CASSCF / RASSCF.** Psi4's `detci` is CPU-only, so there is nothing to gain
  there — and GPU4PySCF does not implement CASSCF at all, so there is nowhere
  to send it. `nimrod/casscf.py` stays local, permanently.
* **Coupled cluster and EOM-CCSD**, for the same reason.
* **SAPT.** Psi4-specific; no counterpart in the PySCF stack.

`worth_offloading()` reports these as `possible=False` with the reason, rather
than letting you discover it after paying for a VM.

---

## Worked example, end to end

Compute the intact sensor's absorption spectrum at TD-PBE0/def2-TZVP on an
A100, falling back to a T4.

**1. Export the geometry.**

```bash
cd /path/to/NiMROD
python - <<'PY'
from nimrod.geometry import sensor_assembly
from nimrod.gpu_bridge import write_xyz
structure, info = sensor_assembly()
write_xyz(structure, "data/gpu/sensor-intact.xyz", comment="intact sensor")
print(structure.formula, len(structure), "atoms; doublet")
print("Ni index", info["Ni"], "labile N", info["N_labile"])
PY
# C28H21N2NiO2 54 atoms; doublet
```

**2. Check the plan without spending anything.** Both layers have a dry run;
neither allocates a VM or imports PySCF.

```bash
gpu/launch_colab.sh \
  --xyz data/gpu/sensor-intact.xyz \
  --out data/gpu/sensor-intact.json \
  --mult 2 --functional pbe0 --basis def2-tzvp --n-states 12 \
  --label sensor-intact --dry-run
```

**3. Run it.**

```bash
gpu/launch_colab.sh \
  --xyz data/gpu/sensor-intact.xyz \
  --out data/gpu/sensor-intact.json \
  --mult 2 --functional pbe0 --basis def2-tzvp --n-states 12 \
  --label sensor-intact \
  --gpu "A100 T4" --timeout 10800 --verbose
```

**4. Or drive the whole thing from Python**, which additionally caches the
result in `data/cache` so a repeated scan point never rents a second VM:

```python
from nimrod.geometry import sensor_assembly
from nimrod.gpu_bridge import run_gpu_energy, colab_status, worth_offloading

structure, info = sensor_assembly()

print(colab_status().explain())
print(worth_offloading(len(structure), "def2-tzvp", n_states=12).reason)

result = run_gpu_energy(
    structure,
    functional="pbe0", basis="def2-tzvp",
    multiplicity=2, n_states=12,
    label="sensor-intact",
    gpus=("A100", "T4"),
    timeout_seconds=10800,
)

print(result.energy, result.preset)          # -> ... 'gpu4pyscf:diis-df'
print(result.properties["s2"])               # <S^2> of the doublet
print(result.properties["mulliken_spin"])    # per-atom spin, input atom order

from nimrod.excited import ExcitedState
for state in (ExcitedState(**s) for s in result.properties["states"]):
    print(state)
```

`result` is an ordinary `nimrod.psi4_driver.JobResult`, so
`nimrod.spin.populations_from_properties(result.properties)` and the rest of
the analysis layer consume it unchanged.

---

## Gotchas the launcher already handles

You do not need to do anything about these; they are documented so that the
code's defensiveness reads as deliberate rather than superstitious.

1. **`colab run --timeout` defaults to 30 seconds.** Not 30 minutes. Every real
   DFT job in this project exceeds it, and the failure looks like a mysterious
   kernel timeout rather than a configuration error. The launcher defaults to
   7200 s and warns below 300 s.
2. **An unrecognised `--gpu` silently becomes A100.** A typo like `A1OO` does
   not error; it gets you an A100 request you did not make, which then usually
   fails for quota. The launcher validates against `T4 L4 G4 H100 A100` locally
   and refuses anything else.
3. **No file comes back from a `colab run`.** The VM is destroyed when the
   script exits, so `--out` on the remote side is unreachable by design. The
   worker therefore also prints its JSON to stdout between
   `===NIMROD-GPU-JSON-BEGIN===` / `===NIMROD-GPU-JSON-END===` sentinels, and
   that is what the launcher harvests. Sentinels rather than "parse all of
   stdout" because an unauthenticated CLI writes its OAuth prompt to stdout too.
4. **`colab run` uploads no data files.** The geometry is gzip+base64'd into a
   single argv element (the 54-atom assembly comes to ~1.2 kB, far inside any
   argv limit).
5. **An omitted `-s` gets a random 6-hex session name**, which makes the
   session impossible to refer to afterwards. The launcher always names
   sessions.
6. **Parallel runs share `~/.config/colab-cli/sessions.json`.** The launcher
   points `--config` at a per-run scratch file so concurrent jobs cannot
   clobber each other's session state.

### Exit codes

The launcher's own failures live in a `9x` block so they can never be confused
with the remote job's, which pass through unchanged.

| Code | Source | Meaning |
|---|---|---|
| 0 | — | success |
| 2 | either | bad arguments |
| 91 | launcher | not authenticated |
| 92 | launcher | no accelerator could be allocated |
| 93 | launcher | remote job returned no parseable JSON |
| 1 | job | unexpected error (traceback on stderr) |
| 3 | job | SCF did not converge |
| 4 | job | TDDFT failed |
| 5 | job | PySCF could not be imported or installed |

`gpu_bridge` maps 91/92/93 onto `ColabAuthError`, `ColabQuotaError` and
`ColabResultError`. Job codes are *not* raised: a diverged SCF comes back as a
`JobResult` with `converged=False` and `error` set, exactly as the Psi4 driver
behaves, so a scan can record the failure and carry on.

---

## Numerical comparability with the Psi4 branch

The worker recomputes `<S^2>` and the Mulliken/Löwdin spin populations from the
density matrices using the *same expressions* as `nimrod/spin.py`, rather than
calling PySCF's own routines, so the two branches cannot disagree through
differing conventions. The property key names are identical — verified by
comparing against a real Psi4 run (16/16 spin keys, 12/12 `ExcitedState`
fields).

Two functional definitions genuinely differ between the codes, and the worker
reports both in `properties.gpu_provenance.warnings`:

* **B3LYP.** Psi4's `b3lyp` uses VWN3(RPA) local correlation; the bare PySCF
  string `b3lyp` has denoted the VWN5 variant in some releases. The worker pins
  the libxc name `HYB_GGA_XC_B3LYP` to remove the ambiguity.
* **wB97X-D.** libxc supplies only the range-separated XC; Psi4 adds the `-D2`
  empirical dispersion and PySCF does not. **Total energies are not
  comparable.** Excitation energies largely are, being differences of states
  that carry the same dispersion term.

Also note the direction of one capability gap: Psi4's TDSCF kernel *cannot* run
meta-GGAs at all (`tpss`, `tpssh`, `m06` — including the project's reference
functional). PySCF's TDDFT can. That makes the GPU path the only way to get
TD-TPSSh numbers here, and it also means those numbers have no Psi4 counterpart
to be checked against. The worker says so in its warnings.

---

## What is verified

Checked in a container with no GPU, no CUDA, no PySCF and no credentials:

* `python -m py_compile` on the worker and the bridge.
* `bash -n` and `shellcheck -S style` (0.11.0) on the launcher — clean, no
  suppressions except one annotated `SC2329` for the `trap`-invoked `cleanup`.
* The worker's `--dry-run` on the real 54-atom assembly, and its argument
  validation (electron-count/multiplicity parity, unknown elements, malformed
  coordinates, `--triplets` against an open-shell reference, missing files).
* That `--dry-run` imports **nothing** outside the standard library — asserted
  by inspecting `sys.modules` afterwards.
* The gzip+base64 geometry round trip.
* That `gpu_bridge.gpu_fingerprint()` and the worker's `fingerprint()` — two
  independent implementations — agree on four different systems. This is what
  makes the pre-launch cache check trustworthy.
* Property-key equality against a real Psi4 UKS run (CH2 triplet/6-31G) and a
  real Psi4 TDDFT run (H2O/6-31G).
* The full launcher control flow against a stub that emulates `colab run`:
  auth pre-flight, accelerator step-down on `400`, sentinel harvesting, JSON
  validation, session teardown, exit-code propagation.
* Graceful degradation against the *real* unauthenticated CLI: `colab_status()`
  reports it, `run_gpu_energy()` raises `ColabAuthError` carrying the full
  remediation, and nothing hangs.

## What is NOT verified

Everything requiring hardware or credentials this container does not have:

* **No GPU4PySCF calculation has ever been run.** Not one SCF, not one TDDFT.
  The GPU code path is written from the documented PySCF/GPU4PySCF APIs and is
  unexecuted.
* **The `pip install gpu4pyscf-cuda12x` step is untested**, as is the choice
  between the cuda12x and cuda11x wheels for whatever CUDA version Colab is
  shipping when you read this.
* **GPU4PySCF's TDDFT coverage is version-dependent.** The worker tries the GPU
  solver, falls back to `mf.to_cpu()` plus CPU PySCF TDDFT if it is missing, and
  records which happened in `properties.tddft_backend`. Which branch you get in
  practice is unknown.
* **No `colab` command has ever succeeded here** — only their failure modes
  were observed. Session allocation, `--gpu` behaviour, quota `400`s and
  keep-alive are all as documented by the bundled skill, not as observed.
* **The functional and basis name mappings are unvalidated against a running
  PySCF.** They are believed correct from the libxc and PySCF naming
  conventions.
* **No cross-code numerical agreement has been demonstrated.** That Psi4 and
  GPU4PySCF give the same energy for the same specification is the *goal* of
  the schema work here, not an established fact. The first real run should be a
  small system computed both ways and compared before any production number is
  taken from the GPU path.
