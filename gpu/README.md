# GPU offload

Psi4 is **CPU-only**. No amount of GPU will speed it up. So the GPU path here is
not "run NiMROD on a GPU" — it is a second engine, **GPU4PySCF**, used for the
one part of the project that a 4-core box cannot reach: large TD-DFT on the
54-atom assembled sensor.

Jobs are dispatched to a Colab GPU runtime with the
[Google Colab CLI](https://github.com/googlecolab/google-colab-cli).

> **Not exercised by the automated runs.** The CLI requires the operator's own
> Google authentication, which cannot be done headlessly. Everything below has
> been verified as far as it can be without credentials: argument handling,
> the `--dry-run` plan, shell syntax, and the JSON contract. The actual remote
> execution has **not** been run.

---

## What is worth offloading

| Work | Where | Why |
|---|---|---|
| 54-atom assembly, UKS + TD-DFT | **GPU** | ~550 basis functions; hours per point on 4 cores |
| def2-TZVP single points on the assembly | **GPU** | basis-set growth hits the CPU hardest |
| 19-atom catalyst scan | CPU | already tractable, and it is the geometry-optimisation loop that dominates |
| Validation tier (CH₂, O₂, NH) | CPU | seconds per job |
| CASSCF / `detci` | **CPU only** | no GPU implementation exists, in Psi4 or PySCF |
| SAPT0 | CPU | Psi4-only in this project |

---

## One-time authentication

The CLI defaults to Application Default Credentials, and the Colab backends need
a specific set of scopes. Mint ADC with **all four**:

```bash
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

Why each one is needed:

- `userinfo.email` — the session backend (`colab.research.google.com`) returns
  **401** without it.
- `colaboratory` — the keep-alive RPC (`colab.pa.googleapis.com`) returns
  **403** without it, and `colab new` will release the VM it just allocated
  rather than leak a billable assignment.
- `openid` + `cloud-platform` — mandated by `gcloud` itself; it rejects a scope
  list missing `cloud-platform`.

Verify in one shot:

```bash
colab sessions        # read-only; lists server-side assignments
colab whoami          # hidden debug command: active email, scopes, expiry
```

A 403 against `colab.pa.googleapis.com` is almost always a missing scope, not a
broken install. Note that `colab auth` is a *different thing* — it injects
GCP credentials into the running VM's kernel and does nothing for CLI
authentication.

---

## Running a job

Ephemeral one-shot — provision, run, tear down:

```bash
./gpu/launch_colab.sh --xyz sensor.xyz --multiplicity 2 --n-states 10
```

Or drive the job script directly:

```bash
colab run --gpu A100 gpu/gpu4pyscf_job.py -- \
  --xyz sensor.xyz --charge 0 --multiplicity 2 \
  --functional b3lyp --basis def2-svp --n-states 10 --out result.json
```

Check what a run *would* do without touching the network or importing PySCF:

```bash
python gpu/gpu4pyscf_job.py --xyz sensor.xyz --multiplicity 2 \
  --functional b3lyp --basis def2-svp --n-states 5 --out /tmp/o.json --dry-run
```

Generate the geometry from the model with:

```bash
nimrod xyz sensor -o sensor.xyz
```

---

## Gotchas that will actually bite you

- **Accelerators are tier-gated.** Most accounts only ever get CPU. A `400` on
  `colab new --gpu ...` means no quota for that accelerator — fall back to
  `--gpu T4`, or omit the flag.
- **An unrecognised `--gpu` value silently falls back to A100**, which then
  usually fails at the next step. Only `T4`, `L4`, `G4`, `H100`, `A100` are real.
- **Idle VMs burn compute units.** Nothing reclaims them except a 24 h keep-alive
  cap. Always `colab stop -s <name>`; `colab run` without `--keep` self-cleans
  even when the script raises.
- **Always name sessions** with `-s <name>`. An omitted name becomes a random
  hex string and later commands get ambiguous.
- **Never run `colab repl`, `console`, `auth` or `drivemount` from a script** —
  they expect a TTY and will hang.
- **Kernel state persists** between `colab exec` calls in one session. That is
  useful for incremental work and a hazard for reproducibility; the one-shot
  `colab run` path avoids it entirely.

---

## The functional-consistency trap

PySCF's bare `'b3lyp'` string has denoted the **VWN5** correlation variant in
some releases, while Psi4's `b3lyp` uses **VWN_RPA**. Left alone, the CPU and
GPU paths would disagree by roughly 1 mEh/atom for no visible reason — small
enough to look like noise, large enough to corrupt a spin-state gap.

`gpu4pyscf_job.py` therefore pins the explicit libxc string
`HYB_GGA_XC_B3LYP` and prints the mapping it used. If you add functionals,
preserve that discipline: **name the libxc functional explicitly**, and check
the printed mapping before comparing a GPU number with a Psi4 number.

---

## Result contract

The job writes JSON using the same key names the Psi4 modules produce, so
results are directly comparable:

```json
{
  "backend": "gpu4pyscf" | "pyscf-cpu",
  "energy": -1234.567890,
  "reference": "UKS",
  "s_squared": 0.7531,
  "states": [
    {"index": 1, "energy_ev": 2.11, "energy_nm": 587.6, "oscillator_strength": 0.031}
  ],
  "mulliken_spin": [0.02, 0.91, ...],
  "fingerprint": "d5fd135a22a01ee3cab861db"
}
```

`nimrod/gpu_bridge.py` parses this back into the shapes the rest of the package
uses, and reports a clear message rather than a stack trace when the CLI is
unauthenticated.

The script falls back to **CPU PySCF** when no CUDA device is present and says
which backend it used — so a silently-slow CPU run cannot be mistaken for a GPU
one.
