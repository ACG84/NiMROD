# NiMROD — theory and design

**Ni**ckel **M**etal-organic **R**eporter **O**f **D**egradation: an ab initio
study of whether a spin-manifold organic colour centre can detect the decay of a
metal-organic catalyst.

This document sets out the physical design, the reasons for each modelling
choice, and — importantly — the places where the method is weak. Computed
results live in `data/results/` and are summarised in the top-level `README.md`.

---

## 1. The sensing idea

A catalyst that is degrading does not announce it. The usual evidence is
indirect: turnover drops, selectivity drifts, colour changes. What we want is a
spectroscopic handle that responds to the *first* structural step of decay,
before bulk activity is visibly lost.

The proposal here has three parts.

**The catalyst.** A square-planar Ni(II) N₂O₂ complex — a truncated salen
analogue. Ni(II) in a four-coordinate square-planar field is d⁸ with a large
gap between the filled dxz/dyz/dz²/dxy manifold and the empty dx²−y², giving a
**closed-shell S = 0 ground state**. This is the intact, working catalyst.

**The degradation.** The canonical first step of salen-type decay is
decoordination of one imine arm, opening a coordination site. As the Ni–N bond
lengthens, the ligand field weakens and the d-orbital splitting collapses. At
some point the **S = 1 state falls below S = 0** and the metal becomes
paramagnetic. That spin-state crossing is a sharp, binary-ish event driven by a
continuous structural change — exactly the nonlinearity a sensor wants.

**The sensor.** An organic colour centre reports the change. A single aryl group
covalently bonded to one carbon of a pyrene converts that carbon from sp² to
sp³. Two things happen at once:

- The local π conjugation is broken, creating a **localised optical transition**
  distinct from the pristine host — this is what makes it a colour centre, and
  it is the same physics that produces sp³ quantum defects in carbon nanotubes.
- One carbon leaves the π system, so the remaining π electron count is odd: the
  defect carries an **unpaired spin, S = 1/2**.

Pyrene C₁₆H₁₀ plus phenyl C₆H₅ gives **C₂₂H₁₅**, 147 electrons, a ground-state
doublet. The same defect is therefore simultaneously an optical emitter and a
spin label, which is what lets one construct report through two independent
channels.

The colour centre is tethered to the catalyst through a 1,4-phenylene spacer
attached to the meso carbon of one chelate arm. The *other* arm is the labile
one, so the degradation event and the reporter sit on opposite sides of the
metal and the signal must travel through the metal centre rather than along a
bond.

### The three readouts

| Channel | Observable | Physical basis |
|---|---|---|
| Magnetic | exchange coupling *J* between defect spin and metal | zero while Ni is S = 0; becomes finite once Ni is paramagnetic |
| Optical | shift of the lowest bright defect transition | the defect's excited state feels the changed metal electronic structure |
| Spin density | Mulliken/Löwdin spin population on defect vs metal | the microscopic mechanism behind both of the above |

A single channel can be fooled — solvatochromism shifts an optical line,
temperature changes a magnetic susceptibility. Two channels moving together, in
the way this model predicts, is a far stronger claim.

---

## 2. Why these particular models

### The catalyst is truncated, deliberately

Real salen is C₁₆H₁₄N₂O₂ around the metal, 35 atoms with Ni. We use
bis(iminoenolato)nickel(II), **19 atoms**, two bidentate [NH=CH–CH=CH–O]⁻ arms
in the *trans* arrangement.

This keeps what matters — a d⁸ metal in a square-planar N₂O₂ field with a
six-membered chelate ring and a conjugated backbone — and drops the aryl rings,
which contribute to the ligand field only weakly. Three payoffs:

1. CASSCF on the d manifold becomes affordable, so the DFT spin-state gaps can
   be cross-checked against a multireference method.
2. A chelate arm can dissociate *completely*. In real salen the tetradentate
   ligand keeps a decoordinated imine tethered nearby, which muddles a clean
   ligand-loss coordinate.
3. The 54-atom assembled sensor stays inside reach of TD-DFT.

The cost is that absolute ligand-field strengths are not those of real salen.
Conclusions here are about the *mechanism* and the *direction and rough
magnitude* of the response, not about a specific catalyst's numbers.

### The colour centre is a molecular analogue, not a nanotube

Genuine organic colour centres are sp³ defects on semiconducting single-walled
carbon nanotubes, where a delocalised 1D exciton is trapped at the defect.
Psi4 is a molecular code and cannot treat a periodic nanotube.

Pyrene is the smallest host with a genuinely delocalised, multi-ring π system
that still permits TD-DFT, EOM-CCSD benchmarking and CASSCF on the same
molecule. The defect physics — broken conjugation, a localised state pulled
below the host's absorption, an unpaired spin at the defect — is the same
mechanism. The *magnitudes* differ: a real nanotube OCC red-shifts emission by
roughly 100–300 meV from a band edge near 1 eV, while a pyrene defect state sits
several eV higher. This is a mechanism model, and the document should not be
read as predicting nanotube emission wavelengths.

---

## 3. Method, and where it is weak

### Spin-state energetics are the hard part

Everything here depends on a 3d transition-metal singlet–triplet gap, which is
the single least reliable quantity in routine DFT. The gap depends strongly on
the exact-exchange fraction: more Hartree–Fock exchange stabilises the high-spin
state, and the swing across common functionals reaches tens of kcal/mol.

We therefore **never report a single functional**. Every spin-state quantity is
computed across a ladder spanning 0% to ~27% exact exchange (BP86, PBE, TPSS,
TPSSh, B3LYP, PBE0, M06, ωB97X-D) and the spread is reported as the uncertainty.
Where the ladder disagrees about the *sign* of a gap, that is stated rather than
averaged away.

### The validation tier exists because of two real traps

Before touching nickel, the machinery is tested on small molecules with
precisely known singlet–triplet gaps. Building that tier surfaced two errors
that would have silently corrupted every downstream number.

**Trap 1: comparing against the wrong experimental state.** For O₂ and NH the
lowest singlet is one of a degenerate pair arising from a π² configuration
(¹Δ and ¹Σ⁺). A single closed-shell determinant (πₓ)² is *not* the ¹Δ state — it
is an equal mixture of ¹Δ and ¹Σ⁺, and its energy should be compared with the
**mean** of the two term energies. Comparing it with ¹Δ alone charges DFT with
an error it did not commit:

| System | ¹Δ (kcal/mol) | ¹Σ⁺ (kcal/mol) | single-determinant reference |
|---|---|---|---|
| O₂ | 22.64 | 37.73 | **30.18** |
| NH | 35.93 | 60.62 | **48.27** |

Using the naive reference inflated apparent errors by 8–12 kcal/mol, more than
the real functional spread.

**Trap 2: the unrestricted singlet collapsing.** Running the closed-shell state
with an unrestricted reference lets the SCF slide into a broken-symmetry Ms = 0
solution, which is roughly a 50:50 singlet/triplet mixture and therefore far too
low. In the first pass this produced gaps of 7.6 kcal/mol for O₂ and ~14
kcal/mol for NH — errors of −23 and −34 — purely from the choice of reference.
Closed-shell states are now run restricted, which cannot spin-break by
construction, and ⟨S²⟩ of the high-spin reference is reported so contamination
is visible rather than hidden.

Both traps are the same lesson: a spin-state gap is only meaningful once you can
say exactly *which two states* produced it.

### ⟨S²⟩ is computed, not trusted

Psi4 does not expose ⟨S²⟩ as a variable in the way one might expect
(`psi4.variable("S^2")` raises). It is computed directly from the AO overlap in
the psi4numpy idiom:

$$\langle S^2\rangle = S_z(S_z+1) + N_\beta - \sum_{ij}\left|\langle \phi_i^\alpha | \phi_j^\beta\rangle\right|^2$$

with the α/β occupied-block overlap **Cₐᵀ S Cᵦ**. This is the diagnostic that
tells us whether a broken-symmetry solution is what we think it is.

### Exchange coupling via broken symmetry

The defect–metal magnetic coupling is extracted with broken-symmetry DFT and the
Yamaguchi projection,

$$J = \frac{E_{\mathrm{BS}} - E_{\mathrm{HS}}}{\langle S^2\rangle_{\mathrm{HS}} - \langle S^2\rangle_{\mathrm{BS}}}$$

under the convention *H* = −2*J* **S₁·S₂**, so **J < 0 is antiferromagnetic**.
The Yamaguchi form is used rather than the Ziegler/Noodleman limits because it
interpolates correctly between weak and strong coupling without assuming which
regime applies — and along a dissociation coordinate the regime changes.

### SAPT is used where SAPT applies

The colour centre is *covalently* tethered, so symmetry-adapted perturbation
theory cannot decompose that interaction — SAPT treats non-covalent interactions
only. It is instead applied to a question where it is valid: once ligand loss
opens a coordination site, a water molecule can bind there, and SAPT0
decomposes that interaction into electrostatics, exchange, induction and
dispersion. Both fragments are closed-shell, so standard closed-shell SAPT0 is
appropriate. This characterises what the vacated site actually *is*, chemically.

### Known limitations

- **Gas phase, 0 K, no ZPE.** No solvent, no thermal or entropic corrections.
  Real ligand dissociation is strongly solvent-assisted, and a solvated open
  site is stabilised in a way this model omits.
- **The scan coordinate is a proxy.** Freezing Ni–N and relaxing everything else
  is not a minimum-energy path and gives no transition state or barrier. It
  answers "what does the electronic structure do as the arm leaves", not "how
  fast does the arm leave".
- **TD-DFT on a spin-contaminated open-shell reference** is a known weak point
  for radicals. The reference ⟨S²⟩ is reported alongside every excitation energy
  so the reader can discount accordingly.
- **Single-reference character is assumed and then checked.** Near the
  spin-crossing the wavefunction may acquire multireference character, where DFT
  is least trustworthy; CASSCF natural-orbital occupations are used to test this
  rather than assuming it away.
- **The truncated catalyst and molecular colour-centre host** mean absolute
  energies and wavelengths are model quantities, not predictions for a specific
  real material.

---

## 4. Computational tiers

| Tier | System | Size | Methods |
|---|---|---|---|
| 0 Validation | CH₂, O₂, NH | 2–3 atoms | functional ladder, adiabatic gaps vs experiment |
| 1 Colour centre | pyrene, C₂₂H₁₅ defect | 26 / 37 atoms | TD-DFT, EOM-CCSD, spin density |
| 2 Catalyst | Ni(iminoenolate)₂ | 19 atoms | relaxed ligand-loss scan, spin gaps, CASSCF cross-check |
| 3 Assembly | tethered sensor | 54 atoms | single-point spin + optical readouts |

Tier 3 is where CPU runs out. Psi4 has no GPU path — it is CPU-only — so the
large TD-DFT work is offloaded to GPU4PySCF on a Colab GPU runtime via the
Colab CLI. See `gpu/README.md`; that path requires the operator's own Google
authentication and is not exercised in the automated runs.

---

## 5. Reproducing

```bash
./setup.sh                                              # micromamba + Psi4 from conda-forge
micromamba run -n nimrod python scripts/run_validation.py
micromamba run -n nimrod python scripts/run_catalyst_scan.py
micromamba run -n nimrod python -m pytest -q
```

Every calculation is cached in `data/cache/` keyed by a hash of the full job
specification, so an interrupted run resumes rather than restarting.
