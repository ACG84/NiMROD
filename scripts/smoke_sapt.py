"""Smoke test for :mod:`nimrod.sapt`.

Every SAPT check here has a known qualitative answer, so a broken psivar
harvest or a mis-signed component cannot hide:

* water dimer   -- the canonical hydrogen bond.  SAPT0/jun-cc-pVDZ must give a
                   total near -5 kcal/mol, dominated by electrostatics, with a
                   large *positive* exchange.  A positive total or a negative
                   exchange means the harvest is wrong.
* He...He 3 A   -- essentially unbound, and what little binding there is must be
                   almost pure dispersion.  Run in aug-cc-pVDZ, not the module
                   default: the jun- truncation strips the diffuse functions
                   from helium entirely, and without them there is no
                   dispersion left to find.
* BH3...OH2     -- a dative bond, the closest cheap analogue of what a water
                   does at a vacated metal site.  Induction must be a far
                   larger share of the attraction here than in the water dimer.
                   This is the term the degradation profile is built on, so it
                   needs its own check.
* counterpoise  -- an independent supermolecular cross-check on the water
                   dimer.  Different theory, different error modes; if it lands
                   within a couple of kcal/mol of the SAPT0 total, both are
                   probably describing the same interaction.
* guards        -- open-shell fragments, the doublet sensor assembly, wrong
                   fragment count and unavailable -D3 functionals must all
                   raise before anything expensive is submitted.
* geometry      -- pure numpy, no SCF: the water probe's internal geometry, and
                   the docking of that probe onto the vacated site of a
                   partially degraded Ni catalyst.

Deliberately NOT run here: SAPT0 on the real Ni catalyst + water.  That is a
22-atom def2-SVPD SAPT0 job, far outside "tiny", and this box is shared.  The
Ni path is exercised geometrically and through the basis-selection logic only.

Run with:

    MAMBA_ROOT_PREFIX=... NIMROD_THREADS=1 \\
        ./bin/micromamba run -n nimrod python scripts/smoke_sapt.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nimrod.geometry import (  # noqa: E402
    Structure,
    elongate_bond,
    ni_salen_model,
    sensor_assembly,
)
from nimrod.sapt import (  # noqa: E402
    DEFAULT_SAPT_BASIS,
    DEFAULT_SAPT_METAL_BASIS,
    SaptScopeError,
    assert_closed_shell_fragments,
    capture_profile,
    catalyst_water_dimer,
    counterpoise_binding,
    parse_fragments,
    sapt0_interaction,
    two_fragment_geometry,
    water_probe,
)

USE_CACHE = False  # the psivar harvest is not part of the cache key; stay honest

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"    {'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        FAILURES.append(message)


def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def report(components) -> None:
    print(f"    {components}")
    if components.converged:
        shares = components.percent_attractive()
        print(
            "    attraction shares:  "
            + "  ".join(f"{k} {v:5.1f}%" for k, v in shares.items())
            + f"   [{components.wall_seconds:.1f} s]"
        )
    for issue in components.warnings():
        print(f"    WARNING: {issue}")


# --------------------------------------------------------------------------
# Reference geometries
# --------------------------------------------------------------------------

WATER_A = Structure.from_psi4(
    """
    O  -1.551007  -0.114520   0.000000
    H  -1.934259   0.762503   0.000000
    H  -0.599677   0.040712   0.000000
    """,
    "water-a",
)
WATER_B = Structure.from_psi4(
    """
    O   1.350625   0.111469   0.000000
    H   1.680398  -0.373741  -0.758561
    H   1.680398  -0.373741   0.758561
    """,
    "water-b",
)

HE_A = Structure(["He"], np.array([[0.0, 0.0, 0.0]]), "he-a")
HE_B = Structure(["He"], np.array([[0.0, 0.0, 3.0]]), "he-b")

#: Planar BH3, boron at the origin, empty p orbital along z.
BH3 = Structure(
    ["B", "H", "H", "H"],
    np.array(
        [[0.0, 0.0, 0.0]]
        + [
            [1.19 * math.cos(math.radians(120 * k)),
             1.19 * math.sin(math.radians(120 * k)), 0.0]
            for k in range(3)
        ]
    ),
    "bh3",
)


# --------------------------------------------------------------------------
# 1. Geometry: the probe molecule
# --------------------------------------------------------------------------

banner("1. Water probe geometry (pure numpy, no SCF)")

axis = np.array([0.0, 0.0, 1.0])
probe = water_probe(np.array([0.0, 0.0, 2.05]), -axis, plane_hint=[1.0, 0.0, 0.0])
o, h1, h2 = probe.coords

r1 = float(np.linalg.norm(h1 - o))
r2 = float(np.linalg.norm(h2 - o))
hoh = math.degrees(
    math.acos(np.dot(h1 - o, h2 - o) / (np.linalg.norm(h1 - o) * np.linalg.norm(h2 - o)))
)
# Angle between the aiming direction (O -> metal, i.e. -z) and each O-H bond.
aim = -axis
tilt1 = math.degrees(math.acos(np.dot(aim, (h1 - o) / r1)))
tilt2 = math.degrees(math.acos(np.dot(aim, (h2 - o) / r2)))

print(f"    r(O-H) = {r1:.4f}, {r2:.4f} A")
print(f"    angle(H-O-H)       = {hoh:.2f} deg")
print(f"    angle(M...O-H)     = {tilt1:.2f}, {tilt2:.2f} deg")
check(abs(r1 - 0.9572) < 1e-6 and abs(r2 - 0.9572) < 1e-6, "O-H bond lengths are 0.9572 A")
check(abs(hoh - 104.52) < 1e-4, "H-O-H angle is 104.52 deg")
check(abs(tilt1 - 127.74) < 1e-2 and abs(tilt2 - 127.74) < 1e-2,
      "M...O-H angles are 127.74 deg (lone-pair-first, C2v)")
check(abs(o[2] - 2.05) < 1e-12, "oxygen sits exactly where it was asked to")


# --------------------------------------------------------------------------
# 2. Geometry: docking onto the degraded Ni catalyst
# --------------------------------------------------------------------------

banner("2. Docking the probe at the vacated Ni site (pure numpy, no SCF)")

catalyst, cat_index = ni_salen_model()
ni, n_labile = cat_index["Ni"], cat_index["N2"]
print(f"    catalyst {catalyst.formula}, {len(catalyst)} atoms, "
      f"Ni index {ni}, labile N index {n_labile}")

clashes = []
for ni_n in (1.87, 2.60, 3.30, 4.50, 6.00):
    stretched = elongate_bond(catalyst, ni, n_labile, ni_n,
                              carry=[cat_index["Hn2"]])
    trial = catalyst_water_dimer(stretched, ni, leaving_index=n_labile)
    warn = trial.clash_warning()
    clashes.append(warn is not None)
    i, j = trial.closest_pair
    print(f"    Ni-N {ni_n:4.2f} A ->  Ni...O {trial.metal_oxygen_distance:.3f} A, "
          f"closest non-metal contact {trial.closest_contact:.2f} A "
          f"({trial.structure.symbols[i]}{i}...{trial.structure.symbols[j]}{j})"
          f"{'   [CLASH]' if warn else ''}")

print("""
    NOTE, and this is a real limitation rather than a test artefact:
    geometry.degradation_series pulls the nitrogen straight out along the Ni-N
    axis and carries only its hydrogen, so the vacated site stays blocked --
    first by the receding N itself, then by the Cc/Hcc atoms of the same arm,
    which never move.  Every rigid point above therefore clashes, and the clash
    detector correctly says so.  A chemically meaningful capture profile needs
    the RELAXED geometries from nimrod.degradation, where the arm swings aside.
""")
check(all(clashes),
      "clash detector fires on every rigidly-elongated point (the arm blocks "
      "the site)")

dimer = catalyst_water_dimer(
    elongate_bond(catalyst, ni, n_labile, 6.00, carry=[cat_index["Hn2"]]),
    ni, leaving_index=n_labile,
)
check(abs(dimer.metal_oxygen_distance - 2.05) < 1e-6,
      "Ni...O lands at the requested 2.05 A")
check(dimer.n_atoms_a == len(catalyst) and dimer.n_atoms_b == 3,
      "fragment sizes are (catalyst, H2O)")
check(dimer.geometry.count("--") == 1,
      "geometry carries exactly one '--' fragment separator")
check(dimer.geometry.splitlines()[0].strip() == "0 1",
      "fragment A begins with its own charge/multiplicity line")
frags = parse_fragments(dimer.geometry)
check(len(frags) == 2 and all(f.explicit_header for f in frags),
      "both fragments carry an explicit charge/multiplicity header")
check("symmetry" not in dimer.geometry and "no_com" not in dimer.geometry,
      "no symmetry/no_com directives (build_molecule appends those)")

# The probe oxygen must lie on the Ni -> leaving-N ray, i.e. in the vacancy.
ray = dimer.structure.coords[n_labile] - dimer.structure.coords[ni]
to_o = dimer.structure.coords[dimer.oxygen_index] - dimer.structure.coords[ni]
cosine = float(np.dot(ray, to_o) / (np.linalg.norm(ray) * np.linalg.norm(to_o)))
check(cosine > 0.9999, f"probe oxygen lies on the vacated-site ray (cos = {cosine:.6f})")

# Hydrogens should be out of the coordination plane, not into the cis donors.
plane_z = np.abs(dimer.structure.coords[dimer.oxygen_index + 1: dimer.oxygen_index + 3, 2])
check(bool(np.all(plane_z > 0.5)),
      f"probe hydrogens are out of the NiN2O2 plane (|z| = {plane_z.round(2).tolist()} A)")


# --------------------------------------------------------------------------
# 3. Guards
# --------------------------------------------------------------------------

banner("3. Closed-shell guards (must raise before anything is submitted)")

assembly, info = sensor_assembly()
print(f"    sensor assembly {assembly.formula}, {len(assembly)} atoms, "
      f"{assembly.n_electrons} electrons -> multiplicity "
      f"{assembly.multiplicity_for(0)}")

cases = [
    (
        "doublet sensor assembly declared as a singlet fragment",
        two_fragment_geometry(assembly, WATER_B),
    ),
    (
        "explicitly open-shell fragment (multiplicity 3)",
        two_fragment_geometry(WATER_A, WATER_B, multiplicity_a=3),
    ),
    (
        "single fragment, no '--' separator",
        f"0 1\n{WATER_A.to_psi4()}",
    ),
    (
        "three fragments",
        f"0 1\n{WATER_A.to_psi4()}\n--\n0 1\n{WATER_B.to_psi4()}\n--\n0 1\n"
        f"{HE_B.to_psi4()}",
    ),
]
for description, geom in cases:
    try:
        assert_closed_shell_fragments(geom)
        check(False, f"guard rejects: {description}")
    except SaptScopeError as exc:
        check(True, f"guard rejects: {description}")
        print(f"          -> {str(exc).splitlines()[0][:150]}")

try:
    counterpoise_binding(two_fragment_geometry(WATER_A, WATER_B), method="b3lyp-d3bj")
    check(False, "guard rejects -D3 functionals (dftd3 missing in this env)")
except SaptScopeError as exc:
    check(True, "guard rejects -D3 functionals (dftd3 missing in this env)")
    print(f"          -> {str(exc)[:200]}")

# Basis auto-selection: metals must not be handed jun-cc-pvdz.
from nimrod.sapt import _needs_metal_basis  # noqa: E402

check(not _needs_metal_basis(two_fragment_geometry(WATER_A, WATER_B)),
      f"organic dimer keeps the default {DEFAULT_SAPT_BASIS}")
check(_needs_metal_basis(dimer.geometry),
      f"Ni dimer falls back to {DEFAULT_SAPT_METAL_BASIS} (jun-cc-pVDZ has no Ni)")

# Ghost centres carry basis functions but no electrons.  Psi4 spells them two
# ways and the guard has to accept both, or a legal counterpoise geometry gets
# refused with a bogus "cannot identify the element".
for spelling, block in (
    ("Gh(O)", "0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n--\n0 1\nGh(O) 0 0 3\nHe 0 0 4"),
    ("@O", "0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n--\n0 1\n@O 0 0 3\nHe 0 0 4"),
):
    try:
        ghost_frags = parse_fragments(block)
        ok = (len(ghost_frags) == 2
              and ghost_frags[1].n_ghosts == 1
              and ghost_frags[1].symbols == ["He"]
              and ghost_frags[1].n_electrons == 2)
    except Exception as exc:  # noqa: BLE001 - the point is that it must not raise
        ok = False
        print(f"          -> {type(exc).__name__}: {str(exc)[:120]}")
    check(ok, f"ghost centres spelled {spelling} parse as zero-electron centres")


# --------------------------------------------------------------------------
# 4. Water dimer: the canonical hydrogen bond
# --------------------------------------------------------------------------

banner("4. Water dimer, SAPT0/jun-cc-pVDZ")

water_dimer = two_fragment_geometry(WATER_A, WATER_B)
water = sapt0_interaction(water_dimer, "jun-cc-pvdz",
                          label="water-dimer", use_cache=USE_CACHE)
report(water)

check(water.converged, "SAPT0 converged")
if water.converged:
    check(-6.0 < water.total < -3.5,
          f"total is {water.total:.3f} kcal/mol (expected roughly -4 to -5)")
    check(water.exchange > 0.0,
          f"exchange is repulsive ({water.exchange:+.3f} kcal/mol)")
    check(water.electrostatics < 0.0,
          f"electrostatics is attractive ({water.electrostatics:+.3f} kcal/mol)")
    check(abs(water.electrostatics) > abs(water.induction)
          and abs(water.electrostatics) > abs(water.dispersion),
          "electrostatics dominates the attraction")
    check(water.character() == "electrostatic",
          f"site character reads '{water.character()}'")
    check(abs(water.component_sum - water.total) < 1e-6,
          "components sum to the reported total")
    check(not water.warnings(), "no physical sanity warnings")

    nan = float("nan")
    dhf = water.extras.get("delta_hf", nan)
    ind2 = water.extras.get("induction2_dimer_basis", nan)
    print(f"    delta-HF (SAPT HF(2) ENERGY)      = {dhf:+.4f} kcal/mol")
    print(f"    Ind(2), dimer basis ('SAPT CT')   = {ind2:+.4f} kcal/mol")
    # Psi4 publishes that last number under the name 'SAPT CT ENERGY', but after
    # a plain sapt0 run it is second-order induction, not charge transfer: it is
    # identically ind20,r + exch-ind20,r = induction - delta_HF.  Assert the
    # identity so the misleading label can never quietly come back.
    check(
        abs(ind2 - (water.induction - dhf)) < 1e-6,
        f"'SAPT CT ENERGY' is Ind(2) in the dimer basis, not charge transfer "
        f"({ind2:+.4f} == induction - delta_HF = {water.induction - dhf:+.4f})",
    )
    check(
        abs(water.extras.get("ind20r", nan) + water.extras.get("exch_ind20r", nan)
            - ind2) < 1e-6,
        "ind20,r + exch-ind20,r reproduces it exactly",
    )
    check(
        abs(water.extras.get("disp20", nan) + water.extras.get("exch_disp20", nan)
            - water.dispersion) < 1e-6,
        "disp20 + exch-disp20 reproduces the reported dispersion",
    )
    check(
        abs(water.extras.get("elst10r", nan) - water.electrostatics) < 1e-6
        and abs(water.extras.get("exch10", nan) - water.exchange) < 1e-6,
        "elst10,r and exch10 reproduce the reported elst/exch",
    )


# --------------------------------------------------------------------------
# 5. He dimer: near zero, dispersion-dominated
# --------------------------------------------------------------------------

banner("5. He...He at 3.0 A, SAPT0/aug-cc-pVDZ")

he = sapt0_interaction(two_fragment_geometry(HE_A, HE_B), "aug-cc-pvdz",
                       label="he-dimer", use_cache=USE_CACHE)
report(he)

check(he.converged, "SAPT0 converged")
if he.converged:
    check(abs(he.total) < 0.1,
          f"total is near zero ({he.total:+.5f} kcal/mol)")
    check(he.total < 0.0, f"weakly bound ({he.total:+.5f} kcal/mol)")
    check(he.exchange > 0.0, f"exchange is repulsive ({he.exchange:+.5f} kcal/mol)")
    check(abs(he.dispersion) > abs(he.electrostatics)
          and abs(he.dispersion) > abs(he.induction),
          "dispersion dominates the attraction")
    check(he.character() == "dispersion-bound",
          f"site character reads '{he.character()}'")


# --------------------------------------------------------------------------
# 6. BH3...OH2: a dative bond, the induction channel under test
# --------------------------------------------------------------------------

banner("6. BH3...OH2 at 1.80 A, SAPT0/jun-cc-pVDZ  (dative-bond analogue)")

bh3_water = catalyst_water_dimer(
    BH3, 0, direction=[0.0, 0.0, 1.0], metal_oxygen_distance=1.80,
)
print(f"    B...O {bh3_water.metal_oxygen_distance:.3f} A, "
      f"closest non-B contact {bh3_water.closest_contact:.2f} A")
check(bh3_water.clash_warning() is None,
      "an unobstructed site docks cleanly (no clash warning)")
dative = sapt0_interaction(bh3_water, "jun-cc-pvdz",
                           label="bh3-water", use_cache=USE_CACHE)
report(dative)

check(dative.converged, "SAPT0 converged")
if dative.converged and water.converged:
    ind_dative = dative.percent_attractive()["induction"]
    ind_water = water.percent_attractive()["induction"]
    print(f"    induction share: dative {ind_dative:.1f}%  vs  "
          f"hydrogen bond {ind_water:.1f}%")
    check(dative.exchange > 0.0, "exchange is repulsive")
    check(ind_dative > ind_water,
          "induction is a larger share of the attraction than in the water dimer")
    check(ind_dative > 30.0,
          f"induction share exceeds 30% ({ind_dative:.1f}%), as a dative bond must")


# --------------------------------------------------------------------------
# 7. Counterpoise cross-check
# --------------------------------------------------------------------------

banner("7. Counterpoise cross-check on the water dimer")

for method in ("wb97x-d", "b3lyp"):
    cp = counterpoise_binding(water_dimer, method=method, basis="jun-cc-pvdz",
                              label=f"water-cp-{method}", use_cache=USE_CACHE)
    print(f"    {cp}   [{cp.wall_seconds:.1f} s]")
    if cp.converged:
        print(f"        uncorrected {cp.raw_binding:+.3f} kcal/mol")
    check(cp.converged, f"CP binding converged ({method})")
    if cp.converged and water.converged:
        check(cp.bsse is not None and cp.bsse > 0.0,
              f"BSSE is positive, i.e. the raw number overbinds ({method})")
        check(cp.agrees_with(water, tolerance=2.0),
              f"CP binding agrees with the SAPT0 total to 2 kcal/mol "
              f"({cp.binding:+.3f} vs {water.total:+.3f}, "
              f"delta {cp.binding - water.total:+.3f}) ({method})")


# --------------------------------------------------------------------------
# 8. capture_profile end to end
# --------------------------------------------------------------------------

banner("8. capture_profile over a Lewis-acidity coordinate (BH3 pyramidalisation)")

print("""    A stand-in for the real degradation profile, small enough to run on a
    shared box.  Bending the BH3 hydrogens away from the incoming water is the
    geometric change a borane undergoes as it forms a dative bond, and it
    lowers the acceptor orbital, so the INDUCTION share should rise
    monotonically.  That is the same term the Ni open-site profile is built on,
    so if it does not respond here it will not be trustworthy there.""")


def pyramidal_bh3(theta_deg: float) -> Structure:
    """BH3 with the hydrogens bent ``theta_deg`` away from +z."""
    t = math.radians(theta_deg)
    return Structure(
        ["B", "H", "H", "H"],
        np.array(
            [[0.0, 0.0, 0.0]]
            + [
                [1.19 * math.cos(t) * math.cos(math.radians(120 * k)),
                 1.19 * math.cos(t) * math.sin(math.radians(120 * k)),
                 -1.19 * math.sin(t)]
                for k in range(3)
            ]
        ),
        f"bh3-{theta_deg:.0f}",
    )


profile = capture_profile(
    [(theta, pyramidal_bh3(theta)) for theta in (0.0, 10.0, 20.0)],
    metal_index=0,
    direction=[0.0, 0.0, 1.0],
    basis="jun-cc-pvdz",
    metal_oxygen_distance=1.80,
    label="bh3-pyramidalisation",
    use_cache=USE_CACHE,
)

for point in profile.points:
    shares = point.sapt.percent_attractive()
    print(f"    theta {point.coordinate:5.1f} deg  {point.sapt}")
    print(f"                     induction share {shares['induction']:5.1f}%")

check(all(p.sapt.converged for p in profile.points),
      "every profile point converged")
check(profile.distances == [0.0, 10.0, 20.0], "coordinates round-trip")
series = profile.component_series()
check(set(series) == {"electrostatics", "exchange", "induction",
                      "dispersion", "total"},
      "component_series exposes the keys plots.plot_sapt_components wants")
check(all(len(v) == 3 for v in series.values()),
      "component_series has one value per point")

induction_shares = [p.sapt.percent_attractive()["induction"] for p in profile.points]
check(induction_shares == sorted(induction_shares),
      f"induction share rises with pyramidalisation "
      f"({', '.join(f'{s:.1f}%' for s in induction_shares)})")

out = profile.save(Path(__file__).resolve().parent.parent
                   / "data" / "smoke" / "sapt-capture-profile.json")
check(out.exists() and out.stat().st_size > 0, f"profile serialises to {out}")

# figures.build_sapt_figure reads a FLAT per-point schema out of
# data/results/sapt_scan.json: p["distance"] plus p["electrostatics"] etc.  If
# the serialiser only emitted the nested "sapt" block the figure builder would
# silently produce nothing, so pin the on-disk contract here.
import json as _json  # noqa: E402

saved = _json.loads(out.read_text())
saved_points = saved.get("points", [])
check(
    len(saved_points) == 3
    and all("distance" in p for p in saved_points)
    and all(
        isinstance(p.get(k), float)
        for p in saved_points
        for k in ("electrostatics", "exchange", "induction", "dispersion")
    ),
    "saved points carry the flat schema figures.build_sapt_figure reads",
)


# --------------------------------------------------------------------------

banner("SUMMARY")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
