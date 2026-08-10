"""Spin-manifold analysis: expectation values, spin densities and exchange coupling.

The NiMROD sensor is a two-spin problem.  One spin is the S = 1/2 unpaired
electron of the organic colour centre, pinned on the sp3 aryl defect of pyrene.
The other lives on the nickel: zero for the intact square-planar d8 complex, but
S = 1 once a chelate arm dissociates and the ligand field drops to three-
coordinate.  Everything this project claims about "sensing" is a statement about
that second spin appearing and then talking to the first one.

Three observables are extracted here.

``<S^2>``
    The total-spin expectation value of a single Slater determinant.  For an
    unrestricted determinant it is *not* S(S+1): the alpha and beta manifolds
    relax independently, so the determinant is a mixture of spin states.  The
    deviation (spin contamination) is not merely an error bar — for a
    broken-symmetry solution it is the quantity that tells you how much
    higher-spin character has leaked in, and it appears directly in the
    denominator of the Yamaguchi coupling formula.

    Psi4 1.11 does **not** publish an ``S^2`` psivar (``psi4.variable("S^2")``
    raises ``KeyError``), so it is computed here from the MO coefficients using
    the standard determinantal expression

        <S^2> = Sz(Sz + 1) + N_beta - sum_ij |<phi_i^alpha | phi_j^beta>|^2

    with i over occupied alpha and j over occupied beta spin orbitals.  The
    third term is the sum of squared alpha/beta MO overlaps; when the two
    manifolds span the same space it cancels N_beta exactly and <S^2> collapses
    to the pure Sz(Sz+1) of a restricted-open determinant.

Atomic spin populations
    Where the unpaired density actually sits.  Both Mulliken and Löwdin
    partitions are reported, because Mulliken is basis-set-sensitive to the
    point of embarrassment while Löwdin (symmetric orthogonalisation) is
    steadier; agreement between the two is the cheapest available check that a
    localisation claim is real rather than a basis-set artefact.  The sensing
    readout "spin density moves off the defect" is a statement about these
    numbers, integrated over fragments.

Exchange coupling J
    Extracted by broken-symmetry DFT.  A single determinant cannot represent the
    open-shell singlet of two antiferromagnetically coupled spins, so one
    converges the unphysical Ms = 0 (or Ms = |S1 - S2|) broken-symmetry
    determinant and maps its energy onto the Heisenberg ladder.  The Yamaguchi
    expression is used because it interpolates correctly between the weak- and
    strong-overlap limits by dividing out the actual computed spin
    contamination rather than assuming an idealised value.

Sign convention (stated once, used everywhere in this module):

    H = -2 J S1 . S2

    J > 0  ->  ferromagnetic  (high-spin state lies lower)
    J < 0  ->  antiferromagnetic (low-spin / broken-symmetry state lies lower)

Beware when comparing with the literature: the competing convention
H = -J S1.S2 doubles every number, and some authors flip the overall sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import HARTREE_TO_CM, HARTREE_TO_KCAL
from .psi4_driver import JobResult, JobSpec, run_energy

__all__ = [
    "spin_squared",
    "ideal_s2",
    "SpinPopulations",
    "atomic_spin_populations",
    "spin_properties",
    "populations_from_properties",
    "fragment_spin",
    "fragment_spins",
    "split_on_bond",
    "sensor_fragments",
    "unrestricted_reference",
    "high_spin_multiplicity",
    "broken_symmetry_multiplicity",
    "run_high_spin",
    "run_broken_symmetry",
    "yamaguchi_j",
    "ExchangeCoupling",
    "exchange_coupling",
]


#: A broken-symmetry determinant whose contamination sits below this above its
#: nominal S(S+1) has collapsed back onto the closed-shell solution.
BS_COLLAPSE_THRESHOLD = 0.05


# --------------------------------------------------------------------------
# <S^2> from the wavefunction
# --------------------------------------------------------------------------


def _ao_overlap(wfn) -> np.ndarray:
    """AO overlap matrix of the wavefunction's own basis, as a numpy array."""
    import psi4

    return np.asarray(psi4.core.MintsHelper(wfn.basisset()).ao_overlap())


def _is_closed_shell(wfn) -> bool:
    """True for a genuinely restricted closed-shell determinant.

    ROHF/ROKS also report ``same_a_b_orbs``, but their densities differ, so the
    density test is what separates "closed shell" from "restricted open shell".
    """
    return bool(
        wfn.nalpha() == wfn.nbeta()
        and wfn.same_a_b_orbs()
        and wfn.same_a_b_dens()
    )


def spin_squared(wfn) -> float:
    """``<S^2>`` of the single determinant carried by ``wfn``.

    Implements

        <S^2> = Sz(Sz + 1) + N_beta - sum_ij |<phi_i^a | phi_j^b>|^2

    with the alpha/beta MO overlap block built as ``Ca_occ.T @ S_ao @ Cb_occ``.

    Exactly ``0.0`` is returned for a closed-shell restricted determinant (where
    the general expression is zero anyway, up to numerical noise).  For a ROHF /
    ROKS determinant the alpha occupied space contains the beta occupied space,
    the sum cancels ``N_beta`` term for term, and the result is the pure
    ``Sz(Sz + 1)``, as it must be.
    """
    if _is_closed_shell(wfn):
        return 0.0

    n_alpha = int(wfn.nalpha())
    n_beta = int(wfn.nbeta())
    sz = 0.5 * (n_alpha - n_beta)

    if n_beta == 0:  # no beta electrons: the determinant is a pure high-spin state
        return float(sz * (sz + 1.0))

    s_ao = _ao_overlap(wfn)
    ca_occ = np.asarray(wfn.Ca_subset("AO", "OCC"))
    cb_occ = np.asarray(wfn.Cb_subset("AO", "OCC"))

    # <phi_i^alpha | phi_j^beta>, shape (n_alpha, n_beta)
    mo_overlap = ca_occ.T @ s_ao @ cb_occ

    s2 = float(sz * (sz + 1.0) + n_beta - np.sum(mo_overlap**2))

    # The expression subtracts two numbers of size N_beta, so a determinant with
    # no spin polarisation lands on zero only to machine precision and can come
    # out at -1e-16.  <S^2> is positive semi-definite, so clean that up — but
    # only within noise, since a genuinely negative value would mean the MO
    # overlap block is wrong and must not be swept away.
    return 0.0 if abs(s2) < 1e-10 else s2


def ideal_s2(multiplicity: int) -> float:
    """``S(S + 1)`` for a spin state of the given multiplicity ``2S + 1``."""
    s = 0.5 * (int(multiplicity) - 1)
    return float(s * (s + 1.0))


# --------------------------------------------------------------------------
# Atomic spin populations
# --------------------------------------------------------------------------


@dataclass
class SpinPopulations:
    """Per-atom spin populations from one wavefunction.

    ``mulliken`` and ``lowdin`` are both length-``natom`` arrays of net alpha
    minus beta population.  Both sum exactly to ``N_alpha - N_beta``; that
    identity is the module's internal consistency check.
    """

    symbols: list[str]
    mulliken: np.ndarray
    lowdin: np.ndarray
    charges: np.ndarray  #: Mulliken atomic charges, for context (not spin)

    @property
    def natom(self) -> int:
        return len(self.symbols)

    def total(self, kind: str = "mulliken") -> float:
        return float(self.get(kind).sum())

    def get(self, kind: str = "mulliken") -> np.ndarray:
        try:
            return {"mulliken": self.mulliken, "lowdin": self.lowdin}[kind]
        except KeyError:
            raise KeyError(f"unknown population scheme {kind!r}") from None

    def largest(self, count: int = 5, kind: str = "mulliken") -> list[tuple[int, str, float]]:
        """The ``count`` atoms carrying the most spin density, by magnitude."""
        pops = self.get(kind)
        order = np.argsort(-np.abs(pops))[:count]
        return [(int(i), self.symbols[i], float(pops[i])) for i in order]


def _ao_centres(wfn) -> np.ndarray:
    """Map AO basis-function index -> atom index."""
    basis = wfn.basisset()
    return np.array(
        [basis.function_to_center(mu) for mu in range(basis.nbf())], dtype=int
    )


def _matrix_sqrt(s_ao: np.ndarray) -> np.ndarray:
    """Symmetric square root ``S^(1/2)`` by eigendecomposition.

    Near-linear-dependencies give tiny (occasionally slightly negative)
    eigenvalues; they are clamped at zero rather than allowed to produce NaNs.
    """
    evals, evecs = np.linalg.eigh(s_ao)
    return (evecs * np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T


def _accumulate_on_atoms(diagonal: np.ndarray, centres: np.ndarray, natom: int) -> np.ndarray:
    return np.bincount(centres, weights=diagonal, minlength=natom)


def atomic_spin_populations(wfn, mol=None) -> SpinPopulations:
    """Mulliken and Löwdin atomic spin populations from the AO spin density.

    The AO spin-density matrix is ``Ds = Da - Db``.  Its population analyses are

        Mulliken:  q_A = sum_{mu in A} (Ds @ S)_{mu,mu}
        Löwdin:    q_A = sum_{mu in A} (S^(1/2) @ Ds @ S^(1/2))_{mu,mu}

    Both are exact partitions of ``N_alpha - N_beta``; they differ only in how
    the overlap population is shared between the two centres of each AO pair.
    Mulliken hands the whole overlap term to whichever centre owns ``mu``, which
    is what makes it wobble with basis set; Löwdin splits it symmetrically.
    """
    mol = mol if mol is not None else wfn.molecule()
    natom = mol.natom()
    symbols = [mol.symbol(i).capitalize() for i in range(natom)]

    s_ao = _ao_overlap(wfn)
    centres = _ao_centres(wfn)
    da = np.asarray(wfn.Da())
    db = np.asarray(wfn.Db())
    spin_density = da - db
    total_density = da + db

    # Mulliken: diagonal of (Ds S).  einsum avoids forming the full product.
    mulliken_diag = np.einsum("ij,ji->i", spin_density, s_ao)
    mulliken = _accumulate_on_atoms(mulliken_diag, centres, natom)

    s_half = _matrix_sqrt(s_ao)
    lowdin_diag = np.einsum("ij,jk,ki->i", s_half, spin_density, s_half)
    lowdin = _accumulate_on_atoms(lowdin_diag, centres, natom)

    # Mulliken charges: Z_A minus the total (alpha + beta) gross population.
    charge_diag = np.einsum("ij,ji->i", total_density, s_ao)
    electrons = _accumulate_on_atoms(charge_diag, centres, natom)
    nuclear = np.array([mol.Z(i) for i in range(natom)], dtype=float)
    charges = nuclear - electrons

    return SpinPopulations(
        symbols=symbols, mulliken=mulliken, lowdin=lowdin, charges=charges
    )


# --------------------------------------------------------------------------
# The property hook
# --------------------------------------------------------------------------


def spin_properties(wfn, mol=None) -> dict[str, Any]:
    """Harvest the full spin picture from a converged wavefunction.

    Designed to be handed straight to
    :func:`nimrod.psi4_driver.run_energy` as ``property_hook=spin_properties``.
    Every value is a plain Python scalar or list so the result survives the
    JSON round-trip through the driver's on-disk cache; use
    :func:`populations_from_properties` to get the numpy arrays back.

    Keys
    ----
    ``s2``, ``s2_ideal``, ``spin_contamination``
        Computed, nominal and excess total spin.
    ``n_alpha``, ``n_beta``, ``sz``, ``multiplicity``
        Occupations of the determinant and the multiplicity that was requested.
    ``mulliken_spin``, ``lowdin_spin``, ``mulliken_charge``
        Per-atom lists, in input atom order.
    ``mulliken_spin_total``, ``lowdin_spin_total``
        Must equal ``n_alpha - n_beta``; a deviation means the AO/atom mapping
        or the density matrices are not what we think they are.
    ``max_spin_atom``, ``max_spin_value``
        Index and Mulliken spin of the most strongly spin-polarised atom.
    """
    mol = mol if mol is not None else wfn.molecule()

    s2 = spin_squared(wfn)
    multiplicity = int(mol.multiplicity())
    s2_ref = ideal_s2(multiplicity)

    pops = atomic_spin_populations(wfn, mol)
    n_alpha, n_beta = int(wfn.nalpha()), int(wfn.nbeta())

    max_index = int(np.argmax(np.abs(pops.mulliken))) if pops.natom else -1

    return {
        "s2": s2,
        "s2_ideal": s2_ref,
        "spin_contamination": s2 - s2_ref,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "sz": 0.5 * (n_alpha - n_beta),
        "multiplicity": multiplicity,
        "natom": pops.natom,
        "symbols": list(pops.symbols),
        "mulliken_spin": [float(x) for x in pops.mulliken],
        "lowdin_spin": [float(x) for x in pops.lowdin],
        "mulliken_charge": [float(x) for x in pops.charges],
        "mulliken_spin_total": float(pops.mulliken.sum()),
        "lowdin_spin_total": float(pops.lowdin.sum()),
        "max_spin_atom": max_index,
        "max_spin_value": float(pops.mulliken[max_index]) if pops.natom else 0.0,
    }


def populations_from_properties(properties: Mapping[str, Any]) -> SpinPopulations:
    """Rebuild a :class:`SpinPopulations` from a cached ``JobResult.properties``."""
    missing = {"symbols", "mulliken_spin", "lowdin_spin"} - set(properties)
    if missing:
        raise KeyError(f"properties dict is missing {sorted(missing)}")
    return SpinPopulations(
        symbols=list(properties["symbols"]),
        mulliken=np.asarray(properties["mulliken_spin"], dtype=float),
        lowdin=np.asarray(properties["lowdin_spin"], dtype=float),
        charges=np.asarray(properties.get("mulliken_charge", []), dtype=float),
    )


# --------------------------------------------------------------------------
# Fragment analysis
# --------------------------------------------------------------------------


def fragment_spin(
    populations: SpinPopulations | np.ndarray | Sequence[float],
    fragment_indices: Iterable[int],
    kind: str = "mulliken",
) -> float:
    """Total spin population carried by one set of atoms.

    ``populations`` may be a :class:`SpinPopulations` (in which case ``kind``
    selects the partition scheme) or a bare per-atom array.
    """
    values = (
        populations.get(kind)
        if isinstance(populations, SpinPopulations)
        else np.asarray(populations, dtype=float)
    )
    indices = list(fragment_indices)
    if not indices:
        return 0.0
    index_array = np.asarray(indices, dtype=int)
    if index_array.min() < 0 or index_array.max() >= values.size:
        raise IndexError(
            f"fragment indices {index_array.min()}..{index_array.max()} "
            f"outside 0..{values.size - 1}"
        )
    return float(values[index_array].sum())


def fragment_spins(
    populations: SpinPopulations,
    fragments: Mapping[str, Sequence[int]],
    kind: str = "mulliken",
) -> dict[str, float]:
    """:func:`fragment_spin` over a whole ``{name: indices}`` map."""
    return {
        name: fragment_spin(populations, indices, kind)
        for name, indices in fragments.items()
    }


def split_on_bond(structure, atom_a: int, atom_b: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition a :class:`nimrod.geometry.Structure` by cutting one bond.

    Returns ``(component_containing_a, component_containing_b)``.  Raises if the
    two atoms remain connected after the cut, which means the "bond" sits inside
    a ring and cutting it does not separate anything.
    """
    natom = len(structure)
    adjacency: list[set[int]] = [set() for _ in range(natom)]
    for i, j in structure.bonds():
        adjacency[i].add(j)
        adjacency[j].add(i)
    adjacency[atom_a].discard(atom_b)
    adjacency[atom_b].discard(atom_a)

    def flood(start: int) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen

    side_a = flood(atom_a)
    if atom_b in side_a:
        raise ValueError(
            f"atoms {atom_a} and {atom_b} are still connected after cutting their "
            "bond; the bond is part of a ring"
        )
    side_b = flood(atom_b)
    return tuple(sorted(side_a)), tuple(sorted(side_b))


def sensor_fragments(structure, info: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    """Split the assembled sensor into chemically meaningful fragments.

    ``structure`` and ``info`` are what :func:`nimrod.geometry.sensor_assembly`
    returns.  The partition is derived from the bonding graph rather than from
    atom ordering, so it survives any later renumbering.

    Fragments
    ---------
    ``catalyst``
        Everything on the nickel side of the meso-C / tether-C bond, metal
        included.  This is the "did a spin appear on the metal" fragment.
    ``colour_centre``
        Everything on the pyrene side, i.e. the phenylene bridge plus the
        pyrene bearing the sp3 defect.  This is the reporter.
    ``bridge``
        The 1,4-phenylene spacer alone, obtained by additionally cutting the
        sp3-C / ipso-C bond.  Spin here means the two centres are talking
        through the tether rather than through space.
    ``pyrene``
        The colour centre minus the bridge.
    ``metal``
        The nickel atom on its own.
    ``defect``
        The sp3 carbon and its hydrogen — the formal seat of the radical.
    """
    required = {"meso_carbon", "tether_carbon", "Ni", "sp3_carbon", "ipso_carbon"}
    missing = required - set(info)
    if missing:
        raise KeyError(f"sensor info map is missing {sorted(missing)}")

    catalyst, colour_centre = split_on_bond(
        structure, info["meso_carbon"], info["tether_carbon"]
    )
    if info["Ni"] not in catalyst:
        raise ValueError("bond cut did not put the nickel on the catalyst side")

    pyrene_side, bridge_side = split_on_bond(
        structure, info["sp3_carbon"], info["ipso_carbon"]
    )
    # Cutting sp3-C/ipso-C splits the *whole* assembly, so the bridge side still
    # carries the catalyst; intersect to keep only the phenylene.
    bridge = tuple(sorted(set(bridge_side) & set(colour_centre)))
    pyrene = tuple(sorted(set(pyrene_side) & set(colour_centre)))

    defect = (info["sp3_carbon"], *sorted(structure.hydrogens_on(info["sp3_carbon"])))

    return {
        "catalyst": catalyst,
        "colour_centre": colour_centre,
        "bridge": bridge,
        "pyrene": pyrene,
        "metal": (info["Ni"],),
        "defect": defect,
    }


# --------------------------------------------------------------------------
# Broken-symmetry DFT
# --------------------------------------------------------------------------


def unrestricted_reference(method: str) -> str:
    """``'uks'`` for a density functional, ``'uhf'`` for a wavefunction method.

    Both the high-spin and the broken-symmetry determinant must be
    *unrestricted*: the BS state is nominally a singlet but is only meaningful
    as a symmetry-broken UKS solution, and the HS state must be treated at the
    same level for the energy difference to mean anything.  The DFT/wavefunction
    decision is delegated to :meth:`JobSpec.resolved_reference` so this module
    never has to keep its own list of method names in sync.
    """
    probe = JobSpec(
        geometry="H 0.0 0.0 0.0", method=method, basis="sto-3g", multiplicity=3
    )
    return probe.resolved_reference()


def high_spin_multiplicity(spin_a: float = 0.5, spin_b: float = 0.5) -> int:
    """Multiplicity of the parallel-aligned state, ``2(S_A + S_B) + 1``."""
    total = spin_a + spin_b
    multiplicity = round(2 * total + 1)
    if abs(2 * total + 1 - multiplicity) > 1e-6 or multiplicity < 1:
        raise ValueError(f"local spins {spin_a}, {spin_b} give no integer multiplicity")
    return int(multiplicity)


def broken_symmetry_multiplicity(spin_a: float = 0.5, spin_b: float = 0.5) -> int:
    """Multiplicity of the Ms = |S_A - S_B| broken-symmetry companion.

    For two S = 1/2 centres this is 1: the BS determinant is run as a
    "singlet" with alpha density on one centre and beta density on the other.
    For the degraded sensor (S = 1/2 radical against an S = 1 nickel) it is 2.
    """
    difference = abs(spin_a - spin_b)
    multiplicity = round(2 * difference + 1)
    if abs(2 * difference + 1 - multiplicity) > 1e-6 or multiplicity < 1:
        raise ValueError(f"local spins {spin_a}, {spin_b} give no integer multiplicity")
    return int(multiplicity)


def _spin_spec(
    geometry: str,
    *,
    method: str,
    basis: str,
    multiplicity: int,
    charge: int,
    options: Mapping[str, Any] | None,
    label: str,
) -> JobSpec:
    return JobSpec(
        geometry=geometry,
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        reference=unrestricted_reference(method),
        options=dict(options or {}),
        label=label,
    )


def run_high_spin(
    geometry: str,
    *,
    method: str,
    basis: str,
    multiplicity: int = 3,
    charge: int = 0,
    options: Mapping[str, Any] | None = None,
    label: str = "high-spin",
    use_cache: bool = True,
) -> JobResult:
    """Converge the parallel-spin (high-spin) determinant.

    Forced unrestricted even though a restricted-open reference would converge
    more easily: the Yamaguchi formula needs the *actual* ``<S^2>`` of the same
    flavour of determinant used for the broken-symmetry state.
    """
    spec = _spin_spec(
        geometry,
        method=method,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        options=options,
        label=label,
    )
    return run_energy(spec, use_cache=use_cache, property_hook=spin_properties)


#: Guess strategies tried in turn when the broken-symmetry SCF keeps falling
#: back onto the closed-shell solution.  ``guess_mix`` rotates the alpha HOMO
#: and LUMO into each other, which is what actually breaks the spatial symmetry;
#: the different underlying guesses change which orbitals those are.
BS_GUESS_LADDER: tuple[dict[str, Any], ...] = (
    {"guess_mix": True},
    {"guess_mix": True, "guess": "gwh"},
    {"guess_mix": True, "guess": "core"},
)


def run_broken_symmetry(
    geometry: str,
    *,
    method: str,
    basis: str,
    multiplicity: int = 1,
    charge: int = 0,
    options: Mapping[str, Any] | None = None,
    label: str = "broken-symmetry",
    use_cache: bool = True,
    escalate: bool = True,
) -> JobResult:
    """Converge the broken-symmetry determinant with ``guess_mix``.

    The BS state is run at the Ms = |S_A - S_B| companion multiplicity of the
    high-spin state (1 for two S = 1/2 centres) on an unrestricted reference,
    with Psi4's ``guess_mix`` option mixing alpha HOMO and LUMO so the SCF has
    something asymmetric to relax from.

    A converged BS solution can still be a failure: if the SCF slides back into
    the closed-shell solution, ``<S^2>`` comes out at its nominal value and no
    symmetry was broken at all.  With ``escalate=True`` the harder guesses in
    :data:`BS_GUESS_LADDER` are tried in that case, and the lowest-energy
    converged attempt is returned regardless.  The outcome is recorded in
    ``result.properties['bs_collapsed']`` — callers must check it rather than
    assume, because for a genuinely closed-shell system (intact square-planar
    Ni(II), no second spin) collapse is the physically correct answer.
    """
    base_options = dict(options or {})
    ladder = BS_GUESS_LADDER if escalate else BS_GUESS_LADDER[:1]

    attempts: list[JobResult] = []
    for guess_options in ladder:
        spec = _spin_spec(
            geometry,
            method=method,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge,
            options={**guess_options, **base_options},
            label=label,
        )
        result = run_energy(spec, use_cache=use_cache, property_hook=spin_properties)
        if not result.ok:
            attempts.append(result)
            continue

        contamination = result.properties.get("spin_contamination")
        collapsed = contamination is None or contamination < BS_COLLAPSE_THRESHOLD
        result.properties["bs_collapsed"] = bool(collapsed)
        result.properties["bs_guess"] = ",".join(f"{k}={v}" for k, v in guess_options.items())
        attempts.append(result)
        if not collapsed:
            return result

    converged = [r for r in attempts if r.ok]
    if not converged:
        return attempts[-1]
    return min(converged, key=lambda r: r.energy)


def yamaguchi_j(e_hs: float, s2_hs: float, e_bs: float, s2_bs: float) -> float:
    """Yamaguchi exchange coupling constant, in cm^-1.

        J = (E_BS - E_HS) / (<S^2>_HS - <S^2>_BS)

    under the convention ``H = -2 J S1 . S2``, so that

        J > 0  ferromagnetic      (high-spin lower)
        J < 0  antiferromagnetic  (broken-symmetry lower)

    Dividing by the *computed* difference in spin contamination, rather than by
    an idealised value, is what makes this expression valid across the whole
    range from weakly-overlapping magnetic orbitals (where it reduces to the
    Noodleman/Ising form, denominator ~ 2 S_A S_B) to strongly-overlapping ones
    (where it reduces to the spin-projected form).  It is the reason no separate
    "weak" and "strong" coupling formulas are needed.

    Raises
    ------
    ValueError
        If the two determinants have effectively the same spin contamination.
        That happens when the BS SCF collapsed onto the high-spin solution or
        onto a closed-shell solution with no magnetic orbitals to break: the
        mapping onto a Heisenberg ladder is then undefined, and returning a
        huge number instead of complaining would be dishonest.
    """
    denominator = float(s2_hs) - float(s2_bs)
    if abs(denominator) < 1e-4:
        raise ValueError(
            "Yamaguchi denominator <S^2>_HS - <S^2>_BS = "
            f"{denominator:.3e} is degenerate; the broken-symmetry determinant "
            "carries the same spin contamination as the high-spin one, so J is "
            "not defined for this pair"
        )
    j_hartree = (float(e_bs) - float(e_hs)) / denominator
    return j_hartree * HARTREE_TO_CM


@dataclass
class ExchangeCoupling:
    """Result of one broken-symmetry exchange-coupling determination.

    ``converged`` means only that both SCFs converged and the Yamaguchi
    expression could be evaluated.  It does **not** mean the number is
    meaningful: always check ``bs_collapsed`` first.  A collapsed BS state turns
    the formula into "half the singlet-triplet gap of an ordinary closed-shell
    molecule", which for the intact square-planar catalyst is a large, real, and
    completely irrelevant number.  :meth:`summary` marks it, and the sensor
    pipeline should report such points as "no second spin" rather than as a
    coupling constant.
    """

    method: str
    basis: str
    multiplicity_hs: int
    multiplicity_bs: int
    e_hs: float | None = None
    e_bs: float | None = None
    s2_hs: float | None = None
    s2_bs: float | None = None
    j_cm: float | None = None
    j_kcal: float | None = None
    bs_collapsed: bool | None = None
    converged: bool = False
    error: str | None = None
    label: str = ""
    #: Per-atom Mulliken spin populations of each determinant, for fragment analysis.
    spin_hs: list[float] = field(default_factory=list)
    spin_bs: list[float] = field(default_factory=list)
    hs_result: JobResult | None = None
    bs_result: JobResult | None = None

    @property
    def coupling_sign(self) -> str:
        if self.j_cm is None:
            return "undetermined"
        if self.j_cm > 0:
            return "ferromagnetic"
        if self.j_cm < 0:
            return "antiferromagnetic"
        return "uncoupled"

    @property
    def delta_e_kcal(self) -> float | None:
        """``E_HS - E_BS`` in kcal/mol; positive means the BS state is lower."""
        if self.e_hs is None or self.e_bs is None:
            return None
        return (self.e_hs - self.e_bs) * HARTREE_TO_KCAL

    def summary(self) -> str:
        if not self.converged:
            return f"{self.label or 'J'}: FAILED ({self.error})"
        return (
            f"{self.label or 'J'}: J = {self.j_cm:+.2f} cm^-1 "
            f"({self.j_kcal:+.4f} kcal/mol, {self.coupling_sign}); "
            f"<S^2> HS {self.s2_hs:.4f} / BS {self.s2_bs:.4f}"
            + ("  [BS COLLAPSED]" if self.bs_collapsed else "")
        )


def exchange_coupling(
    geometry: str,
    *,
    method: str,
    basis: str,
    spin_a: float = 0.5,
    spin_b: float = 0.5,
    charge: int = 0,
    options: Mapping[str, Any] | None = None,
    label: str = "",
    use_cache: bool = True,
    escalate: bool = True,
) -> ExchangeCoupling:
    """Run the high-spin and broken-symmetry determinants and extract J.

    ``spin_a`` and ``spin_b`` are the *local* spins of the two magnetic centres.
    For the NiMROD sensor that is ``0.5`` for the colour-centre radical and
    either ``0.0`` (intact square-planar Ni(II), no coupling to measure) or
    ``1.0`` (three-coordinate Ni(II) after ligand loss).  They set the two
    multiplicities: HS is ``2(S_A + S_B) + 1`` and BS is ``2|S_A - S_B| + 1``.

    The returned object always carries whatever was obtained; a failed SCF or an
    undefined Yamaguchi denominator leaves ``j_cm`` as ``None`` and puts the
    reason in ``error`` rather than raising.
    """
    multiplicity_hs = high_spin_multiplicity(spin_a, spin_b)
    multiplicity_bs = broken_symmetry_multiplicity(spin_a, spin_b)
    if multiplicity_hs == multiplicity_bs:
        raise ValueError(
            f"local spins {spin_a} and {spin_b} give identical high-spin and "
            "broken-symmetry multiplicities; there is no second magnetic centre "
            "and no exchange coupling to compute"
        )

    coupling = ExchangeCoupling(
        method=method,
        basis=basis,
        multiplicity_hs=multiplicity_hs,
        multiplicity_bs=multiplicity_bs,
        label=label,
    )

    hs = run_high_spin(
        geometry,
        method=method,
        basis=basis,
        multiplicity=multiplicity_hs,
        charge=charge,
        options=options,
        label=f"{label}:HS" if label else "HS",
        use_cache=use_cache,
    )
    coupling.hs_result = hs
    if not hs.ok:
        coupling.error = f"high-spin SCF failed: {hs.error}"
        return coupling

    bs = run_broken_symmetry(
        geometry,
        method=method,
        basis=basis,
        multiplicity=multiplicity_bs,
        charge=charge,
        options=options,
        label=f"{label}:BS" if label else "BS",
        use_cache=use_cache,
        escalate=escalate,
    )
    coupling.bs_result = bs
    if not bs.ok:
        coupling.error = f"broken-symmetry SCF failed: {bs.error}"
        return coupling

    coupling.e_hs = hs.energy
    coupling.e_bs = bs.energy
    coupling.s2_hs = hs.properties.get("s2")
    coupling.s2_bs = bs.properties.get("s2")
    coupling.bs_collapsed = bs.properties.get("bs_collapsed")
    coupling.spin_hs = list(hs.properties.get("mulliken_spin", []))
    coupling.spin_bs = list(bs.properties.get("mulliken_spin", []))

    if coupling.s2_hs is None or coupling.s2_bs is None:
        coupling.error = "spin_properties hook did not return <S^2>"
        return coupling

    try:
        coupling.j_cm = yamaguchi_j(
            coupling.e_hs, coupling.s2_hs, coupling.e_bs, coupling.s2_bs
        )
    except ValueError as exc:
        coupling.error = str(exc)
        return coupling

    coupling.j_kcal = coupling.j_cm / HARTREE_TO_CM * HARTREE_TO_KCAL
    coupling.converged = True
    return coupling
