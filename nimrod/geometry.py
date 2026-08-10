"""Structure builders for the NiMROD sensor system.

The three chemical objects this project needs are built here from scratch, with
no dependency on RDKit or Open Babel:

``pyrene`` and friends
    Polycyclic aromatic hydrocarbons generated on an ideal graphene honeycomb
    lattice.  Hexagon centres are supplied on the triangular dual lattice and
    the carbon skeleton follows exactly; edge carbons are hydrogenated
    automatically.  This is verifiable — the builder must reproduce the known
    molecular formula of every named PAH, which :mod:`tests` asserts.

``occ_radical``
    The organic colour centre: a single aryl group covalently bonded to one
    pyrene CH carbon.  That carbon becomes sp3, breaking the local conjugation
    and creating both a localised exciton trap (the colour centre) and an
    unpaired electron (the S = 1/2 spin manifold).  Pyrene C16H10 plus phenyl
    C6H5 gives C22H15, which has an odd electron count and is therefore a
    ground-state doublet.

``ni_salen_model``
    A truncated Ni(II) salen analogue: bis(iminoenolato)nickel(II), a
    square-planar NiN2O2 complex.  It reproduces the electronic core of salen
    (two imine N and two phenolate-like O donors around a d8 centre, giving a
    closed-shell S = 0 ground state) at 19 atoms instead of 35, and unlike real
    salen its chelate arms can fully dissociate, which is what the degradation
    coordinate requires.

Geometries produced here are *starting guesses* with correct connectivity and
no atomic clashes.  They are meant to be relaxed by
:func:`nimrod.psi4_driver.run_optimize` before any production number is taken
from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Element data
# --------------------------------------------------------------------------

COVALENT_RADII = {  # angstrom, Cordero et al. 2008
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "S": 1.05, "Cl": 1.02, "Ni": 1.24, "Fe": 1.32, "Co": 1.26,
    "Cu": 1.32, "Zn": 1.22,
}

ATOMIC_NUMBER = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "S": 16, "Cl": 17, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
}

# Ideal bond lengths used for starting guesses (angstrom)
CC_AROMATIC = 1.42
CC_SINGLE = 1.51
CH_AROMATIC = 1.08
CH_SP3 = 1.09
NH_BOND = 1.02
TETRAHEDRAL_OUT_OF_PLANE = math.radians(54.735)


# --------------------------------------------------------------------------
# Structure container
# --------------------------------------------------------------------------


@dataclass
class Structure:
    """A molecular structure: element symbols plus Cartesian coordinates."""

    symbols: list[str]
    coords: np.ndarray  # (natom, 3) in angstrom
    name: str = ""

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        if len(self.symbols) != len(self.coords):
            raise ValueError(
                f"{len(self.symbols)} symbols but {len(self.coords)} coordinates"
            )

    # -- basic properties ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def n_electrons(self) -> int:
        return sum(ATOMIC_NUMBER[s] for s in self.symbols)

    @property
    def formula(self) -> str:
        """Hill-notation molecular formula, e.g. ``C22H15``."""
        counts: dict[str, int] = {}
        for s in self.symbols:
            counts[s] = counts.get(s, 0) + 1
        parts = []
        for el in ("C", "H"):
            if el in counts:
                n = counts.pop(el)
                parts.append(el if n == 1 else f"{el}{n}")
        for el in sorted(counts):
            n = counts[el]
            parts.append(el if n == 1 else f"{el}{n}")
        return "".join(parts)

    def multiplicity_for(self, charge: int = 0) -> int:
        """Lowest multiplicity consistent with the electron count."""
        n = self.n_electrons - charge
        return 1 if n % 2 == 0 else 2

    # -- geometry -----------------------------------------------------------

    def copy(self) -> "Structure":
        return Structure(list(self.symbols), self.coords.copy(), self.name)

    def translated(self, vec: Sequence[float]) -> "Structure":
        return Structure(list(self.symbols), self.coords + np.asarray(vec), self.name)

    def rotated(self, rotation: np.ndarray) -> "Structure":
        return Structure(list(self.symbols), self.coords @ np.asarray(rotation).T, self.name)

    def centred(self) -> "Structure":
        return self.translated(-self.coords.mean(axis=0))

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def neighbours(self, index: int, tolerance: float = 1.25) -> list[int]:
        """Indices bonded to ``index``, by covalent-radius overlap."""
        out = []
        ri = COVALENT_RADII[self.symbols[index]]
        for j, sym in enumerate(self.symbols):
            if j == index:
                continue
            cutoff = tolerance * (ri + COVALENT_RADII[sym])
            if self.distance(index, j) < cutoff:
                out.append(j)
        return out

    def bonds(self, tolerance: float = 1.25) -> list[tuple[int, int]]:
        out = []
        for i in range(len(self)):
            for j in self.neighbours(i, tolerance):
                if j > i:
                    out.append((i, j))
        return out

    def min_interatomic_distance(self) -> float:
        if len(self) < 2:
            return math.inf
        d = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min())

    def hydrogens_on(self, index: int) -> list[int]:
        return [j for j in self.neighbours(index) if self.symbols[j] == "H"]

    def without(self, indices: Iterable[int]) -> "Structure":
        drop = set(indices)
        keep = [i for i in range(len(self)) if i not in drop]
        return Structure(
            [self.symbols[i] for i in keep], self.coords[keep], self.name
        )

    # -- serialisation ------------------------------------------------------

    def to_psi4(self) -> str:
        """Bare Cartesian block (no charge/multiplicity line)."""
        return "\n".join(
            f"{s:<3s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}"
            for s, c in zip(self.symbols, self.coords)
        )

    def to_xyz(self, comment: str = "") -> str:
        return f"{len(self)}\n{comment or self.name}\n{self.to_psi4()}\n"

    @classmethod
    def from_psi4(cls, block: str, name: str = "") -> "Structure":
        symbols, coords = [], []
        for line in block.strip().splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            symbols.append(parts[0].capitalize())
            coords.append([float(p) for p in parts[1:]])
        return cls(symbols, np.array(coords), name)

    @classmethod
    def from_xyz(cls, text: str) -> "Structure":
        lines = text.strip().splitlines()
        n = int(lines[0].split()[0])
        return cls.from_psi4("\n".join(lines[2 : 2 + n]), name=lines[1].strip())


# --------------------------------------------------------------------------
# Vector helpers
# --------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalise a zero-length vector")
    return np.asarray(v, dtype=float) / n


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate pi about any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, perp))
        return -np.eye(3) + 2 * np.outer(axis, axis)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def plane_normal(points: np.ndarray) -> np.ndarray:
    """Best-fit plane normal of a point cloud (smallest-variance direction)."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    return _unit(vt[-1])


# --------------------------------------------------------------------------
# Polycyclic aromatic hydrocarbons on the honeycomb lattice
# --------------------------------------------------------------------------

_A_LATTICE = math.sqrt(3.0) * CC_AROMATIC          # hexagon-centre spacing
T1 = np.array([_A_LATTICE, 0.0])
T2 = np.array([_A_LATTICE / 2.0, 1.5 * CC_AROMATIC])

#: Hexagon-centre lattice coordinates (multiples of T1, T2) of named PAHs.
PAH_RINGS: dict[str, tuple[tuple[int, int], ...]] = {
    "benzene": ((0, 0),),
    "naphthalene": ((0, 0), (1, 0)),
    "anthracene": ((0, 0), (1, 0), (2, 0)),
    "phenanthrene": ((0, 0), (1, 0), (1, 1)),
    "pyrene": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "coronene": ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1)),
}


def build_pah(ring_centres: Sequence[tuple[int, int]], name: str = "pah") -> Structure:
    """Build a planar PAH from hexagon centres on the triangular dual lattice.

    Carbon vertices are the union of the six corners of every hexagon, merged
    within a tolerance.  Any carbon belonging to only one hexagon sits on the
    perimeter and receives a hydrogen along its outward radial direction.
    """
    centres = [np.asarray(i) * T1 + np.asarray(j) * T2 for i, j in ring_centres]

    # Hexagon corners: radius CC_AROMATIC at 30 deg + k*60 deg, matching a
    # centre lattice whose nearest neighbours share a vertical edge.
    corner_offsets = np.array(
        [
            [CC_AROMATIC * math.cos(math.radians(30 + 60 * k)),
             CC_AROMATIC * math.sin(math.radians(30 + 60 * k))]
            for k in range(6)
        ]
    )

    vertices: list[np.ndarray] = []
    membership: list[int] = []  # how many hexagons each vertex belongs to
    for centre in centres:
        for off in corner_offsets:
            p = centre + off
            for idx, existing in enumerate(vertices):
                if np.linalg.norm(existing - p) < 0.1:
                    membership[idx] += 1
                    break
            else:
                vertices.append(p)
                membership.append(1)

    ring_centroid = np.mean(centres, axis=0)
    symbols = ["C"] * len(vertices)
    coords = [np.array([v[0], v[1], 0.0]) for v in vertices]

    for idx, (v, count) in enumerate(zip(vertices, membership)):
        if count == 1:  # perimeter carbon -> add H
            direction = v - ring_centroid
            norm = np.linalg.norm(direction)
            # Perimeter carbons of a symmetric PAH can sit exactly on the
            # centroid ray; fall back to the vector from the owning hexagon.
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction = direction / norm
            h = v + CH_AROMATIC * direction
            symbols.append("H")
            coords.append(np.array([h[0], h[1], 0.0]))

    struct = Structure(symbols, np.array(coords), name)
    _refine_perimeter_hydrogens(struct)
    return struct


def _refine_perimeter_hydrogens(struct: Structure) -> None:
    """Point each aromatic C-H along the true outward bisector of its ring bonds.

    The radial placement used during construction is only approximate for
    non-convex PAHs; this corrects it using the actual carbon connectivity.
    """
    carbons = [i for i, s in enumerate(struct.symbols) if s == "C"]
    for i in carbons:
        hydrogens = [j for j in struct.neighbours(i) if struct.symbols[j] == "H"]
        if not hydrogens:
            continue
        c_neighbours = [j for j in struct.neighbours(i) if struct.symbols[j] == "C"]
        if len(c_neighbours) != 2:
            continue
        a, b = struct.coords[c_neighbours[0]], struct.coords[c_neighbours[1]]
        bisector = struct.coords[i] - 0.5 * (a + b)
        if np.linalg.norm(bisector) < 1e-6:
            continue
        struct.coords[hydrogens[0]] = struct.coords[i] + CH_AROMATIC * _unit(bisector)


def pyrene() -> Structure:
    """Pyrene, C16H10 — the colour-centre host."""
    return build_pah(PAH_RINGS["pyrene"], "pyrene")


def named_pah(name: str) -> Structure:
    if name not in PAH_RINGS:
        raise KeyError(f"unknown PAH {name!r}; have {sorted(PAH_RINGS)}")
    return build_pah(PAH_RINGS[name], name)


# --------------------------------------------------------------------------
# Aryl fragments
# --------------------------------------------------------------------------


def phenyl(para_substituted: bool = False) -> Structure:
    """A phenyl fragment with the ipso carbon first and its H second.

    The ipso C sits at the origin with the ring extending along +x, so the
    ipso->H vector points along -x and :func:`attach` can align it directly.

    If ``para_substituted`` is True the para hydrogen is retained but recorded
    last, so callers can use it as a second attachment point (this is how the
    colour centre is tethered to the catalyst).
    """
    coords = []
    symbols = []
    ring_radius = 1.39
    # Ring carbons at 60 degree steps; ipso placed so ring extends along +x.
    centre = np.array([ring_radius, 0.0, 0.0])
    ring = []
    for k in range(6):
        ang = math.radians(180 + 60 * k)
        ring.append(centre + ring_radius * np.array([math.cos(ang), math.sin(ang), 0.0]))

    # ring[0] is the ipso carbon (at the origin), ring[3] is para.
    symbols.append("C")
    coords.append(ring[0])
    # ipso hydrogen, pointing away from the ring along -x
    symbols.append("H")
    coords.append(ring[0] + np.array([-CH_AROMATIC, 0.0, 0.0]))

    para_index = None
    for k in range(1, 6):
        symbols.append("C")
        coords.append(ring[k])
    for k in range(1, 6):
        direction = _unit(ring[k] - centre)
        symbols.append("H")
        coords.append(ring[k] + CH_AROMATIC * direction)
        if k == 3:
            para_index = len(symbols) - 1

    struct = Structure(symbols, np.array(coords), "phenyl")
    struct.para_hydrogen = para_index  # type: ignore[attr-defined]
    return struct


# --------------------------------------------------------------------------
# Fragment joining
# --------------------------------------------------------------------------


def _remap_after_deletion(index: int, deleted: Sequence[int]) -> int:
    """Index of atom ``index`` after the atoms in ``deleted`` are removed."""
    if index in deleted:
        raise ValueError(f"atom {index} was deleted and has no remapped index")
    return index - sum(1 for d in deleted if d < index)


def attach(
    base: Structure,
    base_atom: int,
    base_hydrogen: int,
    fragment: Structure,
    fragment_atom: int,
    fragment_hydrogen: int,
    bond_length: float = CC_SINGLE,
    name: str = "",
    track_base: dict[str, int] | None = None,
    track_fragment: dict[str, int] | None = None,
) -> tuple[Structure, dict[str, int]]:
    """Join ``fragment`` to ``base``, replacing one hydrogen on each.

    The fragment is rigidly rotated so its ``fragment_atom -> fragment_hydrogen``
    bond points back along the ``base_atom -> base_hydrogen`` direction, then
    translated to the requested bond length.  Both hydrogens are deleted.

    ``track_base`` and ``track_fragment`` are optional ``{label: index}`` maps in
    the *original* numbering of each partner; they are remapped into the joined
    numbering and merged into the returned index map.  Doing the bookkeeping
    here rather than at the call site is what keeps composed structures
    (colour centre, then catalyst) correctly indexed.
    """
    direction = _unit(base.coords[base_hydrogen] - base.coords[base_atom])
    frag_dir = _unit(fragment.coords[fragment_hydrogen] - fragment.coords[fragment_atom])

    rot = rotation_between(frag_dir, -direction)
    moved = fragment.rotated(rot)
    offset = base.coords[base_atom] + bond_length * direction - moved.coords[fragment_atom]
    moved = moved.translated(offset)

    # Concatenate, then delete the two replaced hydrogens.
    symbols = list(base.symbols) + list(moved.symbols)
    coords = np.vstack([base.coords, moved.coords])
    combined = Structure(symbols, coords, name or f"{base.name}-{fragment.name}")

    n_base = len(base)
    deleted = sorted([base_hydrogen, n_base + fragment_hydrogen])

    index_map = {
        "base_atom": _remap_after_deletion(base_atom, deleted),
        "fragment_atom": _remap_after_deletion(n_base + fragment_atom, deleted),
    }
    for label, idx in (track_base or {}).items():
        index_map[label] = _remap_after_deletion(idx, deleted)
    for label, idx in (track_fragment or {}).items():
        index_map[label] = _remap_after_deletion(n_base + idx, deleted)

    return combined.without(deleted), index_map


# --------------------------------------------------------------------------
# The organic colour centre
# --------------------------------------------------------------------------


def _pyrene_ch_sites(struct: Structure) -> list[int]:
    """Carbons of a PAH that carry exactly one hydrogen (functionalisable)."""
    return [
        i
        for i, s in enumerate(struct.symbols)
        if s == "C" and len(struct.hydrogens_on(i)) == 1
    ]


def occ_radical(
    host: Structure | None = None,
    site: int | None = None,
    aryl: Structure | None = None,
    name: str = "occ-radical",
) -> tuple[Structure, dict[str, int]]:
    """Build the organic colour centre: an sp3 aryl defect on a PAH.

    A single aryl group is bonded to one PAH CH carbon.  That carbon keeps its
    hydrogen, so it becomes sp3 with four substituents (two ring carbons, one
    H, one aryl).  Removing one carbon from the aromatic pi system leaves an
    odd number of pi electrons: the product is a ground-state doublet radical
    whose spin density is localised around the defect, and whose broken
    conjugation creates the localised optical transition that makes it a
    colour centre.

    The sp3 carbon is pyramidalised into a tetrahedral starting geometry, with
    the H and the aryl group placed symmetrically above and below the former
    ring plane.
    """
    host = host or pyrene()
    aryl = aryl if aryl is not None else phenyl()

    sites = _pyrene_ch_sites(host)
    if not sites:
        raise ValueError("host PAH has no CH carbon to functionalise")
    if site is None:
        site = sites[0]
    elif site not in sites:
        raise ValueError(f"atom {site} is not a CH carbon; CH sites are {sites}")

    hydrogen = host.hydrogens_on(site)[0]
    ring_neighbours = [j for j in host.neighbours(site) if host.symbols[j] == "C"]
    if len(ring_neighbours) != 2:
        raise ValueError(f"atom {site} has {len(ring_neighbours)} ring neighbours, expected 2")

    carbons = np.array([host.coords[i] for i, s in enumerate(host.symbols) if s == "C"])
    normal = plane_normal(carbons)

    centre = host.coords[site]
    outward = _unit(centre - 0.5 * (host.coords[ring_neighbours[0]] + host.coords[ring_neighbours[1]]))

    # Tetrahedral directions for the two out-of-plane substituents.
    up = _unit(outward * math.cos(TETRAHEDRAL_OUT_OF_PLANE) + normal * math.sin(TETRAHEDRAL_OUT_OF_PLANE))
    down = _unit(outward * math.cos(TETRAHEDRAL_OUT_OF_PLANE) - normal * math.sin(TETRAHEDRAL_OUT_OF_PLANE))

    defect = host.copy()
    defect.name = name
    # Pyramidalise: lift the sp3 carbon slightly out of the ring plane and put
    # its hydrogen on the "up" tetrahedral vector.
    defect.coords[site] = centre + 0.08 * normal
    defect.coords[hydrogen] = defect.coords[site] + CH_SP3 * up

    # A temporary marker hydrogen along "down" gives attach() something to
    # replace with the aryl group.
    marker = Structure(
        defect.symbols + ["H"],
        np.vstack([defect.coords, defect.coords[site] + CH_SP3 * down]),
        name,
    )
    marker_index = len(marker) - 1

    joined, index_map = attach(
        marker,
        base_atom=site,
        base_hydrogen=marker_index,
        fragment=aryl,
        fragment_atom=0,
        fragment_hydrogen=1,
        bond_length=CC_SINGLE,
        name=name,
    )

    info = {
        "sp3_carbon": index_map["base_atom"],
        "ipso_carbon": index_map["fragment_atom"],
        "n_host_atoms": len(host),
    }
    return joined, info


# --------------------------------------------------------------------------
# The catalyst: bis(iminoenolato)nickel(II)
# --------------------------------------------------------------------------

NI_O_BOND = 1.84
NI_N_BOND = 1.87
BITE_HALF_ANGLE = math.radians(47.0)   # half the O-Ni-N bite angle
MESO_RADIUS = 2.95                     # Ni to backbone meso carbon


def _chelate_arm(
    ni: np.ndarray,
    bisector_angle: float,
) -> tuple[list[str], list[np.ndarray], dict[str, int]]:
    """One planar iminoenolate arm: Ni-O-Ca-Cb-Cc-N six-membered chelate."""

    def polar(radius: float, angle: float) -> np.ndarray:
        return ni + np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])

    o_pos = polar(NI_O_BOND, bisector_angle - BITE_HALF_ANGLE)
    n_pos = polar(NI_N_BOND, bisector_angle + BITE_HALF_ANGLE)
    cb_pos = polar(MESO_RADIUS, bisector_angle)  # meso carbon on the bisector

    ca_pos = _circle_intersection(o_pos, 1.30, cb_pos, 1.39, ni)
    cc_pos = _circle_intersection(n_pos, 1.31, cb_pos, 1.39, ni)

    symbols = ["O", "C", "C", "C", "N"]
    coords = [o_pos, ca_pos, cb_pos, cc_pos, n_pos]

    # Outward-pointing hydrogens on Ca, Cb, Cc and on N.
    def outward_h(pos: np.ndarray, neigh_a: np.ndarray, neigh_b: np.ndarray, length: float) -> np.ndarray:
        return pos + length * _unit(pos - 0.5 * (neigh_a + neigh_b))

    symbols.append("H")
    coords.append(outward_h(ca_pos, o_pos, cb_pos, CH_AROMATIC))
    symbols.append("H")
    coords.append(outward_h(cb_pos, ca_pos, cc_pos, CH_AROMATIC))
    symbols.append("H")
    coords.append(outward_h(cc_pos, n_pos, cb_pos, CH_AROMATIC))
    symbols.append("H")
    coords.append(outward_h(n_pos, ni, cc_pos, NH_BOND))

    local = {"O": 0, "Ca": 1, "Cb": 2, "Cc": 3, "N": 4, "Hca": 5, "Hcb": 6, "Hcc": 7, "Hn": 8}
    return symbols, coords, local


def _circle_intersection(
    c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, away_from: np.ndarray
) -> np.ndarray:
    """In-plane (z=0) intersection of two circles, taking the solution farther
    from ``away_from``.  Used to close the chelate ring geometrically."""
    p1, p2 = c1[:2], c2[:2]
    d_vec = p2 - p1
    d = float(np.linalg.norm(d_vec))
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-9:
        raise ValueError(
            f"cannot close chelate ring: circles r={r1}, r={r2} at separation {d:.3f}"
        )
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h_sq = r1 * r1 - a * a
    h = math.sqrt(max(h_sq, 0.0))
    base = p1 + a * d_vec / d
    perp = np.array([-d_vec[1], d_vec[0]]) / d
    candidates = [base + h * perp, base - h * perp]
    ref = away_from[:2]
    best = max(candidates, key=lambda p: np.linalg.norm(p - ref))
    return np.array([best[0], best[1], 0.0])


def ni_salen_model(name: str = "ni-salen-model") -> tuple[Structure, dict[str, int]]:
    """Square-planar bis(iminoenolato)nickel(II), the truncated salen catalyst.

    Two bidentate [NH=CH-CH=CH-O]- arms chelate a Ni(II) centre through one N
    and one O each, giving a neutral NiN2O2 complex in the *trans* (centro-
    symmetric) arrangement: N trans to N, O trans to O.  The d8 square-planar
    field gives a closed-shell S = 0 ground state, and elongating one Ni-N bond
    opens a coordination site and drives the complex towards S = 1 — the
    degradation coordinate.
    """
    ni = np.zeros(3)
    symbols = ["Ni"]
    coords = [ni]
    index: dict[str, int] = {"Ni": 0}

    for arm_id, bisector in enumerate((0.0, math.pi)):
        arm_symbols, arm_coords, local = _chelate_arm(ni, bisector)
        offset = len(symbols)
        symbols.extend(arm_symbols)
        coords.extend(arm_coords)
        for key, value in local.items():
            index[f"{key}{arm_id + 1}"] = offset + value

    return Structure(symbols, np.array(coords), name), index


# --------------------------------------------------------------------------
# The assembled sensor: colour centre tethered to the catalyst
# --------------------------------------------------------------------------


def sensor_assembly(name: str = "occ-ni-sensor") -> tuple[Structure, dict[str, int]]:
    """The full construct: pyrene colour centre --phenylene-- Ni catalyst.

    The colour centre's aryl group is a 1,4-phenylene: one end carries the sp3
    defect on pyrene, the other bonds to the meso carbon of a chelate arm.  The
    catalyst therefore sits one rigid, conjugation-limiting spacer away from the
    unpaired spin, which is what makes the two readouts (optical shift and
    exchange coupling) respond to the metal's coordination environment without
    the defect simply being absorbed into the metal complex.
    """
    occ, occ_info = occ_radical()
    catalyst, cat_index = ni_salen_model()

    # Locate the para hydrogen of the tethering phenylene in the OCC numbering.
    ipso = occ_info["ipso_carbon"]
    para_carbon, para_hydrogen = _para_position(occ, ipso, occ_info["sp3_carbon"])

    meso = cat_index["Cb1"]
    meso_h = cat_index["Hcb1"]

    # The labile arm is the one *not* carrying the tether, so that ligand loss
    # and the reporting colour centre sit on opposite sides of the metal and
    # the signal is transmitted through the metal rather than through bond.
    joined, index_map = attach(
        base=catalyst,
        base_atom=meso,
        base_hydrogen=meso_h,
        fragment=occ,
        fragment_atom=para_carbon,
        fragment_hydrogen=para_hydrogen,
        bond_length=CC_SINGLE,
        name=name,
        track_base={
            "Ni": cat_index["Ni"],
            "N_labile": cat_index["N2"],
            "N_spectator": cat_index["N1"],
            "O_labile": cat_index["O2"],
            "O_spectator": cat_index["O1"],
            "H_labile_N": cat_index["Hn2"],
        },
        track_fragment={
            "sp3_carbon": occ_info["sp3_carbon"],
            "ipso_carbon": occ_info["ipso_carbon"],
        },
    )

    info = dict(index_map)
    info["meso_carbon"] = index_map.pop("base_atom")
    info["tether_carbon"] = index_map.pop("fragment_atom")
    info.pop("base_atom", None)
    info.pop("fragment_atom", None)
    info["n_catalyst_atoms"] = len(catalyst) - 1
    return joined, info


def _para_position(struct: Structure, ipso: int, exclude: int) -> tuple[int, int]:
    """Find the carbon para to ``ipso`` in its benzene ring, and its hydrogen."""
    # Breadth-first through carbon-only bonds, staying off the pyrene side.
    ring_atoms = [j for j in struct.neighbours(ipso) if struct.symbols[j] == "C" and j != exclude]
    seen = {ipso, exclude}
    frontier = list(ring_atoms)
    distance = {a: 1 for a in ring_atoms}
    seen.update(ring_atoms)
    while frontier:
        current = frontier.pop(0)
        if distance[current] == 3:
            hydrogens = struct.hydrogens_on(current)
            if hydrogens:
                return current, hydrogens[0]
        for nxt in struct.neighbours(current):
            if struct.symbols[nxt] == "C" and nxt not in seen:
                seen.add(nxt)
                distance[nxt] = distance[current] + 1
                frontier.append(nxt)
    raise ValueError("could not locate a para C-H on the tethering aryl ring")


# --------------------------------------------------------------------------
# Degradation coordinate
# --------------------------------------------------------------------------


def elongate_bond(
    struct: Structure,
    anchor: int,
    moving: int,
    target_distance: float,
    carry: Sequence[int] = (),
) -> Structure:
    """Displace ``moving`` (and any atoms in ``carry``) along the anchor->moving
    axis so that the anchor-moving distance becomes ``target_distance``.

    This generates the *starting* geometry for each point of a relaxed scan;
    the remaining degrees of freedom are then optimised with the anchor-moving
    distance frozen.
    """
    direction = _unit(struct.coords[moving] - struct.coords[anchor])
    current = struct.distance(anchor, moving)
    shift = (target_distance - current) * direction
    out = struct.copy()
    for index in {moving, *carry}:
        out.coords[index] = out.coords[index] + shift
    return out


def chelate_hinge(
    struct: Structure, anchor: int, donor: int
) -> tuple[int, np.ndarray, list[int]]:
    """Find the hinge for swinging a chelate arm off the metal.

    Returns ``(pivot, axis, moving)``: the atom the arm rotates about, the
    rotation axis, and the atoms that travel with the donor.

    The chelate ring here is Ni-O-Ca-Cb-Cc-N.  Walking from the donor nitrogen
    back around the ring (with the direct Ni-N bond removed from the graph)
    gives that path, and the hinge is placed at the carbon bearing the oxygen —
    the far side of the ring — so the imine end swings out on the longest
    available lever while the phenolate end stays coordinated.  That is what
    actually happens when a salen arm decoordinates.
    """
    # Shortest path from donor back to the metal, not using the direct bond.
    previous: dict[int, int] = {donor: -1}
    frontier = [donor]
    while frontier:
        current = frontier.pop(0)
        if current == anchor:
            break
        for nxt in struct.neighbours(current):
            if nxt in previous:
                continue
            if current == donor and nxt == anchor:
                continue          # forbid the direct donor-metal bond
            previous[nxt] = current
            frontier.append(nxt)
    if anchor not in previous:
        raise ValueError("donor and metal are not part of a chelate ring")

    path = []
    node = anchor
    while node != -1:
        path.append(node)
        node = previous[node]
    path.reverse()               # donor ... anchor

    if len(path) < 5:
        raise ValueError(f"chelate ring too small to hinge: path {path}")
    pivot = path[3]              # donor, Cc, Cb, Ca  ->  hinge at Ca

    # Atoms that move: everything reachable from the donor without crossing the
    # metal or the pivot.
    blocked = {anchor, pivot}
    moving: list[int] = []
    seen = set(blocked)
    stack = [donor]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        moving.append(current)
        stack.extend(n for n in struct.neighbours(current) if n not in seen)

    # Hinge about the normal to the chelate plane, so the arm swings within it.
    plane = np.array([struct.coords[anchor], struct.coords[pivot], struct.coords[donor]])
    axis = plane_normal(plane)
    return pivot, axis, sorted(moving)


def _rotate_about(
    coords: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rodrigues rotation of ``coords`` about ``axis`` through ``origin``."""
    k = _unit(axis)
    shifted = coords - origin
    return (
        shifted * math.cos(angle)
        + np.cross(k, shifted) * math.sin(angle)
        + np.outer(np.dot(shifted, k), k) * (1.0 - math.cos(angle))
    ) + origin


def swing_arm(
    struct: Structure,
    anchor: int,
    donor: int,
    target_distance: float,
    *,
    hinge: tuple[int, np.ndarray, list[int]] | None = None,
    tolerance: float = 1e-6,
) -> Structure:
    """Open a chelate arm by rigid rotation until ``anchor``-``donor`` matches.

    Unlike a translation of the donor alone, this preserves every internal bond
    length and angle in the moving fragment exactly — it is a rigid-body
    rotation.  Translating the nitrogen along the metal-nitrogen axis instead,
    which is the obvious thing to do, drives the imine N=C bond down to 0.99 A
    at intermediate separations because the nitrogen is constrained inside a
    ring; those geometries are not chemistry.
    """
    pivot, axis, moving = hinge or chelate_hinge(struct, anchor, donor)
    origin = struct.coords[pivot]
    anchor_pos = struct.coords[anchor]
    donor_local = struct.coords[donor]

    def distance_at(angle: float) -> float:
        rotated = _rotate_about(donor_local[None, :], origin, axis, angle)[0]
        return float(np.linalg.norm(rotated - anchor_pos))

    # The reachable range is |anchor-pivot| +/- |pivot-donor|.
    reach_hi = np.linalg.norm(origin - anchor_pos) + np.linalg.norm(donor_local - origin)
    if target_distance > reach_hi + 1e-9:
        raise ValueError(
            f"cannot reach {target_distance:.2f} A by hinging; the arm only "
            f"extends to {reach_hi:.2f} A about this pivot"
        )

    # A given separation is reachable at several angles, in both rotation
    # senses.  They are *not* equivalent: one sense swings the arm out into
    # free space, the other sweeps it straight through the trans ligand and
    # produces 0.1 A contacts.  So collect every root and choose by the
    # resulting closest contact rather than taking the first bracket found.
    grid = [math.radians(step) for step in range(-180, 181)]
    values = [distance_at(a) - target_distance for a in grid]

    roots: list[float] = []
    for (a0, v0), (a1, v1) in zip(zip(grid, values), zip(grid[1:], values[1:])):
        if v0 == 0.0:
            roots.append(a0)
        elif v0 * v1 < 0:
            lo, hi = a0, a1
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if (distance_at(lo) - target_distance) * (distance_at(mid) - target_distance) <= 0:
                    hi = mid
                else:
                    lo = mid
                if abs(hi - lo) < tolerance:
                    break
            roots.append(0.5 * (lo + hi))

    if not roots:
        raise ValueError(
            f"no hinge angle reaches anchor-donor = {target_distance:.2f} A "
            f"(reachable up to {reach_hi:.2f} A)"
        )

    best_struct: Structure | None = None
    best_contact = -math.inf
    for angle in roots:
        candidate = struct.copy()
        candidate.coords[moving] = _rotate_about(struct.coords[moving], origin, axis, angle)
        contact = candidate.min_interatomic_distance()
        if contact > best_contact:
            best_contact, best_struct = contact, candidate

    assert best_struct is not None
    return best_struct


def degradation_series(
    struct: Structure,
    ni_index: int,
    n_index: int,
    distances: Sequence[float] = (1.87, 2.10, 2.35, 2.60, 2.90, 3.30, 3.80, 4.50),
    carry: Sequence[int] = (),
) -> list[tuple[float, Structure]]:
    """Starting geometries along the ligand-loss degradation coordinate.

    ``distances`` are Ni-N separations in angstrom, from the intact
    square-planar complex out to a fully dissociated imine arm.
    """
    return [
        (d, elongate_bond(struct, ni_index, n_index, d, carry=carry))
        for d in distances
    ]


def arm_atoms(struct: Structure, ni_index: int, n_index: int) -> list[int]:
    """Atoms displaced rigidly with the dissociating imine nitrogen.

    Only the nitrogen and the hydrogens bonded to it are carried.  The rest of
    the chelate arm stays where it is and is relaxed by the constrained
    optimiser at each scan point, which is what makes the scan a *relaxed*
    surface rather than a rigid pull.
    """
    carried = [n_index]
    carried.extend(
        j for j in struct.neighbours(n_index) if struct.symbols[j] == "H" and j != ni_index
    )
    return carried
