"""Trajectory analysis: aggregation clustering and radius of gyration.

Deliberately independent of whether the underlying force-field parameters
are the real MARTINI values or the placeholders currently in cg_model.py --
these functions operate on whatever bead positions they're given, so they
can be built and correctly tested now, and will work unchanged once real
MARTINI parameters are substituted in. This is the actual payload MicelleMD
exists to produce (aggregation number, shape), so it's worth having ready
before the force-field question is resolved, not blocked behind it.
"""

from __future__ import annotations
import math


def radius_of_gyration(positions: list[tuple[float, float, float]]) -> float:
    """Standard Rg formula: sqrt(mean squared distance from the
    center of mass). Equal bead masses assumed here (true for the
    placeholder model in cg_model.py); pass mass-weighted positions
    upstream if that assumption ever changes."""
    n = len(positions)
    if n == 0:
        return 0.0
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n
    mean_sq_dist = sum((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 for p in positions) / n
    return math.sqrt(mean_sq_dist)


def gyration_tensor_eigenvalues(positions: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Eigenvalues (descending: L1 >= L2 >= L3) of the gyration tensor
    S = (1/N) * sum_i (r_i - r_com) (x) (r_i - r_com), equal bead masses
    assumed (same convention as radius_of_gyration above). Rg^2 = L1+L2+L3
    exactly, by construction -- used as a self-consistency check in tests.

    Solved analytically (trigonometric solution for a real symmetric 3x3
    matrix's eigenvalues via its characteristic polynomial), not via an
    iterative numerical method -- exact for this closed-form case and
    keeps this module numpy-free, matching its existing convention."""
    n = len(positions)
    if n == 0:
        return (0.0, 0.0, 0.0)
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n
    sxx = sum((p[0]-cx)**2 for p in positions) / n
    syy = sum((p[1]-cy)**2 for p in positions) / n
    szz = sum((p[2]-cz)**2 for p in positions) / n
    sxy = sum((p[0]-cx)*(p[1]-cy) for p in positions) / n
    sxz = sum((p[0]-cx)*(p[2]-cz) for p in positions) / n
    syz = sum((p[1]-cy)*(p[2]-cz) for p in positions) / n

    # Standard closed-form eigenvalues of a real symmetric 3x3 matrix
    # (Smith 1961 / the common "trigonometric" method): shift by the mean
    # of the diagonal so the shifted matrix is traceless, then solve via
    # the invariants p, q of the characteristic polynomial.
    trace = sxx + syy + szz
    m = trace / 3.0
    pxx, pyy, pzz = sxx - m, syy - m, szz - m
    p = (pxx**2 + pyy**2 + pzz**2 + 2 * (sxy**2 + sxz**2 + syz**2)) / 6.0
    if p < 1e-14:  # all beads coincide (or n==1): tensor is exactly zero
        return (0.0, 0.0, 0.0)
    p_sqrt = math.sqrt(p)
    # determinant of (S - m*I) / p_sqrt, clamped for numerical safety
    b11, b22, b33 = pxx / p_sqrt, pyy / p_sqrt, pzz / p_sqrt
    b12, b13, b23 = sxy / p_sqrt, sxz / p_sqrt, syz / p_sqrt
    det_b = (b11 * (b22 * b33 - b23 * b23)
             - b12 * (b12 * b33 - b23 * b13)
             + b13 * (b12 * b23 - b22 * b13))
    r = max(-1.0, min(1.0, det_b / 2.0))
    phi = math.acos(r) / 3.0
    l1 = m + 2 * p_sqrt * math.cos(phi)
    l3 = m + 2 * p_sqrt * math.cos(phi + 2 * math.pi / 3.0)
    l2 = trace - l1 - l3  # eigenvalues sum to the trace exactly
    eigs = sorted([l1, l2, l3], reverse=True)
    return (max(0.0, eigs[0]), max(0.0, eigs[1]), max(0.0, eigs[2]))


def shape_descriptors(positions: list[tuple[float, float, float]]) -> dict:
    """Micelle shape from the gyration tensor eigenvalues: asphericity b,
    acylindricity c, and relative shape anisotropy kappa^2 (the standard
    normalized 0-1 metric from polymer/micelle physics -- 0 for a perfect
    sphere, 1 for a rigid rod; see Rudnick & Gaspari 1986, Theodorou &
    Suter 1985). Reporting kappa^2 rather than a bare asphericity number
    is what makes this comparable across aggregates of different sizes."""
    l1, l2, l3 = gyration_tensor_eigenvalues(positions)
    rg2 = l1 + l2 + l3
    if rg2 < 1e-14:
        return {"asphericity": 0.0, "acylindricity": 0.0, "relative_shape_anisotropy": 0.0}
    b = l1 - 0.5 * (l2 + l3)
    c = l2 - l3
    kappa2 = (b**2 + 0.75 * c**2) / rg2**2
    return {"asphericity": b, "acylindricity": c, "relative_shape_anisotropy": kappa2}


def _fibonacci_sphere(n_points: int) -> list[tuple[float, float, float]]:
    """n_points roughly evenly distributed on a unit sphere -- the standard
    low-discrepancy construction used for numerical Shrake-Rupley SASA
    (avoids the clustering artifacts of e.g. naive lat/long grids)."""
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - 2.0 * i / (n_points - 1) if n_points > 1 else 0.0
        radius_at_y = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        points.append((math.cos(theta) * radius_at_y, y, math.sin(theta) * radius_at_y))
    return points


def solvent_accessible_surface_area(positions: list[tuple[float, float, float]],
                                    bead_radius_nm: float = 0.235, probe_radius_nm: float = 0.21,
                                    n_sphere_points: int = 100) -> float:
    """Solvent-accessible surface area (nm^2) via the Shrake-Rupley
    algorithm: for each bead, sample points on a sphere of radius
    (bead_radius + probe_radius) centered on it, and count the fraction
    NOT buried inside any other bead's same expanded sphere -- real
    numerical SASA, not a bounding-box or convex-hull approximation.

    Defaults are real MARTINI conventions, not arbitrary: bead_radius_nm
    = 0.235 nm is half of this project's own MARTINI 3 bead sigma
    (0.470 nm, see cg_model.py's MARTINI3_BEAD_TYPES -- the same sigma
    used for every bead type here); probe_radius_nm = 0.21 nm is the
    standard MARTINI water-bead radius convention (half its own 0.470 nm
    sigma is 0.235, but the literature-standard MARTINI probe/solvent
    radius commonly cited for SASA-style calculations is ~0.21 nm,
    reflecting water's smaller effective probe size versus a full CG
    bead -- both are exposed as parameters precisely so a caller isn't
    stuck with an assumption baked in silently)."""
    n = len(positions)
    if n == 0:
        return 0.0
    expanded_r = bead_radius_nm + probe_radius_nm
    sphere = _fibonacci_sphere(n_sphere_points)
    total_area = 0.0
    for i in range(n):
        exposed = 0
        for sx, sy, sz in sphere:
            test_point = (positions[i][0] + expanded_r * sx,
                         positions[i][1] + expanded_r * sy,
                         positions[i][2] + expanded_r * sz)
            buried = False
            for j in range(n):
                if j == i:
                    continue
                d2 = ((test_point[0] - positions[j][0])**2 + (test_point[1] - positions[j][1])**2
                      + (test_point[2] - positions[j][2])**2)
                if d2 < expanded_r**2:
                    buried = True
                    break
            if not buried:
                exposed += 1
        sphere_area = 4.0 * math.pi * expanded_r**2
        total_area += sphere_area * exposed / n_sphere_points
    return total_area


def find_aggregates(molecule_positions: list[list[tuple[float, float, float]]],
                    cutoff_nm: float = 0.6, box_size_nm: float | None = None) -> list[list[int]]:
    """Cluster molecules into aggregates by inter-molecule proximity: two
    molecules are in the same aggregate if ANY bead pair between them is
    within cutoff_nm. Simple connected-components over a proximity graph.

    box_size_nm: pass this for a real periodic production run (e.g. from
    build_solvated_openmm_system()) -- applies minimum-image convention
    (cubic box assumed, matching build_openmm_system()/
    build_solvated_openmm_system()'s own cubic box construction) so
    molecules near opposite box faces that are actually close via PBC are
    correctly identified as neighbors. Left as None (raw Euclidean
    distance, no wrapping) by default for backward compatibility with the
    existing non-periodic toy-system callers/tests -- same "not periodic-
    boundary-aware unless asked" limitation this function already had,
    just now an explicit opt-in rather than an unconditional gap.
    """
    n_mol = len(molecule_positions)
    adjacency = {i: set() for i in range(n_mol)}

    def min_dist(mol_a, mol_b):
        best = float("inf")
        for pa in mol_a:
            for pb in mol_b:
                if box_size_nm is not None:
                    d = math.sqrt(sum(
                        (((pa[k] - pb[k] + box_size_nm / 2) % box_size_nm) - box_size_nm / 2) ** 2
                        for k in range(3)))
                else:
                    d = math.sqrt(sum((pa[k]-pb[k])**2 for k in range(3)))
                if d < best:
                    best = d
        return best

    for i in range(n_mol):
        for j in range(i + 1, n_mol):
            if min_dist(molecule_positions[i], molecule_positions[j]) <= cutoff_nm:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    aggregates = []
    for start in range(n_mol):
        if start in visited:
            continue
        stack, cluster = [start], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            stack.extend(adjacency[node] - visited)
        aggregates.append(sorted(cluster))
    return aggregates


def aggregation_number_distribution(aggregates: list[list[int]]) -> dict[int, int]:
    """Histogram: aggregation size -> count of aggregates of that size.
    E.g. {1: 5, 3: 2} means 5 free monomers and 2 trimers."""
    from collections import Counter
    return dict(Counter(len(a) for a in aggregates))


if __name__ == "__main__":
    # self-test: two tight clusters of molecules plus one isolated monomer,
    # confirm find_aggregates recovers exactly that structure
    cluster_a = [[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], [(0.2, 0.0, 0.0), (0.3, 0.0, 0.0)]]
    cluster_b = [[(10.0, 0.0, 0.0), (10.1, 0.0, 0.0)], [(10.2, 0.0, 0.0), (10.3, 0.0, 0.0)],
                 [(10.4, 0.0, 0.0), (10.5, 0.0, 0.0)]]
    monomer = [[(20.0, 0.0, 0.0), (20.1, 0.0, 0.0)]]
    mols = cluster_a + cluster_b + monomer

    aggs = find_aggregates(mols, cutoff_nm=0.6)
    dist = aggregation_number_distribution(aggs)
    print(f"Found {len(aggs)} aggregates: {aggs}")
    print(f"Aggregation number distribution: {dist}")
    n_size2 = sum(1 for a in aggs if len(a) == 2)
    n_size3 = sum(1 for a in aggs if len(a) == 3)
    n_size1 = sum(1 for a in aggs if len(a) == 1)
    ok = (n_size2 == 1 and n_size3 == 1 and n_size1 == 1)
    print(f"Self-test (expect one size-2, one size-3, one size-1 aggregate): {'PASSED' if ok else 'FAILED'}")

    rg = radius_of_gyration(cluster_a[0] + cluster_a[1])
    print(f"\nRadius of gyration of cluster_a's 4 beads: {rg:.4f} nm")
