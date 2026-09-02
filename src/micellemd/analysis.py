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
