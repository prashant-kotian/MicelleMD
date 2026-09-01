"""Aggregation analysis for real atomistic GROMACS trajectories, via
MDAnalysis (conda-forge, a standard, well-established MD analysis library --
not hand-rolled .xtc/.tpr parsing).

Distinct from analysis.py's find_aggregates(), which operates on raw
in-memory bead positions and is explicitly NOT periodic-boundary-aware (fine
for the small toy/CG cases it was built for). A real atomistic self-assembly
run lives in a periodic box (see GROMACS_SETUP.md's 20x SDS run, 6x6x6 nm),
so molecules near a box edge can legitimately be close to a neighbour across
the periodic boundary -- ignoring that would silently undercount real
aggregates. This module uses MDAnalysis's PBC-aware minimum-image distance
calculation (`distance_array(..., box=...)`) instead.

Aggregation criterion is atom-level minimum distance (any atom of molecule A
within cutoff of any atom of molecule B), not center-of-mass distance --
COM-to-COM distance is a poor proxy for "in contact" for extended chain
molecules like a C12 alkyl sulfate (COM sits mid-chain, so two aggregated
molecules' COMs can be well over a nm apart even with their tails
interdigitating). Default cutoff (0.5 nm) matches the typical first-RDF-
minimum contact distance used in the surfactant-MD literature for defining
aggregation, not an arbitrarily chosen number.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import aggregation_number_distribution, radius_of_gyration


def unwrap_cluster_positions(positions_list: list[np.ndarray], box: np.ndarray) -> list[np.ndarray]:
    """Make a cluster whole across periodic boundaries before computing any
    real-space quantity (like Rg) on it.

    find_aggregates_pbc correctly identifies contacts via minimum-image
    distance, but the raw atom coordinates it operates on stay box-wrapped --
    two molecules can be genuine neighbours across a periodic boundary (e.g.
    one at x=0.1 nm, the other at x=5.9 nm in a 6 nm box) while their RAW
    Cartesian positions are ~5.8 nm apart. Computing Rg directly on those raw
    positions silently produces a physically wrong, inflated value. Fixed by
    shifting every molecule's positions by the periodic image that minimizes
    its distance to a fixed reference point (the first molecule's first
    atom), an orthorhombic-box minimum-image unwrap -- valid here since
    GROMACS's `editconf -c` box is cubic (dimensions[3:6] == 90 degrees).
    """
    box_lengths = np.asarray(box[:3])
    ref = positions_list[0][0]
    unwrapped = []
    for positions in positions_list:
        diff = positions - ref
        diff -= box_lengths * np.round(diff / box_lengths)
        unwrapped.append(ref + diff)
    return unwrapped


def find_aggregates_pbc(molecule_atom_groups: list, box: np.ndarray, cutoff_nm: float = 0.5) -> list[list[int]]:
    """Same connected-components logic as analysis.find_aggregates(), but
    atom-level (not bead-level) and periodic-boundary-aware via MDAnalysis's
    minimum-image distance_array.

    molecule_atom_groups: one MDAnalysis AtomGroup per molecule (its atoms'
    positions for the current frame).
    box: the frame's box dimensions (ts.dimensions), passed straight to
    distance_array for correct minimum-image handling.
    """
    cutoff_A = cutoff_nm * 10.0  # MDAnalysis coordinates are in Angstrom
    n_mol = len(molecule_atom_groups)
    adjacency = {i: set() for i in range(n_mol)}

    for i in range(n_mol):
        for j in range(i + 1, n_mol):
            dmat = distance_array(molecule_atom_groups[i].positions,
                                  molecule_atom_groups[j].positions, box=box)
            if dmat.min() <= cutoff_A:
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


def analyze_trajectory(topology_path: str, xtc_path: str, resname: str,
                       cutoff_nm: float = 0.5, frame_stride: int = 1) -> list[dict]:
    """Run aggregation analysis over every `frame_stride`-th frame of a real
    GROMACS trajectory. Returns one summary dict per analyzed frame:
    {time_ps, n_aggregates, distribution, max_aggregate_size, max_aggregate_rg_nm}.

    topology_path should be a .gro file, not .tpr: MDAnalysis 2.9's TPR
    parser doesn't yet support this GROMACS build's tpx version (138,
    confirmed via a real ValueError, not assumed) -- a .gro carries enough
    atom/residue naming to select and cluster molecules correctly, which is
    all this function needs (no bonded topology or exact per-atom masses
    required, matching analysis.py's own equal-mass Rg assumption).
    """
    u = mda.Universe(topology_path, xtc_path)
    residues = u.select_atoms(f"resname {resname}").residues
    if len(residues) == 0:
        raise ValueError(f"No residues found with resname '{resname}' -- check the .tpr's actual residue naming "
                         f"(e.g. via `gmx dump -s file.tpr` or MDAnalysis's own `u.residues.resnames`) "
                         f"rather than assuming this name is right.")
    molecule_atom_groups = [res.atoms for res in residues]

    results = []
    for ts in u.trajectory[::frame_stride]:
        aggs = find_aggregates_pbc(molecule_atom_groups, ts.dimensions, cutoff_nm=cutoff_nm)
        dist = aggregation_number_distribution(aggs)
        largest = max(aggs, key=len)
        raw_positions = [molecule_atom_groups[mol_idx].positions for mol_idx in largest]
        unwrapped = unwrap_cluster_positions(raw_positions, ts.dimensions)
        largest_positions = [tuple(p / 10.0) for mol_positions in unwrapped for p in mol_positions]
        results.append({
            "time_ps": float(ts.time),
            "n_aggregates": len(aggs),
            "distribution": dist,
            "max_aggregate_size": len(largest),
            "max_aggregate_rg_nm": radius_of_gyration(largest_positions),
        })
    return results


if __name__ == "__main__":
    # Real test against the 20x SDS self-assembly run's in-progress
    # trajectory (see GROMACS_SETUP.md) -- whatever frames have been written
    # so far, not fabricated data. Confirms this module actually works
    # before the full 10 ns run finishes, so results are ready the moment it does.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", help=".gro file (not .tpr -- see analyze_trajectory's docstring)")
    parser.add_argument("xtc")
    parser.add_argument("--resname", default="MOL", help="acpype's default ligand residue name in the .gro "
                        "(distinct from the moleculetype name used in the .top's [molecules] section)")
    parser.add_argument("--cutoff-nm", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    results = analyze_trajectory(args.topology, args.xtc, args.resname,
                                 cutoff_nm=args.cutoff_nm, frame_stride=args.stride)
    print(f"Analyzed {len(results)} frames (cutoff={args.cutoff_nm} nm, resname={args.resname})")
    for r in results:
        print(f"  t={r['time_ps']:8.1f} ps  n_aggregates={r['n_aggregates']:3d}  "
              f"largest={r['max_aggregate_size']:2d}  Rg(largest)={r['max_aggregate_rg_nm']:.3f} nm  "
              f"dist={r['distribution']}")
