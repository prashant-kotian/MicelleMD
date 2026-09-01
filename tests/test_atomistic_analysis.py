"""Tests for the PBC-aware aggregation analysis, targeting the two real bugs
found and fixed this session (see atomistic_analysis.py's docstrings):
periodic-boundary-wrapped Rg inflation, and MDAnalysis's unsupported .tpr
version workaround. These run on the Ubuntu machine (conda env `chem_sim`)
-- MDAnalysis isn't installed on Windows for this project."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from micellemd.atomistic_analysis import unwrap_cluster_positions, find_aggregates_pbc


class FakeAtomGroup:
    """Minimal stand-in for an MDAnalysis AtomGroup -- find_aggregates_pbc
    only ever touches `.positions`, so a real Universe isn't needed to unit
    test the clustering logic itself."""
    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype=float)


def test_unwrap_is_identity_when_nothing_is_wrapped():
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    positions_list = [np.array([[1.0, 1.0, 1.0]]), np.array([[2.0, 2.0, 2.0]])]
    unwrapped = unwrap_cluster_positions(positions_list, box)
    assert np.allclose(unwrapped[0], positions_list[0])
    assert np.allclose(unwrapped[1], positions_list[1])


def test_unwrap_fixes_periodic_boundary_split():
    # 10 Angstrom box. Molecule A near x=0.5, molecule B near x=9.5 -- these
    # are genuine periodic neighbours (0.5 + (10-9.5) = 1.0 apart via
    # minimum image), but their RAW coordinates are 9.0 apart. This is
    # exactly the real bug found testing against the live 20-SDS trajectory
    # (an anomalous 3.44 nm Rg for a 3-molecule cluster at t=0).
    box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    positions_list = [np.array([[0.5, 5.0, 5.0]]), np.array([[9.5, 5.0, 5.0]])]
    unwrapped = unwrap_cluster_positions(positions_list, box)
    assert unwrapped[0][0][0] == pytest.approx(0.5)
    assert unwrapped[1][0][0] == pytest.approx(-0.5)  # nearest periodic image of 9.5, relative to 0.5


def test_find_aggregates_pbc_detects_neighbours_across_the_boundary():
    # 100 Angstrom box (10 nm). Molecule A at x=1, molecule B at x=99 --
    # minimum-image distance is 2 Angstrom (0.2 nm), well inside a 0.5 nm
    # cutoff, but their raw coordinates are 98 Angstrom apart. A non-PBC-
    # aware clustering (like analysis.find_aggregates) would wrongly call
    # these isolated; this PBC-aware version must not.
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    mol_a = FakeAtomGroup([[1.0, 50.0, 50.0]])
    mol_b = FakeAtomGroup([[99.0, 50.0, 50.0]])
    mol_c = FakeAtomGroup([[50.0, 50.0, 50.0]])  # far from both (49 Angstrom = 4.9 nm), genuinely isolated

    aggregates = find_aggregates_pbc([mol_a, mol_b, mol_c], box, cutoff_nm=0.5)
    sizes = sorted(len(a) for a in aggregates)
    assert sizes == [1, 2]  # one pair (A+B, across the boundary) plus isolated C


def test_find_aggregates_pbc_no_false_positives_within_cutoff_distance():
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    mol_a = FakeAtomGroup([[10.0, 50.0, 50.0]])
    mol_b = FakeAtomGroup([[50.0, 50.0, 50.0]])  # 40 Angstrom = 4 nm away, not periodic-adjacent
    aggregates = find_aggregates_pbc([mol_a, mol_b], box, cutoff_nm=0.5)
    assert sorted(len(a) for a in aggregates) == [1, 1]
