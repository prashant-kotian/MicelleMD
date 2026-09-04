"""Tests for the new shape-descriptor and SASA additions to analysis.py
(gyration_tensor_eigenvalues, shape_descriptors, solvent_accessible_surface_area) --
part of the MicelleMD ROADMAP.md analysis-output list ("Micelle shape
descriptors: radius of gyration, asphericity/eccentricity" and "Solvent-
accessible surface area (SASA)")."""

from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from micellemd.analysis import (
    radius_of_gyration, gyration_tensor_eigenvalues, shape_descriptors,
    solvent_accessible_surface_area,
)


def test_gyration_tensor_eigenvalues_sum_to_rg_squared():
    """Rg^2 = L1+L2+L3 is an exact mathematical identity (trace of the
    gyration tensor), true for any point set -- a strong self-consistency
    check against the already-established radius_of_gyration()."""
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.3, -0.4, 0.7), (-0.5, 0.2, -0.1)]
    l1, l2, l3 = gyration_tensor_eigenvalues(positions)
    rg = radius_of_gyration(positions)
    assert l1 + l2 + l3 == pytest.approx(rg**2, rel=1e-9)
    assert l1 >= l2 >= l3 >= 0.0


def test_shape_descriptors_isotropic_case_is_near_spherical():
    """A regular tetrahedron's 4 vertices are exactly isotropic (equal
    eigenvalues by symmetry) -- kappa^2 should be exactly 0."""
    # regular tetrahedron vertices, centered at the origin
    positions = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    desc = shape_descriptors(positions)
    assert desc["relative_shape_anisotropy"] == pytest.approx(0.0, abs=1e-9)
    assert desc["asphericity"] == pytest.approx(0.0, abs=1e-9)


def test_shape_descriptors_linear_case_is_rod_limit():
    """Points confined to a single line are the theoretical rod limit --
    kappa^2 should be exactly 1 (all the variance is in one dimension)."""
    positions = [(x, 0.0, 0.0) for x in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    desc = shape_descriptors(positions)
    assert desc["relative_shape_anisotropy"] == pytest.approx(1.0, abs=1e-6)


def test_sasa_single_bead_equals_full_sphere_area():
    """One isolated bead has nothing to bury it -- SASA must equal the
    full expanded-sphere surface area exactly (up to sampling density)."""
    area = solvent_accessible_surface_area([(0.0, 0.0, 0.0)], bead_radius_nm=0.235,
                                           probe_radius_nm=0.21, n_sphere_points=400)
    expected = 4.0 * math.pi * (0.235 + 0.21) ** 2
    assert area == pytest.approx(expected, rel=0.02)


def test_sasa_decreases_with_burial():
    """Two beads close enough to overlap their expanded spheres must have
    less total SASA than two beads far apart (real burial, not a no-op)."""
    close = solvent_accessible_surface_area([(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)], n_sphere_points=200)
    far = solvent_accessible_surface_area([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)], n_sphere_points=200)
    single = solvent_accessible_surface_area([(0.0, 0.0, 0.0)], n_sphere_points=200)
    assert far == pytest.approx(2 * single, rel=0.02)
    assert close < far
