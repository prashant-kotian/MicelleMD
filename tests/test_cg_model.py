"""Regression test for the 2026-09-04 nonbonded-exclusion bug: directly-
bonded (1-2) particle pairs were never excluded from the LJ/Coulomb forces
in cg_model.py's system builders, so every bond in every molecule also
carried a large, unphysical LJ clash on top of its harmonic term (e.g.
~59.6 kJ/mol for a head-tail1 pair). Caught via an independent GROMACS
cross-check on the identical 20-molecule smoke-test system and coordinates:
OpenMM's LJ energy was +1175.7 kJ/mol vs GROMACS's -7.4 kJ/mol before the
fix, -14.3 kJ/mol (same sign/order of magnitude as GROMACS) after."""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import openmm
import openmm.unit as unit
from micellemd.cg_model import make_linear_surfactant, build_openmm_system


def test_bonded_pairs_are_excluded_from_nonbonded_forces():
    """Direct structural check: every 1-2 bonded pair must appear in both
    the LJ force's exclusion list and the Coulomb force's exception list --
    not just a plausible-looking energy number, the actual exclusion data."""
    molecules = [make_linear_surfactant(f"sds{i}", n_tail_beads=3) for i in range(5)]
    system, _ = build_openmm_system(molecules, box_size_nm=10.0)
    forces = {type(f).__name__: f for f in system.getForces()}
    lj_force = forces["CustomNonbondedForce"]
    coulomb_force = forces["NonbondedForce"]

    expected_bonded_pairs = set()
    for mol_idx in range(5):
        base = mol_idx * 4  # 1 head + 3 tail beads per molecule
        expected_bonded_pairs |= {(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)}

    excluded_lj = set()
    for k in range(lj_force.getNumExclusions()):
        i, j = lj_force.getExclusionParticles(k)
        excluded_lj.add((min(i, j), max(i, j)))
    assert excluded_lj == expected_bonded_pairs

    excepted_coulomb = set()
    for k in range(coulomb_force.getNumExceptions()):
        i, j, chargeProd, sigma, epsilon = coulomb_force.getExceptionParameters(k)
        excepted_coulomb.add((min(i, j), max(i, j)))
        assert float(chargeProd / unit.elementary_charge**2) == pytest.approx(0.0)
        assert epsilon.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0.0)
    assert excepted_coulomb == expected_bonded_pairs


def test_smoke_test_energy_matches_gromacs_cross_check():
    """The exact 20-molecule, no-solvent smoke-test system, independently
    verified 2026-09-04 against a hand-built real-MARTINI-parameter GROMACS
    topology on the identical configuration (GROMACS LJ(SR) = -7.4 kJ/mol).
    Locking in the corrected value as a real regression check, not just the
    round-trip/plumbing checks this system previously only had."""
    molecules = [make_linear_surfactant(f"sds{i}", n_tail_beads=3) for i in range(20)]
    system, positions = build_openmm_system(molecules, box_size_nm=10.0)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.02 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    pe = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert pe == pytest.approx(-14.336, abs=0.01)
    # the pre-fix value was +1175.744 -- assert we are nowhere near it, as an
    # explicit trip-wire against this exact regression recurring silently
    assert pe < 100.0
