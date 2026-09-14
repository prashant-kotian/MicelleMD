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
from micellemd.cg_model import (make_linear_surfactant, make_gemini_surfactant,
                                 build_openmm_system, MARTINI3_BEAD_TYPES, _cross_term)


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


def test_gemini_surfactant_has_real_amide_bead_on_each_side():
    """Regression test for the 2026-09-05 fix: make_gemini_surfactant() now
    places a real, sourced SP2 amide-linkage bead between each tail and its
    headgroup (tail1-amide1-head1-spacer-head2-amide2-tail2), resolving the
    KNOWN GAP this function's docstring had flagged since 2026-09-04. Checks
    the actual bead-type sequence, not just that the system builds.

    Updated 2026-09-14: default headgroup_type changed Q4n -> Q1 (see
    MARTINI3_BEAD_TYPES' own comment -- Q4n is the anionic SDS-sulfate bead;
    this function's whole stated purpose is cationic geminis, so Q1, the
    real MARTINI 3 quaternary-ammonium bead, is now the correct default)."""
    mol = make_gemini_surfactant("gem0", n_tail_beads=2, n_spacer_beads=1)
    types = [b.bead_type for b in mol.beads]
    assert types == ["C1", "C1", "SP2", "Q1", "C1", "Q1", "SP2", "C1", "C1"]
    assert "SP2" in MARTINI3_BEAD_TYPES
    assert "Q1" in MARTINI3_BEAD_TYPES
    # real, sourced cross-terms must exist for every bead type SP2 actually
    # touches in this topology (C1 on both sides, Q1 on both sides) --
    # _cross_term returning None would mean _build_type_lookup_tables()
    # silently falls through with a KeyError/None params, so this is a real
    # precondition check, not a redundant one
    assert _cross_term("SP2", "C1") is not None
    assert _cross_term("SP2", "Q1") is not None


def test_gemini_surfactant_explicit_anionic_headgroup_still_available():
    """Backward-compat check: the pre-2026-09-14 anionic (Q4n) gemini variant
    is still reachable by passing headgroup_type explicitly, for anyone who
    needs an anionic gemini rather than this project's actual cationic one."""
    mol = make_gemini_surfactant("gem0", n_tail_beads=2, n_spacer_beads=1, headgroup_type="Q4n")
    types = [b.bead_type for b in mol.beads]
    assert types == ["C1", "C1", "SP2", "Q4n", "C1", "Q4n", "SP2", "C1", "C1"]


def test_q1_cationic_headgroup_cross_terms_all_sourced():
    """Real-source check for the 2026-09-14 Q1 addition (cationic
    quaternary-ammonium headgroup, sourced from POPC's real choline bead in
    the official MARTINI 3 force field -- see MARTINI3_BEAD_TYPES' comment).
    Every pair Q1 can actually appear next to -- including Q4n, since this
    project's own next planned step is a gemini(Q1)+SDS(Q4n) MIXTURE system,
    where both types coexist in one global lookup table -- must have a real
    sourced cross-term, or _build_type_lookup_tables() would hard-error."""
    assert MARTINI3_BEAD_TYPES["Q1"]["sigma"] == pytest.approx(0.470)
    assert MARTINI3_BEAD_TYPES["Q1"]["epsilon"] == pytest.approx(3.980)
    for other in ("C1", "W", "SP2", "Q4n"):
        assert _cross_term("Q1", other) is not None, f"missing real Q1-{other} cross-term"


def test_gemini_plus_sds_mixture_system_builds_and_runs_stably():
    """The actual future use case this project is heading toward: a gemini
    cationic (Q1) surfactant coexisting with linear anionic SDS (Q4n) in one
    System. Confirms the global CustomNonbondedForce lookup table handles
    both headgroup types together without NaN/inf energy -- not just that
    each type works in isolation (the two tests above already cover that)."""
    molecules = ([make_gemini_surfactant(f"gem{i}", n_tail_beads=2, n_spacer_beads=1) for i in range(4)]
                 + [make_linear_surfactant(f"sds{i}", n_tail_beads=3) for i in range(4)])
    system, positions = build_openmm_system(molecules, box_size_nm=10.0)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.02 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    pe = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert pe == pe  # NaN check
    assert abs(pe) < 1e6


def test_gemini_surfactant_system_builds_and_runs_stably():
    """End-to-end check that the new SP2 bead's real MARTINI parameters
    (self- and cross-interaction) are wired all the way through the
    CustomNonbondedForce lookup tables without producing NaN/inf energy --
    not just that the bead exists in the topology."""
    molecules = [make_gemini_surfactant(f"gem{i}", n_tail_beads=2, n_spacer_beads=1) for i in range(8)]
    system, positions = build_openmm_system(molecules, box_size_nm=10.0)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.02 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    pe = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert pe == pe  # NaN check (NaN != NaN)
    assert abs(pe) < 1e6  # sanity bound, not a divergent blow-up
