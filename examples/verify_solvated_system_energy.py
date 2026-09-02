"""Regression test: build_openmm_system()'s CustomNonbondedForce-based
nonbonded scheme against a fully independent, hand-coded reference
calculation (raw pairwise LJ + reaction-field Coulomb, computed directly
from MARTINI3_BEAD_TYPES/NONBONDED_CROSS_TERMS with no OpenMM machinery
involved at all).

Why this script exists: build_openmm_system() originally (2026-09-03,
same day as the MARTINI 3 parameter update) used NonbondedForce's
per-pair addException() to apply cross-type LJ parameters. That is
correct for a small, fixed set of special pairs (like 1-2 bonded
exclusions) but WRONG for a bulk type-pair rule, because OpenMM
exceptions ignore the cutoff distance entirely -- confirmed empirically
on a small 2-molecule/11-bead test system: 16 of 24 exceptions added
were for pairs already >1.1 nm apart, silently forcing permanent,
always-on LJ interactions between particles that should not have been
interacting at all. Caught while building build_solvated_openmm_system()
(needed for production-scale runs) and cross-checking its
CustomNonbondedForce-based scheme against the original exception-based
one -- they disagreed by ~0.2-0.3%, small enough to be tempting to
hand-wave away as numerical noise, but a fully independent hand
calculation (this script) confirmed the CustomNonbondedForce answer was
the physically correct one. build_openmm_system() was then fixed to use
the same CustomNonbondedForce scheme (see cg_model.py's
_make_nonbonded_forces() docstring for the full story).

This script is kept as an ongoing regression test, not just a one-time
bug report -- if this ever fails again, don't assume it's numerical
noise; the same class of subtle-but-real discrepancy has happened once
already in this exact module.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "micellemd"))
import openmm
import openmm.unit as unit
from cg_model import make_linear_surfactant, make_gemini_surfactant, build_openmm_system, MARTINI3_BEAD_TYPES, _cross_term

ONE_4PI_EPS0 = 138.935456  # kJ*nm/mol / e^2, OpenMM's real Coulomb constant
RF_DIELECTRIC = 78.3       # OpenMM NonbondedForce's default reaction-field dielectric


def hand_calculated_energy(bead_types: list[str], charges: list[float], bonds: list[tuple],
                           bond_length_nm: float, bond_k: float, positions, cutoff_nm: float = 1.1) -> float:
    """Independent reference: raw pairwise LJ (real MARTINI self/cross
    terms, no combining-rule assumptions) + reaction-field Coulomb +
    harmonic bonds, computed with no OpenMM force objects at all -- the
    thing that originally caught the exception-based scheme's bug."""
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    krf = (1.0 / cutoff_nm ** 3) * (RF_DIELECTRIC - 1) / (2 * RF_DIELECTRIC + 1)
    crf = (1.0 / cutoff_nm) * (3 * RF_DIELECTRIC) / (2 * RF_DIELECTRIC + 1)

    total = 0.0
    n = len(bead_types)
    for i in range(n):
        for j in range(i + 1, n):
            r = dist(positions[i], positions[j])
            if r >= cutoff_nm:
                continue
            t1, t2 = bead_types[i], bead_types[j]
            params = MARTINI3_BEAD_TYPES[t1] if t1 == t2 else _cross_term(t1, t2)
            sigma, eps = params["sigma"], params["epsilon"]
            total += 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
            qq = charges[i] * charges[j]
            if qq != 0:
                total += ONE_4PI_EPS0 * qq * (1.0 / r + krf * r ** 2 - crf)

    for i, j in bonds:
        r = dist(positions[i], positions[j])
        total += 0.5 * bond_k * (r - bond_length_nm) ** 2
    return total


def openmm_energy(system, positions) -> float:
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


if __name__ == "__main__":
    box_size_nm = 10.0
    mols = ([make_linear_surfactant(f"lin{i}", n_tail_beads=3) for i in range(12)]
            + [make_gemini_surfactant(f"gem{i}", n_tail_beads=2, n_spacer_beads=1) for i in range(8)])
    print(f"Test system: {len(mols)} molecules, {sum(len(m.beads) for m in mols)} beads "
          f"(same toy system cg_model.py's own __main__ demo uses)")

    system, positions = build_openmm_system(mols, box_size_nm)
    pe_openmm = openmm_energy(system, positions)

    bead_types, charges, bonds = [], [], []
    offset = 0
    for mol in mols:
        for bead in mol.beads:
            bead_types.append(bead.bead_type)
            charges.append(bead.charge)
        for i, j in mol.bonds:
            bonds.append((offset + i, offset + j))
        offset += len(mol.beads)

    pe_hand = hand_calculated_energy(bead_types, charges, bonds, mols[0].bond_length_nm, mols[0].bond_k, positions)

    print(f"\nbuild_openmm_system() (CustomNonbondedForce, real scheme): {pe_openmm:.6f} kJ/mol")
    print(f"Independent hand-calculated reference (no OpenMM forces):   {pe_hand:.6f} kJ/mol")
    diff = abs(pe_openmm - pe_hand)
    print(f"Absolute difference: {diff:.6f} kJ/mol")

    # NonbondedForce's default LJ long-range (dispersion) correction is an
    # analytic bulk approximation this hand calculation doesn't include --
    # a few kJ/mol gap from that alone is expected and not a bug; anything
    # beyond a generous few-kJ/mol margin means the real per-pair physics
    # disagrees, which IS a bug (this exact check caught one before).
    if diff < 5.0:
        print("\nPASS -- agrees with the independent reference within the expected long-range-correction margin.")
    else:
        print("\nFAIL -- discrepancy too large to explain by the long-range correction alone; investigate before trusting cg_model.py.")
        sys.exit(1)
