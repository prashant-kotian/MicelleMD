"""Coarse-grained bead/molecule data model and OpenMM system builder.

IMPORTANT — read before using for anything beyond a plumbing test:
The bead types, masses, and Lennard-Jones parameters below are DELIBERATELY
GENERIC PLACEHOLDERS, not the real MARTINI 3 surfactant parameter set. Real
MARTINI nonbonded parameters (the full bead-type interaction matrix) come
from the MARTINI 3 publication and its official .itp parameter files
(Souza et al. 2021, Nat. Methods) and were not available to verify this
session (WebSearch budget exhausted) -- per this project's established
"verify before hardcode" discipline, they are NOT guessed here. This module
exists to prove the OpenMM integration/software plumbing works end to end
(bead placement -> bonded+nonbonded forces -> integrator -> stable dynamics),
which is a real, checkable engineering result independent of the parameter
values. Swap PLACEHOLDER_BEAD_TYPES for the real MARTINI 3 table before
trusting any physical conclusion from a simulation built on this module.

vermouth (the real MARTINI topology-generation tool, maintained by the
MARTINI developers) is installed and importable in this environment --
confirmed working after fixing a real Windows-specific encoding bug (run
with PYTHONUTF8=1). It bundles genuine MARTINI 3.001 force-field
infrastructure, but that bundle is protein/nucleotide-focused, not
surfactant-specific -- the real next step, once WebSearch is available
again, is sourcing the actual MARTINI 3 lipid/surfactant .itp parameters
(from the MARTINI GitHub/website) and wiring vermouth's mapping machinery
to use them, rather than the hand-rolled generic model here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import openmm
import openmm.unit as unit


@dataclass
class CGBead:
    name: str
    bead_type: str  # e.g. "Q0" (charged headgroup), "C1" (hydrophobic tail) -- MARTINI-style naming convention, PLACEHOLDER parameters
    mass_amu: float
    charge: float = 0.0


@dataclass
class CGMolecule:
    name: str
    beads: list[CGBead]
    bonds: list[tuple[int, int]] = field(default_factory=list)  # (bead_index_i, bead_index_j)
    bond_length_nm: float = 0.47  # standard MARTINI bead spacing convention (~0.47 nm), this specific value IS the standard MARTINI bead-bead distance, not surfactant-specific
    bond_k: float = 5000.0  # kJ/mol/nm^2 -- PLACEHOLDER, not a verified MARTINI bond force constant


# PLACEHOLDER bead type -> (mass amu, LJ sigma nm, LJ epsilon kJ/mol)
# Generic hydrophobic/hydrophilic contrast (headgroup polar+charged,
# tail apolar) sufficient to demonstrate phase-segregation-driven
# self-assembly behavior in a toy system -- NOT literature-verified
# MARTINI numbers. See module docstring.
PLACEHOLDER_BEAD_TYPES = {
    "Q0": {"mass": 72.0, "sigma": 0.47, "epsilon": 4.0},   # charged/polar headgroup bead (placeholder)
    "C1": {"mass": 72.0, "sigma": 0.47, "epsilon": 3.5},   # hydrophobic tail bead (placeholder)
    "W":  {"mass": 72.0, "sigma": 0.47, "epsilon": 5.0},   # solvent (water) bead (placeholder)
}


def make_linear_surfactant(name: str, n_tail_beads: int, headgroup_charge: float = 1.0) -> CGMolecule:
    """A simple linear CG surfactant: 1 headgroup bead + n_tail_beads
    hydrophobic tail beads in a chain -- structurally analogous to how a
    real MARTINI surfactant (e.g. SDS: 1 polar bead + 3 tail beads) is
    built, using PLACEHOLDER bead parameters (see module docstring)."""
    beads = [CGBead(name=f"{name}_head", bead_type="Q0", mass_amu=72.0, charge=headgroup_charge)]
    for i in range(n_tail_beads):
        beads.append(CGBead(name=f"{name}_tail{i}", bead_type="C1", mass_amu=72.0))
    bonds = [(i, i + 1) for i in range(len(beads) - 1)]
    return CGMolecule(name=name, beads=beads, bonds=bonds)


def build_openmm_system(molecules: list[CGMolecule], box_size_nm: float = 10.0) -> tuple[openmm.System, list[tuple]]:
    """Build a minimal OpenMM System from a list of CGMolecule instances:
    particles with mass, a HarmonicBondForce for intra-molecule bonds, and
    a NonbondedForce (LJ + Coulomb) using PLACEHOLDER_BEAD_TYPES. Returns
    (system, positions) where positions are simple placed-on-a-grid
    starting coordinates -- good enough to prove the plumbing runs and is
    numerically stable, not a physically meaningful starting configuration."""
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_size_nm, 0, 0), openmm.Vec3(0, box_size_nm, 0), openmm.Vec3(0, 0, box_size_nm))

    bond_force = openmm.HarmonicBondForce()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.1 * unit.nanometer)

    positions = []
    particle_offset = 0
    grid_spacing = 1.5  # nm, generic starting spacing, not physically tuned
    n_placed = 0
    per_row = max(1, int(box_size_nm / grid_spacing))

    for mol in molecules:
        row, col = divmod(n_placed, per_row)
        origin = openmm.Vec3(col * grid_spacing, row * grid_spacing, 0.5 * box_size_nm)
        for i, bead in enumerate(mol.beads):
            params = PLACEHOLDER_BEAD_TYPES[bead.bead_type]
            system.addParticle(params["mass"] * unit.amu)
            nonbonded.addParticle(bead.charge, params["sigma"] * unit.nanometer, params["epsilon"] * unit.kilojoule_per_mole)
            positions.append(origin + openmm.Vec3(0, 0, i * mol.bond_length_nm))
        for i, j in mol.bonds:
            bond_force.addBond(particle_offset + i, particle_offset + j,
                              mol.bond_length_nm * unit.nanometer, mol.bond_k * unit.kilojoule_per_mole / unit.nanometer**2)
        particle_offset += len(mol.beads)
        n_placed += 1

    system.addForce(bond_force)
    system.addForce(nonbonded)
    return system, positions


def run_stability_check(molecules: list[CGMolecule], n_steps: int = 1000, box_size_nm: float = 10.0) -> dict:
    """Real, executable plumbing test: build the system, run a short
    Langevin dynamics simulation, and confirm energy stays finite (no NaN
    blow-up) -- proves bead placement, bonded/nonbonded force setup, and
    the integrator are wired correctly. Does NOT validate physical
    correctness of self-assembly behavior, since the bead parameters are
    placeholders (see module docstring)."""
    system, positions = build_openmm_system(molecules, box_size_nm)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.02 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin)

    initial_state = context.getState(getEnergy=True)
    initial_pe = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    integrator.step(n_steps)

    final_state = context.getState(getEnergy=True, getPositions=True)
    final_pe = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    import math
    stable = not (math.isnan(final_pe) or math.isinf(final_pe))
    return {
        "n_particles": system.getNumParticles(),
        "n_steps_run": n_steps,
        "initial_potential_energy_kJ_mol": initial_pe,
        "final_potential_energy_kJ_mol": final_pe,
        "stable": stable,
    }


if __name__ == "__main__":
    import sys
    print(f"OpenMM version: {openmm.version.version}")

    # build a toy system: 20 simple 4-bead linear surfactants in a periodic box
    mols = [make_linear_surfactant(f"surf{i}", n_tail_beads=3) for i in range(20)]
    print(f"Built {len(mols)} molecules, {sum(len(m.beads) for m in mols)} total beads")

    result = run_stability_check(mols, n_steps=1000)
    print("\nStability check (proves OpenMM plumbing works, NOT physical validity -- see module docstring):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if not result["stable"]:
        print("\nFAILED -- energy diverged, something in the system setup is wrong.")
        sys.exit(1)
    print("\nPASSED -- system remained numerically stable over the test run.")
