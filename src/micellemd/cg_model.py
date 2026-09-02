"""Coarse-grained bead/molecule data model and OpenMM system builder.

Bead types and nonbonded parameters below are the REAL MARTINI 3 values
(Souza et al. 2021, Nat. Methods, DOI: 10.1038/s41592-021-01098-3), sourced
directly from the official `martini_v3.0.0.itp` particle-definition file
(version 3.0.0, dated 2021-03-29 -- the exact citation given in the file
header matches the Souza paper). That file was obtained from
github.com/maccallumlab/martini_openmm (a peer-reviewed Martini-in-OpenMM
implementation, Biophys. J. 2023, DOI: 10.1016/j.bpj.2023.04.007), which
bundles the unmodified official Martini Force Field Initiative distribution
as GROMACS-format test data -- not re-derived or guessed.

Bead assignment for an SDS-analog anionic surfactant (3x C1 tail + 1x Q4n
head) matches the published MARTINI 3 SDS model (Vainikka et al. 2021,
referenced in Frontiers in Materials 10.3389/fmats.2022.1011164), which is
also this project's validated real-N_agg sanity check (Bales et al. 1998,
N_agg ~ 44.8-54.2, already cited in SurfactantKit's
literature_validation_notes.md).

Cross-species (headgroup<->tail<->water) interactions are NOT well
approximated by OpenMM's default Lorentz-Berthelot combining rule --
MARTINI defines an explicit, non-combining nonbonded matrix, and the
combining-rule approximation differs from the real cross terms by up to
~2x on epsilon for these bead pairs (checked, not assumed; see
NONBONDED_CROSS_TERMS below). build_openmm_system() therefore adds explicit
NonbondedForce exceptions for every cross-type particle pair so the real
MARTINI matrix (not the combining-rule approximation) governs the physics.

Scope honestly stated: this module's own self-assembly test (20 small
molecules, 1000 Langevin steps) is a software-plumbing smoke test, not a
production-scale run -- it is not sized or run long enough to reproduce a
quantitative aggregation number for comparison against Bales et al. A real
quantitative CG validation needs a much larger system and longer run
(comparable in scale to the atomistic GAFF route's 10 ns/20-molecule run),
which is the next concrete step, not yet done here.

Counterions (Na+) are not modeled as explicit particles -- the headgroup
bead simply carries the surfactant's formal charge, an implicit-counterion
simplification already implicit in the pre-existing model architecture,
unchanged by this update.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import openmm
import openmm.unit as unit


@dataclass
class CGBead:
    name: str
    bead_type: str  # e.g. "Q4n" (charged headgroup), "C1" (hydrophobic tail) -- real MARTINI 3 bead type name, see MARTINI3_BEAD_TYPES
    mass_amu: float
    charge: float = 0.0


@dataclass
class CGMolecule:
    name: str
    beads: list[CGBead]
    bonds: list[tuple[int, int]] = field(default_factory=list)  # (bead_index_i, bead_index_j)
    bond_length_nm: float = 0.47  # standard MARTINI bead spacing convention (~0.47 nm), this specific value IS the standard MARTINI bead-bead distance, not surfactant-specific
    bond_k: float = 5000.0  # kJ/mol/nm^2 -- still a generic placeholder; nonbonded params below are now real MARTINI 3, but real molecule-specific bonded (bond/angle) constants (e.g. from an actual SDS .itp) were out of scope this round and remain unverified


# Real MARTINI 3 bead type -> (mass amu, LJ sigma nm, LJ epsilon kJ/mol),
# self-interaction (same-type) values. Source: official martini_v3.0.0.itp,
# [ atomtypes ] + [ nonbond_params ] sections -- see module docstring.
MARTINI3_BEAD_TYPES = {
    "Q4n": {"mass": 72.0, "sigma": 0.470, "epsilon": 5.20},  # anionic sulfate-type headgroup (real SDS bead, Vainikka et al. 2021)
    "C1":  {"mass": 72.0, "sigma": 0.470, "epsilon": 3.39},  # apolar alkyl tail bead
    "W":   {"mass": 72.0, "sigma": 0.470, "epsilon": 4.65},  # standard MARTINI water bead
}

# Real MARTINI 3 cross-species (different bead type) nonbonded parameters --
# NOT derivable from the self-interaction values above via a combining rule
# (MARTINI uses an explicit interaction matrix). Source: same
# martini_v3.0.0.itp [ nonbond_params ] section, off-diagonal entries.
# Checked against OpenMM's default Lorentz-Berthelot combining-rule
# approximation: C1-W epsilon differs by ~1.9x (2.06 real vs. 3.97
# combining), C1-Q4n epsilon differs by ~2.0x (2.143 real vs. 4.20
# combining) -- physically significant for the hydrophobic-effect-driven
# self-assembly this model exists to capture, hence applied as explicit
# NonbondedForce exceptions in build_openmm_system() rather than left to
# the default combining rule.
NONBONDED_CROSS_TERMS = {
    ("C1", "W"):   {"sigma": 0.470, "epsilon": 2.060},
    ("C1", "Q4n"): {"sigma": 0.570, "epsilon": 2.143},
    ("Q4n", "W"):  {"sigma": 0.465, "epsilon": 5.960},
}


def _cross_term(type_a: str, type_b: str) -> dict | None:
    """Look up a real MARTINI cross-interaction regardless of pair order."""
    return NONBONDED_CROSS_TERMS.get((type_a, type_b)) or NONBONDED_CROSS_TERMS.get((type_b, type_a))


def make_linear_surfactant(name: str, n_tail_beads: int, headgroup_charge: float = 1.0) -> CGMolecule:
    """A simple linear CG surfactant: 1 headgroup bead + n_tail_beads
    hydrophobic tail beads in a chain -- structurally analogous to how a
    real MARTINI surfactant (e.g. SDS: 1 polar bead + 3 tail beads) is
    built, using real MARTINI 3 bead parameters (see module docstring)."""
    beads = [CGBead(name=f"{name}_head", bead_type="Q4n", mass_amu=72.0, charge=headgroup_charge)]
    for i in range(n_tail_beads):
        beads.append(CGBead(name=f"{name}_tail{i}", bead_type="C1", mass_amu=72.0))
    bonds = [(i, i + 1) for i in range(len(beads) - 1)]
    return CGMolecule(name=name, beads=beads, bonds=bonds)


def make_gemini_surfactant(name: str, n_tail_beads: int, n_spacer_beads: int,
                           headgroup_charge: float = 1.0) -> CGMolecule:
    """A gemini (dimeric) CG surfactant: two head-tail units joined near the
    headgroups by a spacer -- the actual PhD thesis architecture (amidoamine-
    derived gemini cationic surfactants), not just a generic linear chain.

    Topology: tail1 -- head1 -- spacer -- head2 -- tail2, matching the real
    structural definition (spacer links near the headgroups, per this
    project's own SurfBench Tier 2 Category N: 'the spacer covalently
    connects the two head-tail halves near the headgroups'). Uses the same
    real MARTINI 3 bead parameters as make_linear_surfactant -- the open
    caveat here is the spacer-length-to-bead-count mapping (see
    n_spacer_beads below), not the nonbonded physics.

    n_spacer_beads: spacer length in CG beads. Real gemini spacers are
    commonly C2-C12 alkyl chains; a single CG bead typically maps to ~4
    heavy atoms in MARTINI, so n_spacer_beads=1 or 2 corresponds roughly to
    the short-spacer (C2-C6ish) regime most relevant to this project's own
    amidoamine gemini work -- exact atom-to-bead mapping still needs
    literature verification (see ROADMAP.md open questions) before treating
    this correspondence as precise.
    """
    beads = []
    # tail 1 (built head-to-tail so bond order is contiguous: tail1 -> head1)
    for i in range(n_tail_beads):
        beads.append(CGBead(name=f"{name}_tail1_{i}", bead_type="C1", mass_amu=72.0))
    head1_idx = len(beads)
    beads.append(CGBead(name=f"{name}_head1", bead_type="Q4n", mass_amu=72.0, charge=headgroup_charge))
    spacer_start = len(beads)
    for i in range(n_spacer_beads):
        beads.append(CGBead(name=f"{name}_spacer{i}", bead_type="C1", mass_amu=72.0))
    head2_idx = len(beads)
    beads.append(CGBead(name=f"{name}_head2", bead_type="Q4n", mass_amu=72.0, charge=headgroup_charge))
    for i in range(n_tail_beads):
        beads.append(CGBead(name=f"{name}_tail2_{i}", bead_type="C1", mass_amu=72.0))

    bonds = [(i, i + 1) for i in range(len(beads) - 1)]  # linear backbone: tail1-head1-spacer-head2-tail2
    return CGMolecule(name=name, beads=beads, bonds=bonds)


def build_openmm_system(molecules: list[CGMolecule], box_size_nm: float = 10.0) -> tuple[openmm.System, list[tuple]]:
    """Build a minimal OpenMM System from a list of CGMolecule instances:
    particles with mass, a HarmonicBondForce for intra-molecule bonds, and
    a NonbondedForce (LJ + Coulomb) using the real MARTINI3_BEAD_TYPES self
    terms plus explicit NONBONDED_CROSS_TERMS exceptions for cross-species
    pairs (see module docstring for why exceptions are needed). Returns
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
    bead_types = []  # parallel to particle indices, needed to add cross-term exceptions below
    particle_offset = 0
    grid_spacing = 1.5  # nm, generic starting spacing, not physically tuned
    n_placed = 0
    per_row = max(1, int(box_size_nm / grid_spacing))

    for mol in molecules:
        row, col = divmod(n_placed, per_row)
        origin = openmm.Vec3(col * grid_spacing, row * grid_spacing, 0.5 * box_size_nm)
        for i, bead in enumerate(mol.beads):
            params = MARTINI3_BEAD_TYPES[bead.bead_type]
            system.addParticle(params["mass"] * unit.amu)
            nonbonded.addParticle(bead.charge, params["sigma"] * unit.nanometer, params["epsilon"] * unit.kilojoule_per_mole)
            bead_types.append(bead.bead_type)
            positions.append(origin + openmm.Vec3(0, 0, i * mol.bond_length_nm))
        for i, j in mol.bonds:
            bond_force.addBond(particle_offset + i, particle_offset + j,
                              mol.bond_length_nm * unit.nanometer, mol.bond_k * unit.kilojoule_per_mole / unit.nanometer**2)
        particle_offset += len(mol.beads)
        n_placed += 1

    # No 1-2 (bonded-pair) nonbonded exclusions here, matching this model's
    # pre-existing (undocumented but consistent) simplification: bonded
    # beads already interact via HarmonicBondForce, and were never excluded
    # from LJ/Coulomb either before or after this update.
    n_particles = len(bead_types)
    for p1 in range(n_particles):
        for p2 in range(p1 + 1, n_particles):
            if bead_types[p1] == bead_types[p2]:
                continue  # same-type pair: default combining rule already reproduces the correct self term
            cross = _cross_term(bead_types[p1], bead_types[p2])
            if cross is None:
                continue
            charge_prod = nonbonded.getParticleParameters(p1)[0] * nonbonded.getParticleParameters(p2)[0]
            nonbonded.addException(p1, p2, charge_prod, cross["sigma"] * unit.nanometer, cross["epsilon"] * unit.kilojoule_per_mole)

    system.addForce(bond_force)
    system.addForce(nonbonded)
    return system, positions


def run_stability_check(molecules: list[CGMolecule], n_steps: int = 1000, box_size_nm: float = 10.0) -> dict:
    """Real, executable plumbing test: build the system, run a short
    Langevin dynamics simulation, and confirm energy stays finite (no NaN
    blow-up) -- proves bead placement, bonded/nonbonded force setup, and
    the integrator are wired correctly. Uses real MARTINI 3 nonbonded
    parameters but is still too small/short to be a quantitative physical
    self-assembly validation (see module docstring "Scope honestly
    stated")."""
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

    # split final positions back out per-molecule for downstream analysis
    # (e.g. analysis.find_aggregates), which needs per-molecule grouping,
    # not one flat particle list
    final_positions_nm = [p.value_in_unit(unit.nanometer) for p in final_state.getPositions()]
    per_molecule_positions = []
    offset = 0
    for mol in molecules:
        n = len(mol.beads)
        per_molecule_positions.append([tuple(final_positions_nm[offset + i]) for i in range(n)])
        offset += n

    return {
        "n_particles": system.getNumParticles(),
        "n_steps_run": n_steps,
        "initial_potential_energy_kJ_mol": initial_pe,
        "final_potential_energy_kJ_mol": final_pe,
        "stable": stable,
        "final_positions_per_molecule_nm": per_molecule_positions,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analysis import find_aggregates, aggregation_number_distribution, radius_of_gyration

    print(f"OpenMM version: {openmm.version.version}")

    # build a toy system: a mix of linear AND gemini surfactants, proving
    # both topologies work through the same pipeline
    mols = ([make_linear_surfactant(f"lin{i}", n_tail_beads=3) for i in range(12)]
            + [make_gemini_surfactant(f"gem{i}", n_tail_beads=2, n_spacer_beads=1) for i in range(8)])
    print(f"Built {len(mols)} molecules ({sum(len(m.beads) for m in mols)} total beads): "
          f"12 linear (4 beads each) + 8 gemini (7 beads each)")

    result = run_stability_check(mols, n_steps=1000)
    print("\nStability check with real MARTINI 3 nonbonded parameters (proves OpenMM plumbing works; "
          "still too small/short for quantitative physical validation -- see module docstring):")
    for k, v in result.items():
        if k != "final_positions_per_molecule_nm":
            print(f"  {k}: {v}")

    if not result["stable"]:
        print("\nFAILED -- energy diverged, something in the system setup is wrong.")
        sys.exit(1)
    print("PASSED -- system remained numerically stable over the test run.")

    # real end-to-end demonstration: feed the actual simulated positions
    # into the analysis layer built separately in analysis.py
    positions = result["final_positions_per_molecule_nm"]
    aggregates = find_aggregates(positions, cutoff_nm=0.6)
    dist = aggregation_number_distribution(aggregates)
    print(f"\nEnd-to-end analysis on real simulated (real-MARTINI-3-parameter) positions:")
    print(f"  Aggregation number distribution: {dist}")
    print(f"  (Not yet a quantitative self-assembly result -- real LJ parameters now, but still a "
          f"short 1000-step run with a generic grid start on a small system -- but proves cg_model.py -> "
          f"analysis.py integration works on real simulation output, not just synthetic test positions.)")
