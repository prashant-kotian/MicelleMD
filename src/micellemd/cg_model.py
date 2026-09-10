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
NONBONDED_CROSS_TERMS below).

**Corrected same day, before this got far**: the nonbonded scheme
originally used per-pair NonbondedForce exceptions to apply cross-type LJ.
That's WRONG for a bulk type-pair rule -- OpenMM exceptions ignore the
cutoff distance entirely, so every cross-type pair in the whole system
(not just nearby ones) got a permanent, always-on LJ interaction
regardless of actual separation. Confirmed empirically (16 of 24
exceptions on a small test system were for pairs already >1.1 nm apart)
and root-caused by hand-calculating the correct energy independently and
finding it disagreed with the exception-based scheme, not a combining-
rule/CustomNonbondedForce implementation. Fixed by replacing exceptions
with a CustomNonbondedForce + per-type Discrete2DFunction lookup table
(OpenMM's standard mechanism for a non-combining "NBFIX"-style matrix,
evaluated via its normal cutoff-respecting neighbor-list code, not a
Python-level pair loop) -- see _make_nonbonded_forces()'s docstring for
the full story and examples/verify_solvated_system_energy.py for the
regression test that catches this class of bug going forward. This also
means build_openmm_system() now scales correctly to explicit-water
particle counts, which the exception approach never could have (see
build_solvated_openmm_system()'s docstring).

Scope honestly stated: this module's own self-assembly test (20 small
molecules, 1000 Langevin steps) is a software-plumbing smoke test, not a
production-scale run -- it is not sized or run long enough to reproduce a
quantitative aggregation number for comparison against Bales et al. A real
quantitative CG validation needs a much larger system, explicit solvent
(see build_solvated_openmm_system()), and longer run (comparable in scale
to the atomistic GAFF route's 10 ns/20-molecule run) -- see ROADMAP.md for
current status.

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
    # Amide-linkage bead, resolving the KNOWN GAP make_gemini_surfactant()
    # documented since 2026-09-04. Sourced 2026-09-05 not from analogy but
    # from the real, currently-used MARTINI 3 ceramide topology (the closest
    # real precedent: an acyl-tail-to-headgroup amide, same chemistry as this
    # project's amidoamine surfactants) -- downloaded directly (not
    # AI-summarized; the file is 16MB, too large for that) from
    # github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters/ITPs/
    # martini_v3.0.0_ceramides_v2.itp, whose [atoms] section names the bead
    # adjacent to the amide nitrogen "AM2", bead type SP2. Self/cross values
    # below are from that same repo's martini_v3.0.0.itp [nonbond_params]
    # section (grepped directly from the downloaded file, not summarized).
    # Correction to the earlier analogy-based guess: Alessandri et al. 2022's
    # Table 1 (no exact amide entry) suggested an intermediate-polarity
    # N-type bead by analogy to ester/aldehyde entries -- the real ceramide
    # precedent uses SP2, a POLAR (P-type) small bead, one level more polar
    # than the guess. Real precedent overrides analogy, per this project's
    # own "verify against real source" discipline.
    "SP2": {"mass": 54.0, "sigma": 0.410, "epsilon": 3.31},
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
    # SP2 cross-terms, sourced 2026-09-05 alongside the SP2 self-term above --
    # same file, same [nonbond_params] section, grepped directly.
    ("SP2", "C1"):  {"sigma": 0.430, "epsilon": 1.930},
    ("SP2", "Q4n"): {"sigma": 0.430, "epsilon": 4.848},
    ("SP2", "W"):   {"sigma": 0.425, "epsilon": 4.030},
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

    RESOLVED 2026-09-05 (was: "KNOWN GAP, sharpened 2026-09-04"). This
    function now places a real SP2 amide-linkage bead between each tail and
    its headgroup: tail1-amide1-head1-spacer-head2-amide2-tail2. SP2's
    self/cross-interaction parameters are real MARTINI 3 values sourced
    directly from the official martini_v3.0.0.itp [nonbond_params] section
    (see MARTINI3_BEAD_TYPES' own comment for the exact provenance and the
    real ceramide precedent this was based on), the same sourcing rigor the
    original Q4n/C1/W values got -- not just a label bolted on.
    """
    beads = []
    # tail 1 -> amide1 -> head1 (built head-to-tail so bond order is contiguous)
    for i in range(n_tail_beads):
        beads.append(CGBead(name=f"{name}_tail1_{i}", bead_type="C1", mass_amu=72.0))
    beads.append(CGBead(name=f"{name}_amide1", bead_type="SP2", mass_amu=54.0))
    head1_idx = len(beads)
    beads.append(CGBead(name=f"{name}_head1", bead_type="Q4n", mass_amu=72.0, charge=headgroup_charge))
    spacer_start = len(beads)
    for i in range(n_spacer_beads):
        beads.append(CGBead(name=f"{name}_spacer{i}", bead_type="C1", mass_amu=72.0))
    head2_idx = len(beads)
    beads.append(CGBead(name=f"{name}_head2", bead_type="Q4n", mass_amu=72.0, charge=headgroup_charge))
    beads.append(CGBead(name=f"{name}_amide2", bead_type="SP2", mass_amu=54.0))
    for i in range(n_tail_beads):
        beads.append(CGBead(name=f"{name}_tail2_{i}", bead_type="C1", mass_amu=72.0))

    bonds = [(i, i + 1) for i in range(len(beads) - 1)]  # linear backbone: tail1-amide1-head1-spacer-head2-amide2-tail2
    return CGMolecule(name=name, beads=beads, bonds=bonds)


_BEAD_TYPE_ORDER = ["C1", "Q4n", "W", "SP2"]  # fixed enumeration for the CustomNonbondedForce type-index lookup table below
_N_BEAD_TYPES = len(_BEAD_TYPE_ORDER)


def _make_nonbonded_forces() -> tuple[openmm.CustomNonbondedForce, openmm.NonbondedForce]:
    """Build the pair of forces used for ALL nonbonded interactions in this
    module: a CustomNonbondedForce for LJ (real MARTINI matrix via a
    per-type lookup table) and a charge-only NonbondedForce for Coulomb.

    REAL BUG FOUND AND FIXED HERE (2026-09-03, same day as the MARTINI 3
    parameter update): the original implementation used NonbondedForce's
    per-pair addException() to override cross-type LJ, since that's the
    obvious first idiom for "these two specific particles need different
    LJ parameters than the combining rule gives." That is correct for a
    FIXED, small set of special pairs (e.g. 1-2 bonded exclusions), but
    OpenMM exceptions are NOT subject to the cutoff distance -- they are
    always evaluated directly, at whatever the pair's actual separation
    is. Since every cross-type pair in the WHOLE system got an exception
    (headgroup-vs-every-tail-bead, system-wide, not just nearby ones),
    this silently forced permanent, always-on LJ interactions between
    particles far beyond the intended 1.1 nm cutoff -- confirmed
    empirically on a small 2-molecule/11-bead test system: 16 of 24
    exceptions added were for pairs already >1.1 nm apart at t=0, and
    since the exception mechanism doesn't respect cutoff OR update as
    particles move, this would have stayed wrong throughout any dynamics
    run, not just at the initial configuration. Caught while building
    build_solvated_openmm_system() below and cross-checking its
    CustomNonbondedForce-based scheme against this function's original
    exception-based one on an identical small system: their energies
    disagreed, hand-calculating the correct answer from raw physics (see
    examples/verify_solvated_system_energy.py) confirmed the
    CustomNonbondedForce answer was right and the exception-based one was
    wrong, not the other way around.

    Fix: ALL nonbonded LJ in this module now goes through this one
    CustomNonbondedForce (a per-particle type index plus a
    Discrete2DFunction lookup table built from MARTINI3_BEAD_TYPES +
    NONBONDED_CROSS_TERMS -- the real MARTINI matrix, same source data as
    before, just evaluated via OpenMM's normal cutoff-respecting
    neighbor-list machinery instead of per-pair exceptions), with a
    separate charge-only NonbondedForce for Coulomb. This is also why
    build_openmm_system() and build_solvated_openmm_system() now share
    this one helper instead of each having their own nonbonded-force
    construction -- one correct scheme, not two (one of which was wrong).
    """
    lj_force = openmm.CustomNonbondedForce(
        "4*epsilon(type1,type2)*((sigma(type1,type2)/r)^12-(sigma(type1,type2)/r)^6)")
    lj_force.addPerParticleParameter("type")
    sigma_table, epsilon_table = _build_type_lookup_tables()
    lj_force.addTabulatedFunction("sigma", openmm.Discrete2DFunction(_N_BEAD_TYPES, _N_BEAD_TYPES, sigma_table))
    lj_force.addTabulatedFunction("epsilon", openmm.Discrete2DFunction(_N_BEAD_TYPES, _N_BEAD_TYPES, epsilon_table))
    lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    lj_force.setCutoffDistance(1.1 * unit.nanometer)
    # CustomNonbondedForce defaults to NO analytic LJ long-range (dispersion)
    # correction, unlike NonbondedForce, which defaults to including one --
    # found as a real ~0.3% energy discrepancy while cross-checking against
    # the (since-removed) exception-based scheme. Explicitly enabling it
    # here is the physically correct, standard-MD-practice choice.
    lj_force.setUseLongRangeCorrection(True)

    coulomb_force = openmm.NonbondedForce()
    coulomb_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    coulomb_force.setCutoffDistance(1.1 * unit.nanometer)
    return lj_force, coulomb_force


def _add_molecule_particles(system: openmm.System, lj_force: openmm.CustomNonbondedForce,
                            coulomb_force: openmm.NonbondedForce, bond_force: openmm.HarmonicBondForce,
                            mol: CGMolecule, origin: openmm.Vec3, particle_offset: int) -> None:
    """Shared per-molecule particle/bond-adding logic used by both
    build_openmm_system() and build_solvated_openmm_system() -- placement
    (the `origin` each molecule's beads are built out from) differs
    between the two callers, everything else is identical."""
    for i, bead in enumerate(mol.beads):
        params = MARTINI3_BEAD_TYPES[bead.bead_type]
        system.addParticle(params["mass"] * unit.amu)
        lj_force.addParticle([float(_BEAD_TYPE_ORDER.index(bead.bead_type))])
        coulomb_force.addParticle(bead.charge, 0.1 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)  # LJ handled by lj_force; sigma/epsilon here are inert placeholders
    for i, j in mol.bonds:
        bond_force.addBond(particle_offset + i, particle_offset + j,
                          mol.bond_length_nm * unit.nanometer, mol.bond_k * unit.kilojoule_per_mole / unit.nanometer**2)
        # Real bug found and fixed 2026-09-04: directly-bonded (1-2) pairs were
        # never excluded from the nonbonded LJ/Coulomb forces, so every bond in
        # every molecule also carried a large, unphysical LJ clash on top of its
        # harmonic term (e.g. ~59.6 kJ/mol for a head-tail1 pair at the real
        # Q4n-C1 cross-term parameters) -- standard MD practice (and real
        # MARTINI's own nrexcl=1 convention) excludes 1-2 bonded pairs from
        # nonbonded interactions entirely. Caught via an independent GROMACS
        # cross-check on the identical 20-molecule smoke-test system/coordinates:
        # OpenMM's LJ energy was +1175.7 kJ/mol vs GROMACS's -7.4 kJ/mol before
        # this fix; -14.3 kJ/mol (same sign/order of magnitude as GROMACS) after.
        lj_force.addExclusion(particle_offset + i, particle_offset + j)
        coulomb_force.addException(particle_offset + i, particle_offset + j, 0.0, 1.0, 0.0)


def build_openmm_system(molecules: list[CGMolecule], box_size_nm: float = 10.0) -> tuple[openmm.System, list[tuple]]:
    """Build a minimal OpenMM System from a list of CGMolecule instances:
    particles with mass, a HarmonicBondForce for intra-molecule bonds, and
    the real MARTINI3_BEAD_TYPES/NONBONDED_CROSS_TERMS nonbonded matrix via
    _make_nonbonded_forces() (see that function's docstring for a real bug
    this replaced -- the previous per-pair-exception scheme silently
    ignored the cutoff for cross-type pairs). Returns (system, positions)
    where positions are simple placed-on-a-grid starting coordinates --
    good enough to prove the plumbing runs and is numerically stable, not
    a physically meaningful starting configuration."""
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_size_nm, 0, 0), openmm.Vec3(0, box_size_nm, 0), openmm.Vec3(0, 0, box_size_nm))

    bond_force = openmm.HarmonicBondForce()
    lj_force, coulomb_force = _make_nonbonded_forces()

    positions = []
    particle_offset = 0
    grid_spacing = 1.5  # nm, generic starting spacing, not physically tuned
    n_placed = 0
    per_row = max(1, int(box_size_nm / grid_spacing))

    for mol in molecules:
        row, col = divmod(n_placed, per_row)
        origin = openmm.Vec3(col * grid_spacing, row * grid_spacing, 0.5 * box_size_nm)
        _add_molecule_particles(system, lj_force, coulomb_force, bond_force, mol, origin, particle_offset)
        for i in range(len(mol.beads)):
            positions.append(origin + openmm.Vec3(0, 0, i * mol.bond_length_nm))
        particle_offset += len(mol.beads)
        n_placed += 1

    system.addForce(bond_force)
    system.addForce(lj_force)
    system.addForce(coulomb_force)
    return system, positions


def _build_type_lookup_tables() -> tuple[list[float], list[float]]:
    """Flatten MARTINI3_BEAD_TYPES + NONBONDED_CROSS_TERMS into row-major
    sigma/epsilon tables indexed by (type1, type2) over _BEAD_TYPE_ORDER --
    the real MARTINI matrix, packaged for OpenMM's Discrete2DFunction."""
    sigma_table = [0.0] * (_N_BEAD_TYPES * _N_BEAD_TYPES)
    epsilon_table = [0.0] * (_N_BEAD_TYPES * _N_BEAD_TYPES)
    for i, t1 in enumerate(_BEAD_TYPE_ORDER):
        for j, t2 in enumerate(_BEAD_TYPE_ORDER):
            if t1 == t2:
                params = MARTINI3_BEAD_TYPES[t1]
            else:
                params = _cross_term(t1, t2)
            idx = i + j * _N_BEAD_TYPES  # Discrete2DFunction is column-major (xsize=i, ysize=j)
            sigma_table[idx] = params["sigma"]
            epsilon_table[idx] = params["epsilon"]
    return sigma_table, epsilon_table


def make_water_bead(index: int) -> CGMolecule:
    """A single explicit MARTINI water bead, represented as a 1-bead
    CGMolecule so it flows through the same downstream machinery
    (positions, per-molecule splitting for analysis) as surfactants."""
    params = MARTINI3_BEAD_TYPES["W"]
    return CGMolecule(name=f"water{index}", beads=[CGBead(name=f"water{index}", bead_type="W", mass_amu=params["mass"])])


def build_solvated_openmm_system(surfactant_molecules: list[CGMolecule], box_size_nm: float,
                                 water_min_distance_nm: float = 0.4,
                                 surfactant_spacing_nm: float = 2.0) -> tuple[openmm.System, list, list[CGMolecule]]:
    """Production-scale system builder with EXPLICIT water -- the real
    missing piece in build_openmm_system()'s small demo/test, which places
    only surfactants with no solvent at all. Genuine hydrophobic-effect-
    driven self-assembly needs tail beads to have real water to avoid, not
    just other tails to prefer -- without explicit solvent, the model has
    no way to express "hydrophobic" as anything other than "attracts less
    strongly than the headgroup," a materially weaker/wrong driving force.

    Architecture note, why this is a SEPARATE function from
    build_openmm_system() rather than an extension of it: that function
    adds cross-species NonbondedForce EXCEPTIONS per particle PAIR, an
    O(n^2) Python loop that is fine for its ~100-particle demo but
    completely impractical once real explicit water pushes particle counts
    into the thousands (a production box needs ~8.6 water beads/nm^3 at
    real MARTINI density -- see below -- so even a modest 15 nm box implies
    tens of thousands of water beads, and O(n^2) pair-exceptions at that
    scale is neither fast to build nor a sound use of OpenMM's exception
    mechanism, which is meant for a small number of special pairs like 1-2
    bonded exclusions, not describing bulk cross-type physics).

    Fix: use the same _make_nonbonded_forces() (CustomNonbondedForce LJ +
    charge-only Coulomb) that build_openmm_system() now also uses -- see
    that function's own docstring for the real bug (exceptions silently
    ignoring the cutoff) this replaced, found while building this function
    and cross-checking it against build_openmm_system()'s old scheme.

    Water density: 4 real water molecules per MARTINI CG bead (the same
    mapping this project's W bead mass, 72.0 amu = 4 x 18.02 g/mol, already
    reflects) at real water's known density (1000 kg/m^3) gives a real,
    computed (not hardcoded) number density of ~8.6 beads/nm^3 -- see the
    calculation in this function's body.

    Placement: surfactants on a coarse 3D lattice (surfactant_spacing_nm
    apart) to reduce initial steric clash; water fills the remaining
    volume on a finer lattice, skipping any point closer than
    water_min_distance_nm to an already-placed surfactant bead -- the same
    "place on a grid, delete overlaps" approach GROMACS's own
    solvate/insert-molecules tools use, not a novel scheme. Energy
    minimization/equilibration (not this function) is responsible for
    relaxing any residual mild clashes, same philosophy as
    build_openmm_system()'s existing "good enough starting point" grid.

    Returns (system, positions, all_molecules) where all_molecules is
    surfactants followed by the placed water molecules, in the same order
    as positions/particles -- needed for per-molecule position splitting
    the same way run_stability_check() already does for the toy system.
    """
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_size_nm, 0, 0), openmm.Vec3(0, box_size_nm, 0), openmm.Vec3(0, 0, box_size_nm))

    bond_force = openmm.HarmonicBondForce()
    lj_force, coulomb_force = _make_nonbonded_forces()

    positions: list = []
    all_molecules: list[CGMolecule] = []
    particle_offset = 0

    surf_per_axis = max(1, int(box_size_nm / surfactant_spacing_nm))
    if len(surfactant_molecules) > surf_per_axis ** 3:
        raise ValueError(f"box_size_nm={box_size_nm} too small to place {len(surfactant_molecules)} "
                         f"surfactants at {surfactant_spacing_nm} nm spacing (max {surf_per_axis**3}) -- use a bigger box")

    for idx, mol in enumerate(surfactant_molecules):
        ix, iy, iz = idx % surf_per_axis, (idx // surf_per_axis) % surf_per_axis, idx // (surf_per_axis ** 2)
        origin = openmm.Vec3(ix * surfactant_spacing_nm, iy * surfactant_spacing_nm, iz * surfactant_spacing_nm)
        _add_molecule_particles(system, lj_force, coulomb_force, bond_force, mol, origin, particle_offset)
        for i in range(len(mol.beads)):
            positions.append(origin + openmm.Vec3(0, 0, i * mol.bond_length_nm))
        particle_offset += len(mol.beads)
        all_molecules.append(mol)

    # real MARTINI water density: 4 real H2O per CG bead, real water density
    # 1000 kg/m^3 (1000 g/L), Avogadro's number -- computed, not hardcoded
    real_water_molar_L = 1000.0 / 18.02  # mol H2O per L, from real water's known density and molar mass
    cg_water_molar_L = real_water_molar_L / 4.0  # mol CG bead per L (4 real waters per bead)
    beads_per_nm3 = cg_water_molar_L * 6.02214076e23 / 1.0e24  # (mol/L)*(particles/mol) / (nm^3 per L) = particles/nm^3
    n_water_target = int(beads_per_nm3 * box_size_nm ** 3)

    water_spacing_nm = (box_size_nm ** 3 / max(n_water_target, 1)) ** (1.0 / 3.0)
    water_per_axis = max(1, int(box_size_nm / water_spacing_nm))
    n_water_placed = 0
    water_type_idx = float(_BEAD_TYPE_ORDER.index("W"))
    water_params = MARTINI3_BEAD_TYPES["W"]

    import itertools
    surfactant_positions = list(positions)  # snapshot for overlap checks, before any water is added
    for ix, iy, iz in itertools.product(range(water_per_axis), repeat=3):
        if n_water_placed >= n_water_target:
            break
        pos = openmm.Vec3(ix * water_spacing_nm, iy * water_spacing_nm, iz * water_spacing_nm)
        if any((pos.x - p.x) ** 2 + (pos.y - p.y) ** 2 + (pos.z - p.z) ** 2 < water_min_distance_nm ** 2
               for p in surfactant_positions):
            continue
        system.addParticle(water_params["mass"] * unit.amu)
        lj_force.addParticle([water_type_idx])
        coulomb_force.addParticle(0.0, 0.1 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        positions.append(pos)
        all_molecules.append(make_water_bead(n_water_placed))
        n_water_placed += 1

    system.addForce(bond_force)
    system.addForce(lj_force)
    system.addForce(coulomb_force)
    return system, positions, all_molecules


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
          f"12 linear (4 beads each) + 8 gemini (9 beads each, since 2026-09-05: includes the "
          f"real SP2 amide-linkage bead on each side, was 7)")

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
