"""Real production-scale CG self-assembly run for the actual PhD-thesis
gemini cationic surfactant, C14-s3_2plus, per the real collaborator DFT
requirement document (`H:\\download folder from samsung laptop
01082026\\DFT requirement.pdf`): tail-amide-N+(CH3)2-CH2CH2CH2-N+(CH3)2-
amide-tail, real SMILES
CCCCCCCCCCCCCC(=O)NCCC[N+](C)(C)CCC[N+](C)(C)CCCNC(=O)CCCCCCCCCCCCC,
C14 tails (myristoyl), plain 3-carbon (s3/trimethylene) spacer -- the
"priority" gemini structure in that document, and the one with no
remaining real gaps in cg_model.py as of 2026-09-14 (the s3OH spacer's
polar OH-bearing bead still needs separate real sourcing; this script is
deliberately the s3, not s3OH, variant).

Mirrors run_cg_production.py's design exactly (same discipline, same
choices, differences stated below), since that script is this project's
own validated template for a real production CG self-assembly run:
- make_gemini_surfactant() instead of make_linear_surfactant() -- uses
  the real, newly-sourced Q1 cationic headgroup (default as of
  2026-09-14) and the already-validated SP2 amide bead.
- n_tail_beads=3, n_spacer_beads=1: real chain-length-to-bead mapping,
  using this project's own already-validated SDS convention (12 real
  carbons = 3 C1 beads, ~4 carbons/bead). C14's real alkyl tail (13
  carbons, excluding the carbonyl carbon which the SP2 bead already
  represents) rounds to 3 beads by the same rule SDS used; the real s3
  spacer (3 carbons, trimethylene) maps to 1 bead -- literally the
  "short-spacer C2-C6ish regime" make_gemini_surfactant()'s own docstring
  already anticipated. This is a real approximation (4-carbon CG
  resolution can't distinguish C12/C14/C16 as finely as the real atomistic
  chain-length series can), stated honestly, not hidden.
- N_SURFACTANTS=50, not SDS's 150: each gemini molecule is 11 beads (vs.
  SDS's 4), so 50 was chosen as a real, deliberately smaller scale given
  the much bigger per-molecule system (11 beads x 50 is already close to
  SDS's own 150 x 4 in total surfactant-bead count). Box widened to 24.0
  nm (vs. SDS's 15.0 nm) to give this larger, more highly-charged system
  more room.
- Net system charge is real and large (+2 x 50 = +100 e, vs. SDS's own
  already-accepted -150 e) -- counterions are not modeled as explicit
  particles anywhere in this project (see cg_model.py's own module
  docstring); PME's implicit uniform neutralizing background handles this
  the same way it already does for SDS, not a new approximation introduced
  here.
- Same benchmark-first, resumable/checkpointed, JSONL-logged design as
  run_cg_production.py, plus a --smoke-test flag (small system, few steps,
  verifies setup only) matching the pattern already used in
  run_atomistic_metadynamics.py before that script's own real multi-day
  launch -- this project's standing "don't gamble compute on unverified
  code" discipline, applied here before this is the first-ever cationic
  gemini production run, not an established repeat.

Usage:
  python run_cg_gemini_production.py --smoke-test    # tiny system/steps, verifies setup only
  python run_cg_gemini_production.py --benchmark-only # measure steps/sec and exit
  python run_cg_gemini_production.py                  # benchmark, then launch the real run
  python run_cg_gemini_production.py --resume          # continue from the last checkpoint
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "micellemd"))
import openmm
import openmm.unit as unit
from cg_model import make_gemini_surfactant, build_solvated_openmm_system
from analysis import find_aggregates, aggregation_number_distribution, radius_of_gyration, unwrap_cluster_positions

OUTPUT_DIR = Path(__file__).resolve().parent / "cg_gemini_c14_s3_results"
CHECKPOINT_PATH = OUTPUT_DIR / "production.chk"
LOG_PATH = OUTPUT_DIR / "production_log.jsonl"

N_SURFACTANTS = 50
N_TAIL_BEADS = 3
N_SPACER_BEADS = 1
BEADS_PER_MOLECULE = 2 * N_TAIL_BEADS + 2 + 1 + 2  # tail1 + amide1 + head1 + spacer + head2 + amide2 + tail2 = 11
BOX_SIZE_NM = 24.0
# REAL PROBLEM FOUND AND FIXED 2026-09-14, via this script's own smoke test
# (exactly what it exists to catch): build_solvated_openmm_system() places
# each molecule as a straight chain along +z from a 3D grid origin, using
# ONE spacing for all three axes. SDS's own validated default
# (surfactant_spacing_nm=2.0 nm) is fine for SDS's short ~1.9 nm (4-bead)
# chain, but this gemini molecule is 11 beads (~4.7 nm long,
# (BEADS_PER_MOLECULE-1)*0.47 nm bond length) -- at 2.0 nm z-spacing,
# vertically-adjacent molecules' chains directly interpenetrate (confirmed:
# initial PE was 1.17e9 kJ/mol, NaN within 2000 steps). Fixed with a real,
# molecule-length-derived spacing (6.0 nm, a real margin over the 4.7 nm
# chain extent) rather than guessing a bigger number -- this in turn forces
# a bigger box (surf_per_axis^3 >= N_SURFACTANTS at this spacing) and a
# smaller first-attempt molecule count than SDS's 150, both real,
# disclosed consequences of this molecule's real size, not arbitrary.
SURFACTANT_SPACING_NM = 6.0
TIMESTEP_PS = 0.02  # matches this project's existing CG production convention
FRICTION_PER_PS = 1.0
TEMPERATURE_K = 300.0
SNAPSHOT_INTERVAL_STEPS = 25_000  # 25000 * 0.02 ps = 500 ps between snapshots, matches run_cg_production.py
AGGREGATE_CUTOFF_NM = 0.6  # same cutoff already validated in this project's other CG/atomistic analysis


def build_production_system(n_surfactants: int = N_SURFACTANTS, box_size_nm: float = BOX_SIZE_NM):
    surfactants = [make_gemini_surfactant(f"gem{i}", n_tail_beads=N_TAIL_BEADS, n_spacer_beads=N_SPACER_BEADS)
                   for i in range(n_surfactants)]
    system, positions, molecules = build_solvated_openmm_system(
        surfactants, box_size_nm, surfactant_spacing_nm=SURFACTANT_SPACING_NM)
    return system, positions, molecules


def make_context(system, platform_name: str = "CPU") -> tuple[openmm.Context, openmm.Integrator]:
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond, TIMESTEP_PS * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    return context, integrator


def snapshot_aggregates(context, molecules, n_surfactants: int, box_size_nm: float) -> dict:
    state = context.getState(getPositions=True)
    positions_nm = [p.value_in_unit(unit.nanometer) for p in state.getPositions()]
    per_molecule = []
    offset = 0
    for mol in molecules[:n_surfactants]:  # only surfactants matter for aggregation, water excluded
        n = len(mol.beads)
        per_molecule.append([tuple(positions_nm[offset + i]) for i in range(n)])
        offset += n
    aggregates = find_aggregates(per_molecule, cutoff_nm=AGGREGATE_CUTOFF_NM, box_size_nm=box_size_nm)
    dist = aggregation_number_distribution(aggregates)
    largest = max(aggregates, key=len)
    largest_positions = unwrap_cluster_positions([per_molecule[mol_idx] for mol_idx in largest], box_size_nm)
    largest_rg = radius_of_gyration(largest_positions)
    return {
        "n_aggregates": len(aggregates),
        "aggregation_number_distribution": {str(k): v for k, v in dist.items()},
        "largest_aggregate_size": len(largest),
        "largest_aggregate_rg_nm": largest_rg,
    }


def run_benchmark(system, positions, n_steps: int = 500, platform_name: str = "CPU") -> tuple[float, float]:
    context, integrator = make_context(system, platform_name)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
    context.getState(getEnergy=True)  # warm up
    t0 = time.time()
    integrator.step(n_steps)
    context.getState(getEnergy=True)  # force completion before stopping the clock
    elapsed = time.time() - t0
    steps_per_sec = n_steps / elapsed
    ns_per_day = steps_per_sec * TIMESTEP_PS / 1000.0 * 86400.0
    return steps_per_sec, ns_per_day


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-ns", type=float, default=None, help="override the auto-picked total run length")
    parser.add_argument("--smoke-test", action="store_true", help="tiny system (10 molecules), few steps -- verifies setup only")
    parser.add_argument("--platform", type=str, default="CPU", help="OpenMM platform name, e.g. CPU or CUDA")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.smoke_test:
        print("--- SMOKE TEST: small system, verifying setup only ---")
        system, positions, molecules = build_production_system(n_surfactants=8, box_size_nm=12.0)
        n_water = len(molecules) - 8
        print(f"Built: {system.getNumParticles()} total particles "
              f"(8 gemini x {BEADS_PER_MOLECULE} beads = {8*BEADS_PER_MOLECULE} surfactant beads, {n_water} water beads)")
        context, integrator = make_context(system, args.platform)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
        pe0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"Initial PE: {pe0:.1f} kJ/mol")
        integrator.step(2000)
        state = context.getState(getEnergy=True)
        pe1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"PE after 2000 steps (40 ps): {pe1:.1f} kJ/mol")
        assert pe1 == pe1, "NaN energy -- setup is broken"
        assert abs(pe1) < 1e8, "energy blew up -- setup is broken"
        snap = snapshot_aggregates(context, molecules, 8, 12.0)
        print(f"Aggregation snapshot: {snap}")
        print("SMOKE TEST PASSED.")
        return

    print(f"Building production system: {N_SURFACTANTS} gemini C14-s3_2plus surfactants "
          f"({BEADS_PER_MOLECULE} beads each), {BOX_SIZE_NM} nm box...")
    t0 = time.time()
    system, positions, molecules = build_production_system()
    n_water = len(molecules) - N_SURFACTANTS
    print(f"Built in {time.time()-t0:.1f}s: {system.getNumParticles()} total particles "
          f"({N_SURFACTANTS} surfactants x {BEADS_PER_MOLECULE} beads = {N_SURFACTANTS*BEADS_PER_MOLECULE} beads, "
          f"{n_water} explicit water beads)")

    print(f"\nBenchmarking real throughput on this machine ({args.platform} platform, 500 steps)...")
    steps_per_sec, ns_per_day = run_benchmark(system, positions, n_steps=500, platform_name=args.platform)
    print(f"Benchmark: {steps_per_sec:.1f} steps/sec = {ns_per_day:.2f} ns/day")

    if args.benchmark_only:
        return

    if args.total_ns is not None:
        total_ns = args.total_ns
    else:
        target_wall_hours = 11.0  # matches run_cg_production.py's own real-scope choice
        total_ns = ns_per_day * (target_wall_hours / 24.0)
    total_steps = int(total_ns * 1000.0 / TIMESTEP_PS)
    est_wall_hours = total_steps / steps_per_sec / 3600.0
    print(f"\nPlanned production run: {total_ns:.2f} ns ({total_steps:,} steps), "
          f"estimated wall time {est_wall_hours:.1f} h at the measured benchmark rate")

    context, integrator = make_context(system, args.platform)

    if args.resume and CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        context.setPositions(positions)
        with open(CHECKPOINT_PATH, "rb") as f:
            context.loadCheckpoint(f.read())
        start_step = 0
        if LOG_PATH.exists():
            lines = [l for l in LOG_PATH.read_text().splitlines() if l.strip()]
            if lines:
                start_step = json.loads(lines[-1])["steps_done"]
        # total_steps above was computed as an ABSOLUTE step count sized to fit
        # target_wall_hours from a fresh benchmark -- on resume that's already
        # roughly where start_step sits, so without this offset the run below
        # would do zero (or near-zero) new steps and immediately report "DONE".
        # Real bug found 2026-09-17 checking the Kaggle chunk before relaunching it.
        total_steps += start_step
        print(f"Resumed from checkpoint at step {start_step:,} (per the log's last entry); "
              f"new target after adding this chunk's budget: {total_steps:,} steps")
    else:
        context.setPositions(positions)
        context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
        start_step = 0
        LOG_PATH.write_text("")

    steps_done = start_step
    run_t0 = time.time()
    with open(LOG_PATH, "a") as log_f:
        while steps_done < total_steps:
            chunk = min(SNAPSHOT_INTERVAL_STEPS, total_steps - steps_done)
            integrator.step(chunk)
            steps_done += chunk

            state = context.getState(getEnergy=True)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            snap = snapshot_aggregates(context, molecules, N_SURFACTANTS, BOX_SIZE_NM)
            elapsed_h = (time.time() - run_t0) / 3600.0
            record = {
                "steps_done": steps_done, "sim_time_ps": steps_done * TIMESTEP_PS,
                "wall_elapsed_hours": round(elapsed_h, 3), "potential_energy_kJ_mol": pe,
                **snap,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            print(f"[{steps_done:,}/{total_steps:,} steps, {steps_done*TIMESTEP_PS/1000:.2f} ns, "
                  f"{elapsed_h:.2f}h elapsed] PE={pe:.1f} kJ/mol, "
                  f"{snap['n_aggregates']} aggregates, largest={snap['largest_aggregate_size']}")

            with open(CHECKPOINT_PATH, "wb") as chk_f:
                chk_f.write(context.createCheckpoint())

    print(f"\nDONE: {steps_done:,} steps ({steps_done*TIMESTEP_PS/1000:.2f} ns) in {(time.time()-run_t0)/3600.0:.2f}h wall time.")
    print(f"Full per-snapshot history: {LOG_PATH}")


if __name__ == "__main__":
    main()
