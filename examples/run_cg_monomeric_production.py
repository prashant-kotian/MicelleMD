"""Real production-scale CG self-assembly run for the real monomeric
amidoamine comparator, C14-Et_plus, per the real collaborator DFT
requirement document (`H:\\download folder from samsung laptop
01082026\\DFT requirement.pdf`): tail-amide-N+(CH3)2(CH2CH3), real SMILES
CCCCCCCCCCCCCC(=O)NCCC[N+](C)(C)CC, C14 acyl tail -- the "architecture
effect: monomeric versus gemini" comparator paired with C14-s3_2plus
(run_cg_gemini_production.py).

Mirrors run_cg_production.py/run_cg_gemini_production.py's design
(same discipline, differences stated below):
- make_monomeric_amidoamine_surfactant() instead of make_linear_surfactant()
  -- real tail-amide-head topology (see that function's own docstring for
  why make_linear_surfactant()'s direct head-tail bond is wrong for this
  molecule's real amidoamine chemistry), using the real, already-sourced
  SP2 amide + Q1 cationic headgroup beads.
- n_tail_beads=3: same real chain-length-to-bead convention already used
  for the gemini run (C14's 13-carbon alkyl tail, excluding the carbonyl
  carbon the SP2 bead already represents, ~4 carbons/bead -> 3 beads).
- N_SURFACTANTS=150, BOX_SIZE_NM=15.0, SURFACTANT_SPACING_NM=2.5: this
  molecule is only 5 beads (~1.88 nm chain, 4 bonds x 0.47 nm) -- much
  closer to SDS's own 4-bead/~1.41 nm chain than the gemini's 11-bead/
  ~4.7 nm one, so SDS's own validated scale (150 molecules, 15 nm box) is
  reused directly rather than shrunk. Spacing bumped from SDS's 2.0 nm to
  2.5 nm as a real, deliberate safety margin (1.88 nm chain leaves only a
  thin gap at 2.0 nm) -- verified by smoke test below, not assumed safe.

Usage:
  python run_cg_monomeric_production.py --smoke-test    # tiny system/steps, verifies setup only
  python run_cg_monomeric_production.py --benchmark-only # measure steps/sec and exit
  python run_cg_monomeric_production.py                  # benchmark, then launch the real run
  python run_cg_monomeric_production.py --resume          # continue from the last checkpoint
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
from cg_model import make_monomeric_amidoamine_surfactant, build_solvated_openmm_system
from analysis import find_aggregates, aggregation_number_distribution, radius_of_gyration, unwrap_cluster_positions

OUTPUT_DIR = Path(__file__).resolve().parent / "cg_monomeric_c14_results"
CHECKPOINT_PATH = OUTPUT_DIR / "production.chk"
LOG_PATH = OUTPUT_DIR / "production_log.jsonl"

N_SURFACTANTS = 150
N_TAIL_BEADS = 3
BEADS_PER_MOLECULE = N_TAIL_BEADS + 1 + 1  # tail + amide + head = 5
BOX_SIZE_NM = 15.0
SURFACTANT_SPACING_NM = 2.5
TIMESTEP_PS = 0.02
FRICTION_PER_PS = 1.0
TEMPERATURE_K = 300.0
SNAPSHOT_INTERVAL_STEPS = 25_000  # 25000 * 0.02 ps = 500 ps between snapshots
AGGREGATE_CUTOFF_NM = 0.6


def build_production_system(n_surfactants: int = N_SURFACTANTS, box_size_nm: float = BOX_SIZE_NM):
    surfactants = [make_monomeric_amidoamine_surfactant(f"mono{i}", n_tail_beads=N_TAIL_BEADS)
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
    for mol in molecules[:n_surfactants]:
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


def run_benchmark(system, positions, n_steps: int = 500) -> tuple[float, float]:
    context, integrator = make_context(system)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
    context.getState(getEnergy=True)
    t0 = time.time()
    integrator.step(n_steps)
    context.getState(getEnergy=True)
    elapsed = time.time() - t0
    steps_per_sec = n_steps / elapsed
    ns_per_day = steps_per_sec * TIMESTEP_PS / 1000.0 * 86400.0
    return steps_per_sec, ns_per_day


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-ns", type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.smoke_test:
        print("--- SMOKE TEST: small system, verifying setup only ---")
        system, positions, molecules = build_production_system(n_surfactants=10, box_size_nm=8.0)
        n_water = len(molecules) - 10
        print(f"Built: {system.getNumParticles()} total particles "
              f"(10 monomeric x {BEADS_PER_MOLECULE} beads = {10*BEADS_PER_MOLECULE} surfactant beads, {n_water} water beads)")
        context, integrator = make_context(system)
        context.setPositions(positions)
        context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
        pe0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"Initial PE: {pe0:.1f} kJ/mol")
        integrator.step(2000)
        pe1 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"PE after 2000 steps (40 ps): {pe1:.1f} kJ/mol")
        assert pe1 == pe1, "NaN energy -- setup is broken"
        assert abs(pe1) < 1e8, "energy blew up -- setup is broken"
        snap = snapshot_aggregates(context, molecules, 10, 8.0)
        print(f"Aggregation snapshot: {snap}")
        print("SMOKE TEST PASSED.")
        return

    print(f"Building production system: {N_SURFACTANTS} monomeric C14-Et_plus surfactants "
          f"({BEADS_PER_MOLECULE} beads each), {BOX_SIZE_NM} nm box...")
    t0 = time.time()
    system, positions, molecules = build_production_system()
    n_water = len(molecules) - N_SURFACTANTS
    print(f"Built in {time.time()-t0:.1f}s: {system.getNumParticles()} total particles "
          f"({N_SURFACTANTS} surfactants x {BEADS_PER_MOLECULE} beads = {N_SURFACTANTS*BEADS_PER_MOLECULE} beads, "
          f"{n_water} explicit water beads)")

    print("\nBenchmarking real throughput on this machine (CPU platform, 500 steps)...")
    steps_per_sec, ns_per_day = run_benchmark(system, positions, n_steps=500)
    print(f"Benchmark: {steps_per_sec:.1f} steps/sec = {ns_per_day:.2f} ns/day")

    if args.benchmark_only:
        return

    if args.total_ns is not None:
        total_ns = args.total_ns
    else:
        target_wall_hours = 11.0
        total_ns = ns_per_day * (target_wall_hours / 24.0)
    total_steps = int(total_ns * 1000.0 / TIMESTEP_PS)
    est_wall_hours = total_steps / steps_per_sec / 3600.0
    print(f"\nPlanned production run: {total_ns:.2f} ns ({total_steps:,} steps), "
          f"estimated wall time {est_wall_hours:.1f} h at the measured benchmark rate")

    context, integrator = make_context(system)

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
        print(f"Resumed from checkpoint at step {start_step:,} (per the log's last entry)")
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
