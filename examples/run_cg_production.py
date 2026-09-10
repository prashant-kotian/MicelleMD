"""Real production-scale CG self-assembly run using the now-fixed,
now-explicit-solvent nonbonded scheme (build_solvated_openmm_system(),
_make_nonbonded_forces() -- see cg_model.py and ROADMAP.md for the real
bug found and fixed in this scheme earlier the same day this script was
written, and why explicit water matters for genuine hydrophobic-effect
self-assembly).

Design, each choice stated rather than left implicit:
- 150 linear SDS-analog surfactants (1x Q4n head + 3x C1 tail, the same
  architecture Vainikka et al. 2021's real MARTINI SDS model uses, and
  this project's own validated Bales et al. 1998 N_agg sanity check
  target, ~44.8-54.2) -- enough total molecules that even a single real
  micelle-sized aggregate wouldn't exhaust the population, unlike the
  20-molecule atomistic run where the largest stable cluster (9) was
  capped by simply running out of monomers.
- Real explicit MARTINI water at real density (see
  build_solvated_openmm_system()) -- required for genuine hydrophobic-
  effect-driven aggregation, not decorative.
- CPU platform, not Reference (Reference is a slow, correctness-only
  reference implementation not meant for production -- confirmed
  available on this machine via Platform.getPlatformByName('CPU')).
- Resumable and checkpointed, matching this project's established
  pattern for long unattended runs (the atomistic GROMACS route's own
  "-cpi checkpoint, safe to interrupt" precedent): periodic binary
  checkpoints via context.createCheckpoint(), periodic aggregate-
  distribution snapshots written to a JSON log as the run progresses
  (not just a final-frame result), so progress is visible without
  waiting for completion and nothing is lost if interrupted.
- A short real benchmark run happens FIRST to measure actual throughput
  on this machine before committing to a total step count -- the same
  "benchmark before committing to a run schedule" discipline the
  atomistic route used (36 ns/day on 4 threads, measured, not assumed).

Usage:
  python run_cg_production.py --benchmark-only   # measure steps/sec and exit
  python run_cg_production.py                    # benchmark, then launch the real run
  python run_cg_production.py --resume            # continue from the last checkpoint
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
from cg_model import make_linear_surfactant, build_solvated_openmm_system
from analysis import find_aggregates, aggregation_number_distribution, radius_of_gyration, unwrap_cluster_positions

OUTPUT_DIR = Path(__file__).resolve().parent / "cg_production_results"
CHECKPOINT_PATH = OUTPUT_DIR / "production.chk"
LOG_PATH = OUTPUT_DIR / "production_log.jsonl"

N_SURFACTANTS = 150
BOX_SIZE_NM = 15.0
TIMESTEP_PS = 0.02  # matches this module's existing convention (LangevinMiddleIntegrator, run_stability_check)
FRICTION_PER_PS = 1.0
TEMPERATURE_K = 300.0
SNAPSHOT_INTERVAL_STEPS = 25_000  # 25000 * 0.02 ps = 500 ps between snapshots
AGGREGATE_CUTOFF_NM = 0.6  # same cutoff already used/validated in the atomistic route's own analysis


def build_production_system():
    surfactants = [make_linear_surfactant(f"sds{i}", n_tail_beads=3) for i in range(N_SURFACTANTS)]
    system, positions, molecules = build_solvated_openmm_system(surfactants, BOX_SIZE_NM)
    return system, positions, molecules


def make_context(system, platform_name: str = "CPU") -> tuple[openmm.Context, openmm.Integrator]:
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond, TIMESTEP_PS * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    return context, integrator


def snapshot_aggregates(context, molecules, n_surfactants: int) -> dict:
    state = context.getState(getPositions=True)
    positions_nm = [p.value_in_unit(unit.nanometer) for p in state.getPositions()]
    per_molecule = []
    offset = 0
    for mol in molecules[:n_surfactants]:  # only surfactants matter for aggregation, water excluded
        n = len(mol.beads)
        per_molecule.append([tuple(positions_nm[offset + i]) for i in range(n)])
        offset += n
    aggregates = find_aggregates(per_molecule, cutoff_nm=AGGREGATE_CUTOFF_NM, box_size_nm=BOX_SIZE_NM)
    dist = aggregation_number_distribution(aggregates)
    largest = max(aggregates, key=len)
    # REAL BUG FIXED 2026-09-06: this used to feed raw, still-wrapped
    # positions directly into radius_of_gyration(), silently producing
    # physically implausible Rg whenever the largest aggregate straddled
    # the periodic box boundary (observed live: ~11.2-11.3 nm sustained in
    # a 15 nm box for a 21-molecule aggregate that should be ~1-2 nm) --
    # see unwrap_cluster_positions()'s own docstring in analysis.py.
    largest_positions = unwrap_cluster_positions([per_molecule[mol_idx] for mol_idx in largest], BOX_SIZE_NM)
    largest_rg = radius_of_gyration(largest_positions)
    return {
        "n_aggregates": len(aggregates),
        "aggregation_number_distribution": {str(k): v for k, v in dist.items()},
        "largest_aggregate_size": len(largest),
        "largest_aggregate_rg_nm": largest_rg,
    }


def run_benchmark(system, positions, n_steps: int = 500) -> float:
    context, integrator = make_context(system)
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
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Building production system: {N_SURFACTANTS} linear SDS-analog surfactants, {BOX_SIZE_NM} nm box...")
    t0 = time.time()
    system, positions, molecules = build_production_system()
    n_water = len(molecules) - N_SURFACTANTS
    print(f"Built in {time.time()-t0:.1f}s: {system.getNumParticles()} total particles "
          f"({N_SURFACTANTS} surfactants x 4 beads = {N_SURFACTANTS*4} beads, {n_water} explicit water beads)")

    print("\nBenchmarking real throughput on this machine (CPU platform, 500 steps)...")
    steps_per_sec, ns_per_day = run_benchmark(system, positions, n_steps=500)
    print(f"Benchmark: {steps_per_sec:.1f} steps/sec = {ns_per_day:.2f} ns/day")

    if args.benchmark_only:
        return

    if args.total_ns is not None:
        total_ns = args.total_ns
    else:
        # target a real, honestly-scoped unattended wall time (~10-12h),
        # matching the atomistic route's own ~10h campaign, not an
        # arbitrarily large number picked without reference to measured speed
        target_wall_hours = 11.0
        total_ns = ns_per_day * (target_wall_hours / 24.0)
    total_steps = int(total_ns * 1000.0 / TIMESTEP_PS)
    est_wall_hours = total_steps / steps_per_sec / 3600.0
    print(f"\nPlanned production run: {total_ns:.2f} ns ({total_steps:,} steps), "
          f"estimated wall time {est_wall_hours:.1f} h at the measured benchmark rate")

    context, integrator = make_context(system)

    if args.resume and CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        context.setPositions(positions)  # required before loadCheckpoint to establish particle count/topology
        with open(CHECKPOINT_PATH, "rb") as f:
            context.loadCheckpoint(f.read())
        # the checkpoint itself doesn't store a step count, but the JSONL
        # log's last entry does -- read it back so total_steps below means
        # "stop at this many steps total" consistently across a resume,
        # not "run total_steps AGAIN on top of whatever the checkpoint
        # already had" (a real, if harmless, bookkeeping bug in an earlier
        # version of this script: it always started counting from 0 on
        # resume, silently running longer than --total-ns actually said)
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
        LOG_PATH.write_text("")  # fresh run: clear any stale log

    steps_done = start_step
    run_t0 = time.time()
    with open(LOG_PATH, "a") as log_f:
        while steps_done < total_steps:
            chunk = min(SNAPSHOT_INTERVAL_STEPS, total_steps - steps_done)
            integrator.step(chunk)
            steps_done += chunk

            state = context.getState(getEnergy=True)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            snap = snapshot_aggregates(context, molecules, N_SURFACTANTS)
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
