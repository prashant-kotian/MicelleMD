"""Cheapest real diagnostic for the chunk-3 resume mystery (see ROADMAP.md,
2026-09-20 entry): the existing hard-fail check in run_cg_gemini_production.py
proves loadCheckpoint() genuinely applies real position data (particle 0
moves from its fresh-grid position to a real checkpoint position) -- yet
chunk 3's very first logged snapshot, 500 ps (25,000 steps) later, showed
fully dispersed monomers instead of the checkpoint's real 3-aggregate state,
even though chunk 3's own FINAL state (115M steps later) ended up identical
to chunk 2's own final distribution.

This script isolates WHERE the wrong reading first appears by snapshotting
the aggregate state at several SHORT intervals (every 2,000 steps, not
run_cg_gemini_production.py's production interval of 25,000) starting
immediately after loadCheckpoint() -- before any new integration at all,
then after 2k/4k/6k/8k/10k steps. Two possible outcomes:
  (a) the snapshot taken with ZERO new integration steps already shows
      dispersed monomers -- this would implicate the analysis/snapshot code
      (find_aggregates / snapshot_aggregates), since the checkpoint's raw
      position data is proven correct by the existing position-only check,
      so a wrong AGGREGATE reading at that exact instant means the
      aggregate-detection logic itself handles loaded-vs-fresh-built
      position arrays differently somehow.
  (b) the t=0 snapshot correctly shows 3 aggregates, and the dispersal
      appears within the following few thousand steps -- this would instead
      implicate something in the force/integration setup specific to a
      resumed run (e.g. stale/inconsistent velocities, or a periodic-image
      wrapping difference between freshly-built and loaded coordinates that
      only manifests once particles start moving).
Only ~10,000 steps total (~5 seconds of raw GPU compute at chunk 2's
measured ~1,965 steps/sec) -- this is deliberately far cheaper than another
full paid session, per this project's "don't gamble compute" discipline.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "micellemd"))
import openmm.unit as unit
from run_cg_gemini_production import (
    build_production_system, make_context, snapshot_aggregates,
    N_SURFACTANTS, BOX_SIZE_NM, TEMPERATURE_K,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "cg_gemini_c14_s3_results"
CHECKPOINT_PATH = OUTPUT_DIR / "production.chk"

SNAPSHOT_STEPS = [0, 2000, 2000, 2000, 2000, 2000]  # cumulative: 0,2k,4k,6k,8k,10k


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"FATAL: no checkpoint at {CHECKPOINT_PATH} -- stage it first (same as the chunk-3 kernel does).")
        sys.exit(1)

    print("Building the same production system (must match the checkpoint's particle layout exactly)...")
    system, positions, molecules = build_production_system()
    context, integrator = make_context(system, "CUDA")
    context.setPositions(positions)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    pre_load_p0 = context.getState(getPositions=True).getPositions()[0]
    with open(CHECKPOINT_PATH, "rb") as f:
        context.loadCheckpoint(f.read())
    post_load_p0 = context.getState(getPositions=True).getPositions()[0]
    if pre_load_p0 == post_load_p0:
        print("FATAL: checkpoint load was a silent no-op (particle 0 position unchanged). Aborting.")
        sys.exit(1)
    print(f"Checkpoint load verified real: particle 0 moved from {pre_load_p0} to {post_load_p0}")

    cumulative_steps = 0
    t0 = time.time()
    for i, chunk in enumerate(SNAPSHOT_STEPS):
        if chunk > 0:
            integrator.step(chunk)
            cumulative_steps += chunk
        snap = snapshot_aggregates(context, molecules, N_SURFACTANTS, BOX_SIZE_NM)
        pe = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        record = {"cumulative_new_steps": cumulative_steps, "potential_energy_kJ_mol": pe, **snap}
        print(f"[t+{cumulative_steps:,} new steps] {json.dumps(record)}")

    print(f"\nDiagnostic finished in {time.time()-t0:.1f}s wall time (excluding system build).")
    print("Expected checkpoint state for comparison: n_aggregates=3, largest_aggregate_size=19, "
          "distribution {15:1, 19:1, 16:1} (chunk 2's real final state).")


if __name__ == "__main__":
    main()
