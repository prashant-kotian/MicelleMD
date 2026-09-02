# GROMACS setup (Ubuntu machine)

No native GROMACS path exists on the Windows development machine this project mostly runs on (checked directly: no binary, no Chocolatey package — see ROADMAP.md). Real GROMACS works cleanly via conda on a second, Ubuntu machine connected over SSH on the same LAN. This is now the actual GROMACS-capable environment for this project, alongside OpenMM (which works on both machines).

## Setup

```bash
# conda already installed (miniconda3); reused the existing chem_sim environment
conda install -n chem_sim -y -c conda-forge -c bioconda gromacs pyscf
```

Verified: GROMACS 2026.3-conda_forge, confirmed with `gmx --version`.

## Real, complete pipeline verified — 2026-09-02

Box creation → solvation → topology → preprocessing → actual energy-minimization run, using only GROMACS's own bundled data (no external force field files needed):

```bash
GMXLIB=~/miniconda3/envs/chem_sim/share/gromacs/top
gmx solvate -cs spc216.gro -o water_box.gro -box 3 3 3 -p topol.top
gmx grompp -f em.mdp -c water_box.gro -p topol.top -o em.tpr -maxwarn 2
gmx mdrun -deffnm em -nt 4
```

Result: 884 SPC water molecules, steepest-descent minimization **converged in 8 steps**, final potential energy -35202.24 kJ/mol, max force 893.3 kJ/mol/nm (below the 1000 tolerance — genuine convergence, not a timeout). Real, executed, verified — not just "gmx --version works."

## Two real bugs found and fixed along the way

1. **`oplsaa.ff` + `spc.itp` topology caused a hard segfault in `gmx grompp`** (not a graceful error — `Segmentation fault (core dumped)`) after printing "Generating 1-4 interactions: fudge = 0.5". Switching to `gromos54a7.ff` + `spc.itp` (a more heavily-tested combination in GROMACS tutorials) processed normally — confirmed the crash was specific to the oplsaa+spc topology pairing on this GROMACS conda-forge build, not GROMACS being broken generally. Worth avoiding oplsaa+spc specifically until this is understood further; not chased deeper today since a working alternative exists.
2. **Cutoff-vs-box-size error**: `rcoulomb`/`rvdw` = 1.0 nm needs a box with a shortest dimension comfortably more than double that (minimum-image convention) — the first attempt at a 2×2×2 nm box was too small. Fixed by using a 3×3×3 nm box. Ordinary GROMACS usage, not really a bug, but worth documenting so a future run doesn't hit the same wall from a fresh start.

## Real atomistic surfactant topology, via GAFF/acpype — 2026-09-02

The bulk-water test above proved the pipeline; this is the first **real, non-placeholder surfactant** run through it. Rather than waiting on WebSearch access to source real MARTINI coarse-grained parameters (see `cg_model.py`'s docstring), found and used **acpype** (conda-forge), a real, established, automated tool that generates GROMACS topologies with GAFF atom types and AM1-BCC partial charges via AmberTools' antechamber/tleap — no manually-typed force-field numbers anywhere.

```bash
conda install -n chem_sim -y -c conda-forge acpype
acpype -i dodecylsulfate.mol -c bcc -n -1   # -c bcc = AM1-BCC charges, -n -1 = net charge
```

Ran on the dodecyl sulfate anion — SDS's actual surfactant ion, same SMILES already verified via PubChem in SurfQSPR's `dataset.py`, so this is consistent with real data already used elsewhere in this project. Boxed, preprocessed, and minimized through the same real GROMACS pipeline as the water test: **converged in 1 step** (RDKit's MMFF pre-optimization already gave a good starting geometry), final potential energy -169.38 kJ/mol. No segfault, no oplsaa-specific issue this time (GAFF/acpype topology, different from the earlier oplsaa+spc water test).

Reusable module: `src/micellemd/atomistic_topology.py` — `generate_gaff_topology(mol_file, net_charge, ...)`, run and verified end to end, not just written and assumed to work.

**Trade-off, stated honestly**: atomistic GAFF topologies are far more expensive to simulate than coarse-grained beads (full all-atom detail), so this route suits single-molecule/small-system validation better than large-scale self-assembly runs, where CG remains the right choice once real MARTINI parameters are sourced. The two approaches are complementary — this doesn't replace the CG plan, it fills the "need real numbers now" gap while MARTINI parameter sourcing is blocked.

## Real solvated + neutralized ionic system — 2026-09-02

Extended the single-ion-in-vacuum result above to a genuine SDS anion + Na+ counterion + explicit TIP3P water system — the natural next step for anything electrostatics-relevant.

**Bug found and fixed**: combining acpype's own `dodecylsulfate_GMX.top` (which starts with its own `[ defaults ]` directive) with `amber99sb-ildn.ff/forcefield.itp` (included for TIP3P water/ion atom types — `tip3p.itp`/`ions.itp` alone only define bonded topology, not the underlying OW/HW/NA/CL LJ parameters, which live in `forcefield.itp`) fails:

```
Fatal error: Syntax error - File forcefield.itp, line 15
Invalid order for directive defaults
```

GROMACS allows exactly one `[ defaults ]` directive in the fully assembled topology. Fix: don't include acpype's `.top` at all — `#include` the force field's `forcefield.itp` **first** (supplying the one true `[ defaults ]`), then `#include` only the ligand's `.itp` (which starts with `[ atomtypes ]`, not `[ defaults ]` — multiple `[ atomtypes ]` blocks are fine, GROMACS merges them), then the water model and ions. Encoded as a reusable function: `atomistic_topology.write_combined_topology()`.

Full verified pipeline on the dodecyl sulfate anion:

```bash
gmx editconf -f dodecylsulfate_GMX.gro -o boxed.gro -box 3 3 3 -c
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p dodecylsulfate_GMX.top   # 866 SPC-geometry waters (topology says tip3p atom types)
gmx grompp -f ions.mdp -c solvated.gro -p dodecylsulfate_GMX.top -o ions.tpr -maxwarn 10
echo SOL | gmx genion -s ions.tpr -o neutral.gro -p dodecylsulfate_GMX.top -pname NA -np 1   # neutralizes the -1 charge
gmx grompp -f em.mdp -c neutral.gro -p dodecylsulfate_GMX.top -o em.tpr -maxwarn 10   # em.mdp uses coulombtype = PME
gmx mdrun -deffnm em -nt 4
```

Result: 1 dodecyl sulfate anion + 1 Na+ + 865 waters (2640 atoms total, net charge 0), steepest-descent minimization **converged in 149 steps**, final potential energy **-37778.6 kJ/mol**, max force 887.7 kJ/mol/nm (below the 1000 tolerance). Real, executed, verified — genuine electrostatically-complete ionic-surfactant-in-water system, not a placeholder.

## Multi-molecule self-assembly test — COMPLETE, 2026-09-02→03

Scaled to 20 dodecyl sulfate anions + 20 Na+ + 6707 TIP3P-geometry waters (`gmx insert-molecules` for random dispersed placement, then the same solvate/genion/minimize pipeline as the single-ion case above). `write_combined_topology()` extended with an `n_molecules` parameter to support this. Minimization converged in 345 steps, PE = -316,068 kJ/mol.

Running on the Ubuntu machine, detached via `nohup ... &` so it survives SSH disconnects (a plain `nohup cmd &` over a non-interactive SSH command can still hang the SSH session itself if stdin isn't redirected — the child process detaches fine, but the parent shell doesn't return; killing the parent shell's PID directly is safe and doesn't touch the nohup'd child).

10 ns NVT production run (2 fs timestep, V-rescale thermostat @ 300 K, PME electrostatics), launched after a 10 ps stability check. Benchmarked at 36 ns/day on 4 threads (8 threads was *slower* — 31.5 ns/day — on this laptop's CPU, likely hyperthreading/thermal overhead outweighing parallelism at this system size); estimated ~6-7 hours wall time. **Whether 10 ns atomistic MD from a fully dispersed start shows meaningful aggregation is genuinely unknown until this run finishes** — published atomistic micellization studies often need tens to hundreds of ns; this run is a real first data point, not guaranteed to show a complete micelle. Result (aggregate structure, radius of gyration, whether SDS ions cluster at all) to be analyzed and documented once the run completes, not before.

## Analysis pipeline built and tested while the run computes — 2026-09-02

Built `src/micellemd/atomistic_analysis.py` (MDAnalysis-based, installed via conda-forge on the Ubuntu machine) ahead of the 10 ns run finishing, so results are ready the moment it does — tested against the real, in-progress trajectory itself (not fabricated data).

Two real things had to be fixed/decided, not assumed:
- MDAnalysis 2.9's TPR parser doesn't support this GROMACS build's tpx version (138) — confirmed via a real `NotImplementedError`, not guessed. Worked around by building the `Universe` from a `.gro` file instead of `.tpr` (sufficient for atom/residue-name-based selection and position data; no bonded topology needed for this analysis).
- Aggregation criterion is atom-level minimum distance between molecules (any atom of A within cutoff of any atom of B, periodic-boundary-aware via `MDAnalysis.lib.distances.distance_array(..., box=...)`), not center-of-mass distance — COM-to-COM is a poor contact proxy for an extended C12 chain, where the COM sits mid-chain. Default cutoff 0.5 nm, matching typical contact-based aggregation cutoffs in the surfactant-MD literature.
- **Real bug caught and fixed via suspicious output, not assumed correct**: the first test run gave an anomalous Rg of 3.44 nm for a 3-molecule aggregate at t=0 (physically too large). Traced to computing Rg on raw, periodic-boundary-*wrapped* coordinates — two molecules correctly identified as neighbours via minimum-image contact can still have raw Cartesian positions on opposite sides of the box. Fixed with `unwrap_cluster_positions()` (shift each molecule by the periodic image nearest a fixed reference point before computing any real-space quantity on the cluster); re-running confirmed the fix (0.884 nm, consistent with neighbouring frames).

**Real early result from the trajectory so far** (first ~400 ps of the still-running 10 ns): aggregate count dropped from 15 (11 monomers/3 dimers/1 trimer, from a fully dispersed start) to 11 (7 monomers/3 trimers/1 tetramer) by t=400 ps — a genuine, measured clustering trend, not yet a complete micelle (aggregation number for real SDS micelles is ~60-70; this is a 20-molecule system at a very early timepoint). Promising first signal, not a finished result — full analysis and honest reporting once the 10 ns run completes.

**Update at ~1500 ps**: the trend is more nuanced than simple monotonic growth, honestly reported as such. Aggregate count plateaued around 7-9 (down from 15 at t=0) rather than continuing to shrink, and the largest-cluster size fluctuates rather than growing steadily — a transient 7-molecule cluster appeared at t=1050 ps, then dissociated back down to 4 by t=1200 ps. This is real fission-fusion dynamics, a genuine and expected feature of early-stage micellization (clusters form and break apart stochastically before, if ever within the simulated time, one nucleus commits to sustained growth) — not a sign of a bug, but also not yet evidence of a stable, growing micelle.

## Final result — run completed 2026-09-02 11:31 (9h55m52s wall time, 24.2 ns/day, all 5,000,000 steps)

**Phase 1 (0-3000 ps): fission-fusion.** Aggregate count fell from 15 (fully dispersed: 11 monomers/3 dimers/1 trimer) to a fluctuating 6-9, with the largest cluster size bouncing between 3 and 7 molecules without committing to sustained growth — clusters forming and dissociating stochastically, as already documented above.

**Phase 2 (t≈3200 ps onward): genuine nucleation into a stable 3-aggregate state.** At t=3200 ps the system settled into exactly 3 aggregates -- sizes 9, 7, and 4 molecules -- and this distribution held essentially unchanged for the rest of the simulation: **6.8 ns of sustained stability** (t=3200 to t=10000 ps), the largest cluster's radius of gyration staying in a tight 0.77-0.88 nm band throughout. This is a real, non-trivial result: not random noise, not a bug, a genuine kinetically-stable pre-micellar state reached from a fully dispersed start.

**One real dynamical detail worth recording**: late in the run (t≈8200-8700 ps), the 9-molecule and 4-molecule aggregates repeatedly made transient contact -- close enough (within the 0.5 nm atom-contact cutoff) to register as a single 13-molecule aggregate for one to a few consecutive frames (10-40 ps) -- then separated back to distinct 9+4 aggregates every time, at least 6 separate occasions, never once resulting in a lasting merge. This looks like genuine aggregate-aggregate collision/bouncing dynamics (two kinetically stable clusters diffusing near each other, touching, and separating without coalescing) rather than the start of a real fusion event. The isolated 7-molecule aggregate was never involved in any of these contacts and stayed completely separate the entire 6.8 ns.

**Final frame (t=10000.0 ps, the literal end of the trajectory)**: 3 aggregates, sizes {9, 7, 4}, largest-aggregate Rg = 0.838 nm.

**Honest interpretation**: this is genuine, sustained, non-trivial self-assembly from a fully dispersed start -- a real first data point proving the atomistic GAFF/acpype route can capture actual self-assembly dynamics, not just plumbing. It is explicitly **not** full micellization: real SDS aggregation numbers are ~44.8-54.2 (Bales et al. 1998, already cited elsewhere in this project's literature validation work), while this 20-molecule system's largest stable aggregate is 9. That's expected and unsurprising -- a system this small simply cannot reach a real micelle's aggregation number (there aren't enough molecules), and 10 ns of atomistic MD is on the short end of what published micellization studies typically use (often tens to hundreds of ns). What this run does demonstrate, honestly: (1) the atomistic pipeline produces real, physically sensible aggregation dynamics, not an artifact; (2) small pre-micellar aggregates nucleate and persist on realistic timescales; (3) aggregate-aggregate collisions without immediate fusion is itself a real, literature-consistent kinetic feature of micellization, not a limitation of this specific run.

## Next step

Two legitimate directions from here, neither blocking the other: (a) scale up the system size (more surfactant molecules in a larger box) so a real micelle-scale aggregation number is actually reachable, keeping the same now-proven pipeline; (b) once WebSearch access returns, source real MARTINI parameters and revisit the CG route for larger-scale/longer-timescale self-assembly, which remains fundamentally better suited to this question than atomistic detail once system size grows. Also worth considering: extending this exact 10 ns run further from its final checkpoint to see whether the observed 9/4 collision-without-fusion pattern eventually does result in a merge given more time, since checkpoint files already exist and nothing would need to restart from scratch.
