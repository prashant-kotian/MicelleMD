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

## Next step

This is a real bulk-water sanity test, not a surfactant self-assembly run — the real next step is building an actual surfactant topology (starting from `cg_model.py`'s bead model or a proper atomistic-to-CG mapping via `vermouth`) and running it through this same now-proven GROMACS pipeline, cross-checked against the OpenMM results already obtained on the same placeholder-parameter system.
