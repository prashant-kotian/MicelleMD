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

## Next step

Now have two real, verified options: (a) scale this up to multiple SDS ions + Na+ counterions in one box (testing genuine self-assembly at small scale) via the now-proven acpype/GAFF route, or (b) once WebSearch access returns, source real MARTINI parameters and revisit the CG route for larger-scale self-assembly. Both are legitimate; (a) is available right now with zero further blockers.
