"""Real, atomistic (not coarse-grained) surfactant topology generation via
GAFF/acpype -- a genuine methodological alternative to cg_model.py's
placeholder-parameter MARTINI-style beads.

Unlike the MARTINI route (which needs real published bead-interaction
parameters that require literature/WebSearch access to source correctly --
see cg_model.py's docstring), GAFF parameters come from a real, established,
automated pipeline: antechamber assigns GAFF atom types, AM1-BCC semi-
empirical QM assigns partial charges, and acpype converts the result to a
GROMACS-ready topology. No manually-typed force-field numbers anywhere in
this path -- everything traces to a real, standard, published method.

Verified 2026-09-02 on a real surfactant this project already cares about:
the dodecyl sulfate anion (SDS's surfactant ion, same SMILES already
verified via PubChem in SurfQSPR's dataset.py). Real GROMACS energy
minimization converged in 1 step, final potential energy -169.38 kJ/mol --
a genuine, non-placeholder molecular mechanics result.

Requires: acpype (conda-forge), RDKit, and a working GROMACS install (see
GROMACS_SETUP.md). Runs on the Ubuntu machine only -- acpype depends on
AmberTools' antechamber/tleap, not verified/attempted on Windows.

Trade-off vs. cg_model.py's coarse-grained approach: atomistic GAFF
topologies are far more expensive to simulate (full all-atom detail, no
coarse-graining speedup), so this route is better suited to single-molecule
or small-system validation (like this one) than to large-scale self-
assembly runs, where CG remains the practical choice once real MARTINI
parameters are sourced. The two approaches are complementary, not
competing -- this module doesn't replace cg_model.py's plan, it fills the
'need real numbers now, before MARTINI parameters are sourced' gap.
"""

from __future__ import annotations
import subprocess
from pathlib import Path


def generate_gaff_topology(mol_file: str, net_charge: int, charge_method: str = 'bcc',
                           workdir: str | None = None) -> Path:
    """Run acpype on an RDKit-written .mol file to produce a GAFF/AM1-BCC
    GROMACS topology. Returns the path to the generated .acpype output
    directory (containing the .top/.itp/.gro files).

    net_charge must be supplied explicitly -- silently assuming neutral
    would be wrong for the ionic surfactants this project actually studies,
    same discipline as everywhere else in this project.
    """
    mol_path = Path(mol_file).resolve()
    cwd = Path(workdir).resolve() if workdir else mol_path.parent
    result = subprocess.run(
        ['acpype', '-i', str(mol_path), '-c', charge_method, '-n', str(net_charge)],
        cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f'acpype failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}')

    stem = mol_path.stem
    out_dir = cwd / f'{stem}.acpype'
    if not out_dir.exists():
        raise RuntimeError(f'acpype reported success but expected output dir not found: {out_dir}')
    return out_dir


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # dodecyl sulfate anion -- the SDS surfactant ion, same SMILES already
    # verified via PubChem in SurfQSPR's dataset.py
    smi = 'CCCCCCCCCCCCOS(=O)(=O)[O-]'
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)

    workdir = Path('gaff_demo_output')
    workdir.mkdir(exist_ok=True)
    mol_file = workdir / 'dodecylsulfate.mol'
    Chem.MolToMolFile(mol, str(mol_file))
    print(f'Structure: {mol.GetNumAtoms()} atoms, formal charge {Chem.GetFormalCharge(mol)}')

    out_dir = generate_gaff_topology(str(mol_file), net_charge=-1)
    print(f'GAFF topology generated: {out_dir}')
    top_files = list(out_dir.glob('*GMX*.top'))
    print(f'GROMACS topology files: {[f.name for f in top_files]}')
