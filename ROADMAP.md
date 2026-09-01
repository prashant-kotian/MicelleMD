# MicelleMD Roadmap

Working plan, consolidated from PhD project discussion (2026-09-01/02). Not started yet — this captures the scope and design decisions already made so nothing gets lost before work actually begins, following completion of Paper 3.

## Objective

SurfactantKit computes CPP and aggregation number as **closed-form geometric estimates** from Tanford's tail-volume/length formulas — a fast, analytical approximation, explicitly documented in SurfactantKit itself as "a geometric estimate, not a substitute for aggregation numbers measured directly." MicelleMD is the real-computation complement: *simulate* the same surfactant self-assembling and measure aggregation number, shape, and packing directly from the simulation, rather than estimating it from a single-molecule geometric argument.

This directly strengthens the Paper 3 thesis too: it demonstrates the distinction between "AI recalling/estimating a number" and "actually running the physics" using the project's own tooling as the worked example — SurfactantKit's *estimate* vs. MicelleMD's *simulation* of the identical physical quantity (aggregation number) is a clean, citable illustration of exactly the gap Paper 3's benchmark is built to measure.

## Planned technical approach

- **Method**: coarse-grained MD, MARTINI-style force field (not atomistic — computationally tractable for self-assembly timescales, standard choice in the surfactant-CG-MD literature).
- **Engine**: thin wrapper over an established MD package (GROMACS or OpenMM) — deliberately not a from-scratch MD implementation. The scientific/engineering contribution is the surfactant-specific topology generation and analysis layer, not the integrator.
- **Topology generation**: automated construction of CG surfactant topologies for the surfactant classes already central to this PhD's work — single-chain ionic/nonionic, gemini (two-tail, spacer-linked), and amidoamine-derived architectures specifically (the amide linkage and its hydrogen-bonding behavior is a nontrivial CG-mapping decision worth getting right, not a standard case the usual MARTINI surfactant library covers off the shelf).
- **Simulation**: self-assembly runs starting from random/dispersed initial configuration, standard approach for observing spontaneous micellization in CG-MD.
- **Analysis outputs**:
  - Aggregation number distribution (not just a single mean — real micelles are polydisperse, and reporting the distribution rather than a point estimate is itself a more honest comparison against SurfactantKit's single-number geometric estimate)
  - Micelle shape descriptors: radius of gyration, asphericity/eccentricity
  - Solvent-accessible surface area (SASA)
  - Counterion association/binding degree (a real-computation cross-check against SurfactantKit's `counterion_binding_degree`, which is itself a conductometric-slope-method *estimate*, not a first-principles calculation — same "estimate vs. simulate" comparison pattern)
- **Comparison hooks**: explicit, built-in comparison utility that runs the *same* surfactant through both SurfactantKit's geometric aggregation-number estimate and MicelleMD's simulated result, reporting agreement/disagreement — this comparison IS a core planned deliverable, not an afterthought.

## Open design questions (to resolve when work starts)

- Exact CG mapping scheme for the amidoamine amide linkage (standard MARTINI bead types may not cleanly represent amide hydrogen-bonding behavior — may need a validated custom bead assignment, itself worth literature-checking before hardcoding, per this project's established "verify before hardcode" discipline).
- Which specific surfactants to validate against first — likely reuse the same literature-validated systems already in SurfactantKit's `literature_validation_notes.md` (e.g. SDS, DTAB, CTAB) where independent real aggregation-number measurements already exist (Bales et al. 1998's ~44.8-54.2 range for SDS is a ready-made validation target).
- Compute resource planning — CG-MD self-assembly runs are far more compute-intensive than anything in SurfactantKit; needs its own resource/timeline plan before committing to a run schedule.

## Status

Repo created 2026-09-01 (private, MIT-licensed, `prashant-kotian` account).

**2026-09-02 update — engine plumbing proven working, real physics parameters still pending:**
- OpenMM (the chosen MD engine) installs and runs correctly in this environment.
- `vermouth` (the real MARTINI topology-generation tool, maintained by the actual MARTINI developers) also installs and imports correctly — after fixing a real Windows-specific encoding bug in the package itself (run with `PYTHONUTF8=1`). It bundles genuine MARTINI 3.001 force-field infrastructure, but that bundle is protein/nucleotide-focused out of the box, not surfactant-specific.
- `src/micellemd/cg_model.py`: a CG bead/molecule data model and OpenMM system builder, run and verified end to end — built an 80-bead toy system (20 simple 4-bead linear surfactants), ran 1000 steps of real Langevin dynamics, energy relaxed from -10.5 to -253.6 kJ/mol and stayed numerically stable throughout. **This proves the software plumbing works — it does NOT prove anything physically about surfactant self-assembly**, because the bead-type Lennard-Jones/mass/charge parameters used are explicitly labeled placeholders, not the real MARTINI 3 surfactant parameter set (verifying those needs WebSearch access to the MARTINI publication/parameter files, unavailable this session).
- **Real GROMACS now working too, on a second (Ubuntu) machine**: connected a second laptop over SSH on the same LAN (existing key-based auth already set up), installed GROMACS via conda (no native Windows path exists — checked directly, no binary, no Chocolatey package). Ran a complete, real pipeline — box creation, solvation (884 SPC waters), topology, preprocessing, and an actual steepest-descent energy minimization that **converged in 8 steps** (potential energy -35202.24 kJ/mol) — using only GROMACS's own bundled data. Full setup, and two real bugs found and fixed along the way (a hard segfault specific to one force-field/water-model pairing, and a cutoff-vs-box-size sizing issue), documented in `GROMACS_SETUP.md`. This means MicelleMD now has genuine access to its originally-named primary engine, not just the OpenMM fallback.
- **Next concrete step**: two real paths now, either is legitimate: (a) once WebSearch access is available, source the real MARTINI 3 lipid/surfactant `.itp` parameters (from the MARTINI GitHub/website, Souza et al. 2021 Nat. Methods) and replace `PLACEHOLDER_BEAD_TYPES` in `cg_model.py` with verified values, or (b) build an actual surfactant topology for GROMACS (via `vermouth`'s mapping machinery or hand-built) and run it through the now-proven GROMACS pipeline in `GROMACS_SETUP.md`, cross-checked against the OpenMM placeholder-parameter results already obtained. The CG mapping question for the amidoamine amide linkage still needs resolving either way.

**2026-09-02 update — real solvated, neutralized, ionic atomistic system, via the GAFF/acpype route (path fully independent of the MARTINI-parameter blocker above):**
- Extended the single-ion-in-vacuum GAFF result to a genuine SDS anion + Na+ counterion + explicit TIP3P water system. Found and fixed a real GROMACS bug along the way: combining acpype's own auto-generated topology with a full AMBER-family force field's water/ion definitions causes an "Invalid order for directive defaults" error, because both sources define their own `[ defaults ]` and GROMACS only allows one in the assembled topology. Fixed by restructuring the include order (force field's `forcefield.itp` first, then the ligand's bare `.itp`, not acpype's full `.top`) and encoded as a reusable, tested function: `atomistic_topology.write_combined_topology()`.
- Full pipeline verified twice end to end (once manually, once purely through the module function on a fresh RDKit embedding): box → solvate (866 waters) → neutralize with 1 Na+ via `genion` → real PME energy minimization, **converged in 149 steps**, potential energy ≈ -37,700 kJ/mol. This is a genuinely electrostatically-complete ionic surfactant system in explicit water, not a placeholder or a vacuum toy.
- This GAFF/atomistic route and the MARTINI/CG route remain complementary, not competing (see the trade-off note in `GROMACS_SETUP.md`): atomistic is now the proven path for real, verifiable single-molecule and small-system physics; CG remains the right choice for large-scale self-assembly once real MARTINI parameters are sourced.
