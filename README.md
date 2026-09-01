# MicelleMD

Open-source toolkit for **coarse-grained molecular dynamics of surfactant self-assembly** — building surfactant topologies from structure, running self-assembly simulations, and extracting aggregation number, micelle shape, radius of gyration, and solvent-accessible surface area.

## Motivation

Analytical colloid-science theory (as implemented in [SurfactantKit](https://github.com/prashant-kotian/surfactantkit)) gives fast, closed-form *estimates* of micelle geometry — critical packing parameter, geometric aggregation number, and so on. Those are order-of-magnitude sanity checks derived from single-molecule geometry, not simulations of the actual aggregate.

MicelleMD is the complementary, computational-chemistry side of the same question: instead of estimating aggregation number from Tanford's tail-volume formula, *simulate* the surfactants assembling and measure it. This is the kind of quantity a general-purpose reasoning model cannot compute without running real physics.

## Planned scope

- Automated MARTINI-style coarse-grained topology generation for common surfactant classes (single-chain ionic/nonionic, gemini, amidoamine-derived).
- Self-assembly simulation driver (thin wrapper over an established MD engine — GROMACS/OpenMM — not a re-implementation of MD).
- Analysis: aggregation number distribution, micelle radius of gyration and asphericity, solvent-accessible surface area, counterion association.
- Comparison hooks so a simulated aggregation number can be placed side by side with SurfactantKit's geometric estimate for the same surfactant.

## Status

Two working simulation engines (OpenMM and, via a second Linux machine, native GROMACS) and two topology routes: a coarse-grained MARTINI-style bead model (including the actual gemini two-tail/spacer architecture this PhD studies) currently running on explicitly-labeled placeholder parameters pending real MARTINI 3 literature values, and a real, no-placeholder atomistic route (GAFF/AM1-BCC via `acpype`) already producing genuine, non-placeholder molecular mechanics results. Built and minimized a real SDS anion + Na+ + explicit-water system, then scaled to a 20-molecule self-assembly test (real GROMACS bug found and fixed along the way — a duplicate force-field `[ defaults ]` directive) with a PBC-aware aggregation-clustering analysis pipeline (MDAnalysis-based, two more real bugs found and fixed: an unsupported .tpr version, and a periodic-boundary Rg-inflation bug) tested against the live, still-running trajectory. See `ROADMAP.md` and `GROMACS_SETUP.md` for the full, dated history.

## Relationship to the wider PhD work

Part of a series of open surfactant/interfacial-science tools developed alongside a PhD on amidoamine-derived gemini cationic surfactants and their mixed systems. Sibling projects: SurfactantKit (analytical theory + MCP tools), SurfQSPR (structure-to-property machine learning), and SurfQM (quantum-chemistry descriptor extraction).

## License

MIT — see `LICENSE`.
