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

Early scaffolding. This repository is being established now so its development history begins with the project; the analysis modules are built out following completion of the SurfactantKit/SurfBench work that motivates it.

## Relationship to the wider PhD work

Part of a series of open surfactant/interfacial-science tools developed alongside a PhD on amidoamine-derived gemini cationic surfactants and their mixed systems. Sibling projects: SurfactantKit (analytical theory + MCP tools), and SurfQSPR (structure-to-property machine learning).

## License

MIT — see `LICENSE`.
