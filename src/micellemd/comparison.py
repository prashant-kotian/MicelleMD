"""Comparison hook: run the same surfactant through both SurfactantKit's
closed-form geometric aggregation-number/CPP estimate and MicelleMD's
simulated result, reporting agreement/disagreement -- the explicit,
built-in comparison utility named as a core deliverable in ROADMAP.md's
objective ("SurfactantKit's *estimate* vs. MicelleMD's *simulation* of the
identical physical quantity... a clean, citable illustration of exactly
the gap Paper 3's benchmark is built to measure").

Requires SurfactantKit importable (a separate, sibling repo/package --
`pip install -e ../SurfactantKit` or equivalent on this machine)."""

from __future__ import annotations

from surfactantkit.cpp import (
    tanford_tail_volume, tanford_critical_length,
    critical_packing_parameter, aggregation_number_spherical,
    classify_aggregate_morphology,
)


def compare_aggregation_number(n_carbons: int, head_area_A2: float, simulated_aggregation_number: int) -> dict:
    """Compare SurfactantKit's geometric aggregation-number estimate
    against a real MicelleMD-simulated aggregate size for the same
    surfactant tail length.

    n_carbons: saturated alkyl tail carbon count (same convention as
    SurfactantKit's tanford_tail_volume/tanford_critical_length).
    head_area_A2: optimal headgroup area (sq Angstrom) -- REQUIRED, no
    default. This is real, condition-specific literature knowledge (see
    SurfactantKit's own critical_packing_parameter docstring and the
    2026-09-04 SurfBench finding that headgroup area is genuinely
    contested across measurement methods, unlike Davies' HLB numbers) --
    do not guess a value here either; pass one you can cite, or measure
    it from the simulated aggregate's own surface area per headgroup.
    simulated_aggregation_number: the real, measured largest-aggregate
    size from a MicelleMD trajectory (e.g. atomistic_analysis.py's
    max_aggregate_size or analysis.py's aggregation_number_distribution).

    Returns the geometric estimate's own intermediate values (so the
    comparison is auditable, not just a final ratio), the simulated
    value as given, and their ratio.
    """
    if n_carbons <= 0:
        raise ValueError("n_carbons must be positive")
    if head_area_A2 <= 0:
        raise ValueError("head_area_A2 must be positive -- and must be a real, cited value, not a guess")
    if simulated_aggregation_number <= 0:
        raise ValueError("simulated_aggregation_number must be a positive real measurement")

    tail_volume_A3 = tanford_tail_volume(n_carbons)
    critical_length_A = tanford_critical_length(n_carbons)
    cpp = critical_packing_parameter(tail_volume_A3, head_area_A2, critical_length_A)
    morphology = classify_aggregate_morphology(cpp)
    geometric_aggregation_number = aggregation_number_spherical(tail_volume_A3, critical_length_A)

    return {
        "geometric_estimate": {
            "tail_volume_A3": tail_volume_A3,
            "critical_length_A": critical_length_A,
            "head_area_A2": head_area_A2,
            "cpp": cpp,
            "predicted_morphology": morphology,
            "aggregation_number": geometric_aggregation_number,
        },
        "simulated_aggregation_number": simulated_aggregation_number,
        "ratio_simulated_to_geometric": simulated_aggregation_number / geometric_aggregation_number,
    }
