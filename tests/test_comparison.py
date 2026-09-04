"""Tests for comparison.py -- the SurfactantKit-estimate-vs-MicelleMD-
simulation comparison hook named as a core deliverable in ROADMAP.md."""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from micellemd.comparison import compare_aggregation_number


def test_compare_aggregation_number_returns_auditable_intermediates():
    result = compare_aggregation_number(n_carbons=12, head_area_A2=50.0, simulated_aggregation_number=16)
    geo = result["geometric_estimate"]
    assert geo["tail_volume_A3"] > 0
    assert geo["critical_length_A"] > 0
    assert geo["cpp"] > 0
    assert geo["predicted_morphology"]
    assert geo["aggregation_number"] > 0
    assert result["simulated_aggregation_number"] == 16
    assert result["ratio_simulated_to_geometric"] == pytest.approx(
        16 / geo["aggregation_number"])


def test_compare_aggregation_number_rejects_bad_input():
    with pytest.raises(ValueError):
        compare_aggregation_number(n_carbons=0, head_area_A2=50.0, simulated_aggregation_number=16)
    with pytest.raises(ValueError):
        compare_aggregation_number(n_carbons=12, head_area_A2=0.0, simulated_aggregation_number=16)
    with pytest.raises(ValueError):
        compare_aggregation_number(n_carbons=12, head_area_A2=50.0, simulated_aggregation_number=0)
