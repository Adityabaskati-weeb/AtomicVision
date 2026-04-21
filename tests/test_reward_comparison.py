from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomicvision.evaluation import run_reward_comparison, write_comparison_artifacts


def test_reward_comparison_contains_all_requested_rows() -> None:
    comparison = run_reward_comparison(
        difficulties=("easy", "medium"),
        policies=("cheap_submit", "oracle"),
        episodes=2,
    )

    assert len(comparison.rows) == 4
    assert comparison.episodes_per_policy == 2


def test_reward_comparison_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unknown policies"):
        run_reward_comparison(policies=("not_real",), episodes=1)


def test_write_comparison_artifacts() -> None:
    output_dir = Path("outputs/test-artifacts/reward-comparison")
    comparison = run_reward_comparison(
        difficulties=("easy",),
        policies=("cheap_submit", "oracle"),
        episodes=2,
    )

    artifacts = write_comparison_artifacts(comparison, output_dir)

    assert set(artifacts) == {"json", "csv", "markdown", "svg"}
    assert json.loads(artifacts["json"].read_text(encoding="utf-8"))["episodes_per_policy"] == 2
    assert "AtomicVision Reward Comparison" in artifacts["markdown"].read_text(encoding="utf-8")
    assert artifacts["svg"].read_text(encoding="utf-8").startswith("<svg")
