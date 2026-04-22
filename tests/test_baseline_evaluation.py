from __future__ import annotations

import json
import subprocess
import sys

import pytest

from atomicvision.evaluation import POLICY_NAMES, evaluate_policy, run_policy_episode


@pytest.mark.parametrize("policy_name", POLICY_NAMES)
def test_policy_episode_completes(policy_name: str) -> None:
    result = run_policy_episode(policy_name, seed=1, difficulty="easy")

    assert result.done is True
    assert result.steps >= 1


def test_oracle_scores_higher_than_cheap_submit() -> None:
    seeds = [0, 1, 2, 3]

    cheap = evaluate_policy("cheap_submit", seeds=seeds, difficulty="medium")
    oracle = evaluate_policy("oracle", seeds=seeds, difficulty="medium")

    assert oracle.mean_reward > cheap.mean_reward
    assert oracle.mean_f1 == 1.0


def test_policy_evaluation_is_deterministic() -> None:
    seeds = [5, 6, 7]

    first = evaluate_policy("prior_submit", seeds=seeds, difficulty="hard")
    second = evaluate_policy("prior_submit", seeds=seeds, difficulty="hard")

    assert first == second


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown policy"):
        run_policy_episode("not_a_policy", seed=1, difficulty="easy")


def test_run_eval_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "atomicvision.evaluation.run_eval",
            "--policy",
            "prior_submit",
            "--difficulty",
            "easy",
            "--episodes",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["policy_name"] == "prior_submit"
    assert payload["episodes"] == 2


def test_training_eval_wrapper_outputs_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "training/evaluate_atomicvision_agent.py",
            "--policies",
            "prior_submit",
            "--difficulty",
            "easy",
            "--episodes",
            "2",
            "--no-write",
            "--json-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["difficulty"] == "easy"
    assert payload["episodes"] == 2
    assert payload["rows"][0]["policy_name"] == "prior_submit"
