"""Evaluation utilities for AtomicVision."""

from atomicvision.evaluation.metrics import AggregateMetrics, aggregate_rewards
from atomicvision.evaluation.comparison import (
    DEFAULT_DIFFICULTIES,
    DEFAULT_POLICIES,
    RewardComparison,
    run_reward_comparison,
    write_comparison_artifacts,
)
from atomicvision.evaluation.policies import (
    POLICY_NAMES,
    PolicyEpisodeResult,
    PolicyEvaluationSummary,
    evaluate_policy,
    run_policy_episode,
)

__all__ = [
    "POLICY_NAMES",
    "DEFAULT_DIFFICULTIES",
    "DEFAULT_POLICIES",
    "AggregateMetrics",
    "PolicyEpisodeResult",
    "PolicyEvaluationSummary",
    "RewardComparison",
    "aggregate_rewards",
    "evaluate_policy",
    "run_policy_episode",
    "run_reward_comparison",
    "write_comparison_artifacts",
]
