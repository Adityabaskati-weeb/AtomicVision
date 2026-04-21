"""Aggregate metrics for AtomicVision evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

from atomicvision.rewards.scoring import RewardBreakdown


@dataclass(frozen=True)
class AggregateMetrics:
    """Summary metrics across multiple evaluated episodes."""

    episodes: int
    mean_reward: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_concentration_mae: float
    mean_scan_cost_penalty: float
    timeout_rate: float


def aggregate_rewards(rewards: list[RewardBreakdown]) -> AggregateMetrics:
    """Aggregate reward breakdowns into judge-facing metrics."""

    if not rewards:
        raise ValueError("at least one reward breakdown is required")

    count = len(rewards)
    timeouts = sum(1 for reward in rewards if reward.timeout_penalty < 0.0)
    return AggregateMetrics(
        episodes=count,
        mean_reward=round(_mean(reward.total_reward for reward in rewards), 6),
        mean_precision=round(_mean(reward.precision for reward in rewards), 6),
        mean_recall=round(_mean(reward.recall for reward in rewards), 6),
        mean_f1=round(_mean(reward.f1 for reward in rewards), 6),
        mean_concentration_mae=round(
            _mean(reward.concentration_mae for reward in rewards),
            6,
        ),
        mean_scan_cost_penalty=round(
            _mean(reward.scan_cost_penalty for reward in rewards),
            6,
        ),
        timeout_rate=round(timeouts / count, 6),
    )


def _mean(values) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)

