"""Generate judge-facing reward comparison artifacts."""

from __future__ import annotations

import argparse

from atomicvision.evaluation.comparison import (
    DEFAULT_DIFFICULTIES,
    DEFAULT_POLICIES,
    run_reward_comparison,
    write_comparison_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AtomicVision reward comparison assets.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/reward-comparison")
    parser.add_argument("--difficulties", nargs="+", default=list(DEFAULT_DIFFICULTIES))
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    args = parser.parse_args()

    comparison = run_reward_comparison(
        difficulties=tuple(args.difficulties),
        policies=tuple(args.policies),
        episodes=args.episodes,
        seed_start=args.seed_start,
    )
    artifacts = write_comparison_artifacts(comparison, args.output_dir)
    for kind, path in artifacts.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()

