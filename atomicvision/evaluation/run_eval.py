"""Command line baseline evaluation for AtomicVision."""

from __future__ import annotations

import argparse
import json

from atomicvision.evaluation.policies import POLICY_NAMES, evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AtomicVision baseline evaluation.")
    parser.add_argument("--policy", choices=POLICY_NAMES, default="prior_submit")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    summary = evaluate_policy(args.policy, seeds=seeds, difficulty=args.difficulty)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

