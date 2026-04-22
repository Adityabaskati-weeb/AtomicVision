"""Evaluate the AtomicVision agent policy behavior from a notebook or shell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomicvision.evaluation import POLICY_NAMES, evaluate_policy  # noqa: E402


DEFAULT_POLICIES = ("cheap_submit", "prior_submit", "scan_heavy", "oracle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AtomicVision policy behavior on deterministic local episodes.",
    )
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument(
        "--output-json",
        default="outputs/evaluation/atomicvision_agent_eval.json",
        help="Path for the JSON report.",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write a JSON report.")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    unknown = [policy for policy in args.policies if policy not in POLICY_NAMES]
    if unknown:
        raise ValueError(f"Unknown policies: {', '.join(unknown)}")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    rows = [
        evaluate_policy(policy, seeds=seeds, difficulty=args.difficulty).to_dict()
        for policy in args.policies
    ]
    report = {
        "difficulty": args.difficulty,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "note": (
            "GRPO training logs converged to the efficient ask_prior -> "
            "submit_defect_map behavior; compare that against prior_submit."
        ),
        "rows": rows,
    }

    output_json = "" if args.no_write else args.output_json
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not args.json_only:
        print(_table(rows))
        if output_json:
            print(f"\nWrote JSON report to {output_json}")
        print("\nFull JSON:")
    print(json.dumps(report, indent=2, sort_keys=True))


def _table(rows: list[dict[str, float | int | str]]) -> str:
    headers = ("policy", "reward", "f1", "mae", "steps", "cost", "timeouts")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {policy_name} | {mean_reward:.3f} | {mean_f1:.3f} | "
            "{mean_concentration_mae:.4f} | {mean_steps:.2f} | "
            "{mean_scan_cost:.2f} | {timeout_rate:.2f} |".format(**row)
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
