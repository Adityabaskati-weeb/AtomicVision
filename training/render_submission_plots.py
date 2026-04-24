"""Render validator-facing training plots for the repository README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
LOSS_HISTORY = [
    (1, 0.185758),
    (2, 0.490042),
    (3, 0.468173),
    (4, 0.226877),
    (5, 0.108195),
    (6, 0.626313),
    (7, 0.781352),
    (8, 0.333146),
    (9, 0.057802),
    (10, 0.057916),
    (11, 0.883443),
    (12, 0.557062),
    (13, 0.466861),
    (14, 0.015496),
    (15, 0.247391),
    (16, 0.287989),
    (17, 0.402126),
    (18, 0.259643),
    (19, 0.264154),
    (20, 0.009972),
    (21, 0.007194),
    (22, 0.337385),
    (23, 0.300913),
    (24, 0.129860),
    (25, 0.163828),
    (26, 0.128786),
    (27, 0.388579),
    (28, 0.003683),
    (29, 0.311117),
    (30, 0.002321),
    (31, 0.222219),
    (32, 0.004358),
    (33, 0.010711),
    (34, 0.002939),
    (35, 0.280600),
    (36, 0.605243),
    (37, 0.001922),
    (38, 0.140333),
    (39, 0.413483),
    (40, 0.134504),
]
REWARD_POINTS = [
    ("Prior-submit baseline", 4.366),
    ("SFT-copy direct rollout", 4.458),
    ("Cost-aware masked SFT checkpoint-40", 4.475),
]


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    render_loss_curve(DOCS_DIR / "training-loss-curve.png")
    render_reward_curve(DOCS_DIR / "training-reward-curve.png")
    print("Wrote:", DOCS_DIR / "training-loss-curve.png")
    print("Wrote:", DOCS_DIR / "training-reward-curve.png")


def render_loss_curve(output_path: Path) -> None:
    updates = [update for update, _loss in LOSS_HISTORY]
    losses = [loss for _update, loss in LOSS_HISTORY]

    plt.figure(figsize=(9, 5))
    plt.plot(updates, losses, color="#2563eb", linewidth=2.2, marker="o", markersize=3)
    plt.title("AtomicVision Merged SFT Loss Curve")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def render_reward_curve(output_path: Path) -> None:
    stages = [stage for stage, _reward in REWARD_POINTS]
    rewards = [reward for _stage, reward in REWARD_POINTS]

    plt.figure(figsize=(9, 5))
    plt.plot(range(len(stages)), rewards, color="#16a34a", linewidth=2.4, marker="o", markersize=7)
    plt.xticks(range(len(stages)), stages, rotation=10, ha="right")
    plt.title("AtomicVision Reward Progression")
    plt.xlabel("Training stage")
    plt.ylabel("Mean reward")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
