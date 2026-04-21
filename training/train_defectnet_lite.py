"""Train DefectNet-lite on synthetic AtomicVision spectra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atomicvision.models import TrainingConfig, train_defectnet_lite


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DefectNet-lite on synthetic spectra.")
    parser.add_argument("--samples", type=int, default=64, help="Training sample count.")
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/defectnet_lite.pt")
    parser.add_argument("--metrics", default="outputs/metrics/defectnet_lite_training.json")
    args = parser.parse_args()

    config = TrainingConfig(
        train_samples=args.samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    result = train_defectnet_lite(
        config,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
    )
    for metrics in result.history:
        print(
            "epoch={epoch} train_loss={train_loss:.6f} "
            "val_loss={val_loss:.6f} val_f1={val_f1:.6f} "
            "val_concentration_mae={val_concentration_mae:.6f}".format(**metrics.__dict__)
        )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
