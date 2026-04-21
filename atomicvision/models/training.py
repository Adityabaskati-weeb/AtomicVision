"""Training utilities for DefectNet-lite."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from atomicvision.models.defectnet_lite import DefectNetLite, build_targets, case_to_tensor
from atomicvision.synthetic import CANDIDATE_DEFECTS, generate_case


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a reproducible DefectNet-lite training run."""

    train_samples: int = 64
    val_samples: int = 16
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-3
    difficulty: str = "medium"
    seed: int = 0
    threshold: float = 0.5


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics captured for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    val_concentration_mae: float


@dataclass(frozen=True)
class TrainingResult:
    """Result of a DefectNet-lite training run."""

    config: TrainingConfig
    best_epoch: int
    best_val_loss: float
    history: list[EpochMetrics]
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": [asdict(item) for item in self.history],
            "checkpoint_path": self.checkpoint_path,
        }


class SyntheticDefectDataset(Dataset):
    """Deterministic synthetic spectra dataset."""

    def __init__(
        self,
        seeds: list[int] | tuple[int, ...],
        difficulty: str = "medium",
    ) -> None:
        self.seeds = list(seeds)
        self.difficulty = difficulty

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case = generate_case(seed=self.seeds[index], difficulty=self.difficulty)
        labels, concentrations = build_targets(case)
        return case_to_tensor(case), labels, concentrations


def set_reproducible_seed(seed: int) -> None:
    """Seed Python and PyTorch for repeatable smoke runs."""

    random.seed(seed)
    torch.manual_seed(seed)


def train_defectnet_lite(
    config: TrainingConfig,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> TrainingResult:
    """Train DefectNet-lite and optionally save the best validation checkpoint."""

    if config.train_samples <= 0 or config.val_samples <= 0:
        raise ValueError("train_samples and val_samples must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_reproducible_seed(config.seed)
    model = DefectNetLite()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    label_loss = nn.BCEWithLogitsLoss()
    concentration_loss = nn.L1Loss()

    train_seeds = list(range(config.seed, config.seed + config.train_samples))
    val_start = config.seed + 100_000
    val_seeds = list(range(val_start, val_start + config.val_samples))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        SyntheticDefectDataset(train_seeds, difficulty=config.difficulty),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        SyntheticDefectDataset(val_seeds, difficulty=config.difficulty),
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    history: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            label_loss,
            concentration_loss,
        )
        val_metrics = evaluate_defectnet_lite(
            model,
            val_loader,
            label_loss,
            concentration_loss,
            threshold=config.threshold,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            val_loss=round(val_metrics["loss"], 6),
            val_f1=round(val_metrics["f1"], 6),
            val_concentration_mae=round(val_metrics["concentration_mae"], 6),
        )
        history.append(metrics)
        if metrics.val_loss < best_val_loss:
            best_val_loss = metrics.val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "candidate_defects": list(CANDIDATE_DEFECTS),
                "config": asdict(config),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )

    result = TrainingResult(
        config=config,
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 6),
        history=history,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result


def evaluate_defectnet_lite(
    model: DefectNetLite,
    loader: DataLoader,
    label_loss: nn.Module | None = None,
    concentration_loss: nn.Module | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate DefectNet-lite over a DataLoader."""

    model.eval()
    label_loss = label_loss or nn.BCEWithLogitsLoss()
    concentration_loss = concentration_loss or nn.L1Loss()
    total_loss = 0.0
    batches = 0
    true_positives = 0.0
    predicted_positives = 0.0
    actual_positives = 0.0
    concentration_error = 0.0
    concentration_items = 0

    with torch.no_grad():
        for spectra, labels, concentrations in loader:
            logits, predicted_concentrations = model(spectra)
            loss = label_loss(logits, labels) + concentration_loss(
                predicted_concentrations,
                concentrations,
            )
            probabilities = torch.sigmoid(logits)
            predicted_labels = probabilities >= threshold
            actual_labels = labels >= 0.5
            true_positives += torch.logical_and(predicted_labels, actual_labels).sum().item()
            predicted_positives += predicted_labels.sum().item()
            actual_positives += actual_labels.sum().item()
            concentration_error += torch.abs(
                predicted_concentrations - concentrations
            ).sum().item()
            concentration_items += concentrations.numel()
            total_loss += loss.item()
            batches += 1

    precision = _safe_divide(true_positives, predicted_positives)
    recall = _safe_divide(true_positives, actual_positives)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "loss": _safe_divide(total_loss, batches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "concentration_mae": _safe_divide(concentration_error, concentration_items),
    }


def load_defectnet_lite_checkpoint(
    checkpoint_path: str | Path,
    map_location: str = "cpu",
) -> DefectNetLite:
    """Load a DefectNet-lite inference model from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model = DefectNetLite()
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _train_one_epoch(
    model: DefectNetLite,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    label_loss: nn.Module,
    concentration_loss: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for spectra, labels, concentrations in loader:
        optimizer.zero_grad()
        logits, predicted_concentrations = model(spectra)
        loss = label_loss(logits, labels) + concentration_loss(
            predicted_concentrations,
            concentrations,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return _safe_divide(total_loss, batches)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

