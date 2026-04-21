from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from atomicvision.models import (
    DefectNetLite,
    SyntheticDefectDataset,
    TrainingConfig,
    build_targets,
    case_to_tensor,
    load_defectnet_lite_checkpoint,
    predict_case,
    train_defectnet_lite,
)
from atomicvision.synthetic import CANDIDATE_DEFECTS, generate_case


def test_case_to_tensor_shape() -> None:
    case = generate_case(seed=1, difficulty="easy")
    tensor = case_to_tensor(case)

    assert tensor.shape == (3, len(case.frequency_axis))


def test_build_targets_shape_and_labels() -> None:
    case = generate_case(seed=2, difficulty="medium")
    labels, concentrations = build_targets(case)

    assert labels.shape == (len(CANDIDATE_DEFECTS),)
    assert concentrations.shape == (len(CANDIDATE_DEFECTS),)
    assert labels.sum().item() == len(case.defects)
    assert concentrations.max().item() > 0.0


def test_defectnet_lite_forward_shapes() -> None:
    case = generate_case(seed=3, difficulty="medium")
    model = DefectNetLite()

    logits, concentrations = model(case_to_tensor(case).unsqueeze(0))

    assert logits.shape == (1, len(CANDIDATE_DEFECTS))
    assert concentrations.shape == (1, len(CANDIDATE_DEFECTS))
    assert torch.all(concentrations >= 0.0)


def test_defectnet_lite_rejects_wrong_shape() -> None:
    model = DefectNetLite()

    try:
        model(torch.zeros(1, 2, 128))
    except ValueError as exc:
        assert "spectra must have shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong input shape")


def test_single_training_step_runs() -> None:
    cases = [generate_case(seed=i, difficulty="easy") for i in range(4)]
    x = torch.stack([case_to_tensor(case) for case in cases])
    labels, concentrations = zip(*(build_targets(case) for case in cases), strict=True)
    y_labels = torch.stack(labels)
    y_concentrations = torch.stack(concentrations)

    model = DefectNetLite()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    logits, predicted_concentrations = model(x)
    loss = loss_fn(logits, y_labels) + torch.nn.functional.l1_loss(
        predicted_concentrations,
        y_concentrations,
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_predict_case_returns_valid_lengths() -> None:
    case = generate_case(seed=8, difficulty="easy")
    model = DefectNetLite()

    prediction = predict_case(model, case, threshold=0.0)

    assert prediction.predicted_defects
    assert len(prediction.predicted_defects) == len(prediction.predicted_concentrations)
    assert 0.0 <= prediction.confidence <= 1.0


def test_synthetic_defect_dataset_returns_training_tensors() -> None:
    dataset = SyntheticDefectDataset(seeds=[1, 2], difficulty="easy")

    spectra, labels, concentrations = dataset[0]

    assert len(dataset) == 2
    assert spectra.shape[0] == 3
    assert labels.shape == (len(CANDIDATE_DEFECTS),)
    assert concentrations.shape == (len(CANDIDATE_DEFECTS),)


def test_training_utility_saves_checkpoint_and_metrics() -> None:
    artifact_dir = Path("outputs/test-artifacts/test_training_utility")
    checkpoint = artifact_dir / "defectnet_lite.pt"
    metrics_path = artifact_dir / "metrics.json"
    config = TrainingConfig(
        train_samples=6,
        val_samples=2,
        epochs=1,
        batch_size=3,
        difficulty="easy",
        seed=123,
    )

    result = train_defectnet_lite(
        config,
        checkpoint_path=checkpoint,
        metrics_path=metrics_path,
    )

    assert checkpoint.exists()
    assert metrics_path.exists()
    assert result.best_epoch == 1
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["best_epoch"] == 1


def test_checkpoint_loads_for_inference() -> None:
    checkpoint = Path("outputs/test-artifacts/test_checkpoint_loads/defectnet_lite.pt")
    config = TrainingConfig(
        train_samples=4,
        val_samples=2,
        epochs=1,
        batch_size=2,
        difficulty="easy",
        seed=4,
    )
    train_defectnet_lite(config, checkpoint_path=checkpoint)

    model = load_defectnet_lite_checkpoint(checkpoint)
    case = generate_case(seed=13, difficulty="easy")
    prediction = predict_case(model, case, threshold=0.0)

    assert prediction.predicted_defects
    assert len(prediction.predicted_defects) == len(prediction.predicted_concentrations)
