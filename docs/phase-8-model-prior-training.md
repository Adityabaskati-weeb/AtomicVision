# Phase 8 Model Prior Training

## Purpose

Phase 8 turns DefectNet-lite from a model definition into a reusable model-backed prior for the OpenEnv lab. It adds reproducible training utilities, validation metrics, checkpoint saving/loading, and an optional `prior_mode="model"` path inside the environment.

## Research And Engineering Notes

Phase 8 follows these implementation choices:

- Use `torch.utils.data.Dataset` and `DataLoader` to keep data generation decoupled from the training loop.
- Seed Python and PyTorch for reproducible smoke runs.
- Use a fixed train/validation seed split so validation cases are not seen during training.
- Save the best validation model as a PyTorch `state_dict` checkpoint.
- Load checkpoints with `model.eval()` before inference.
- Keep the OpenEnv reward logic separate from model training.

## Implemented Scope

- `SyntheticDefectDataset` for deterministic generated spectra.
- `TrainingConfig`, `EpochMetrics`, and `TrainingResult` records.
- Reproducible `train_defectnet_lite()` training utility.
- `evaluate_defectnet_lite()` validation metrics.
- Best-validation checkpoint saving.
- JSON training metrics output.
- Checkpoint loading for inference.
- Updated training CLI with train/validation settings.
- `AtomicVisionEnvironment(prior_mode="model")`.
- Optional `prior_checkpoint_path` for model-backed `ask_prior`.
- Tests for dataset output, checkpoint writing, checkpoint loading, and model prior integration.

## Validation Gate

Phase 8 is complete only when:

- Training utilities save a checkpoint and metrics JSON.
- Checkpoints load into an inference model.
- Model-backed prior returns a valid OpenEnv prior prediction.
- Environment can use an explicit checkpoint path.
- Training CLI runs on a tiny CPU smoke configuration.
- Full test suite passes locally.

## Current Resource Needs

Phase 8 can run locally on CPU for smoke tests. For the winning version, later phases will need:

- A Hugging Face token with Space write access.
- A Colab notebook runtime for TRL/Unsloth GRPO training.
- Optional GPU runtime for faster model and agent training.
- A short demo script or video recording setup for the under-2-minute submission artifact.

## Phase 9 Entry Criteria

Start Phase 9 only after Phase 8 tests and training smoke checks pass. Phase 9 should create judge-facing reward comparison artifacts and prepare the GRPO/TRL Colab bridge.
