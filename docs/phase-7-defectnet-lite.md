# Phase 7 DefectNet-Lite

## Purpose

Phase 7 adds a compact PyTorch spectral-attention model inspired by DefectNet. This phase creates a trainable model component but does not yet replace the OpenEnv heuristic prior or implement GRPO training.

## Implemented Scope

- `DefectNetLite` PyTorch module.
- Three-channel input representation: pristine spectrum, defective spectrum, and delta spectrum.
- Conv1D spectral encoder.
- Multihead attention over spectral tokens.
- Multi-label defect classifier.
- Concentration regressor.
- Case-to-tensor conversion.
- Target construction from synthetic material cases.
- Thresholded prediction helper.
- Minimal training smoke script.

## Validation Gate

Phase 7 is complete only when:

- PyTorch imports locally.
- Case tensors have expected shape.
- Targets align with the candidate defect set.
- Model forward pass returns correct logits and concentration shapes.
- Invalid tensor shapes raise a clear error.
- One optimizer step runs.
- Prediction helper returns matched defect/concentration lists.
- Full test suite passes locally.

## Phase 8 Entry Criteria

Start Phase 8 only after DefectNet-lite tests pass. Phase 8 will integrate model-backed prior inference into the OpenEnv environment or add a reproducible training/evaluation path for the model.
