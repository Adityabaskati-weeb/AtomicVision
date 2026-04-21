# Phase 5 OpenEnv Wrapper

## Purpose

Phase 5 wraps the synthetic materials world and reward scorer in a real OpenEnv-compatible environment. This phase does not implement model training, the dashboard, or Hugging Face deployment.

## Implemented Scope

- `AtomicVisionAction` typed OpenEnv action model.
- `AtomicVisionObservation` typed OpenEnv observation model.
- `AtomicVisionState` typed OpenEnv state model.
- `AtomicVisionEnvironment` with `reset()`, `step()`, and `state`.
- Action handling for scans, zoom scans, reference comparison, prior requests, and final submissions.
- Budget tracking, scan history, step count, and timeout behavior.
- FastAPI OpenEnv app entrypoint.
- Minimal OpenEnv manifest.
- Environment contract tests.

## Validation Gate

Phase 5 is complete only when:

- Reset returns a valid initial observation.
- Scan actions update budget and history.
- Reference comparison reveals the pristine spectrum.
- Prior requests return a candidate prediction.
- Truth submissions terminate with positive reward.
- Invalid zoom actions return a penalty without crashing.
- Step limit triggers timeout scoring.
- OpenEnv app imports successfully.
- Full test suite passes locally.

## Phase 6 Entry Criteria

Start Phase 6 only after OpenEnv wrapper tests pass. Phase 6 will add baseline policies and evaluation scripts so reward improvement can be shown before LLM/GRPO training.
