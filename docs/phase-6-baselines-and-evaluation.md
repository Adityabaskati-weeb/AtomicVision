# Phase 6 Baselines And Evaluation

## Purpose

Phase 6 adds deterministic baseline policies and evaluation scripts. This phase proves the environment can produce measurable reward comparisons before LLM/GRPO training begins.

## Implemented Scope

- `cheap_submit` baseline.
- `random` baseline.
- `scan_heavy` baseline.
- `prior_submit` baseline.
- `oracle` upper-bound policy for sanity checking.
- Per-episode policy result records.
- Aggregate policy evaluation summaries.
- JSON CLI evaluation entrypoint.

## Why This Matters For Judging

Reward improvement is worth 20% of first-round judging. Before training a model, AtomicVision needs stable baseline numbers so later reward curves have a clear reference point.

The oracle policy is not a real agent baseline. It exists only as an upper-bound sanity check to prove the reward function gives high scores to correct behavior.

## Validation Gate

Phase 6 is complete only when:

- Every baseline policy completes an episode.
- Oracle scores above cheap submit.
- Policy evaluation is deterministic for fixed seeds.
- Unknown policies raise clear errors.
- CLI evaluation emits valid JSON.
- Full test suite passes locally.

## Phase 7 Entry Criteria

Start Phase 7 only after baseline evaluation tests pass. Phase 7 will add a minimal PyTorch DefectNet-lite prior or improve the current heuristic prior into a trainable model component.
