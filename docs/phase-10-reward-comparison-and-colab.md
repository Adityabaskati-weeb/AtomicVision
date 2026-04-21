# Phase 10 Reward Comparison And Colab Bridge

## Purpose

Phase 10 creates judge-facing reward comparison assets and a notebook bridge for the upcoming fine-tuning phase. This phase does not run TRL/Unsloth training yet.

## Implemented Scope

- Deterministic reward comparison runner across policies and difficulties.
- JSON, CSV, Markdown, and SVG artifact generation.
- Judge-facing reward comparison report.
- Colab bridge notebook scaffold.
- Tests for comparison generation and artifact writing.

## Why Fine-Tuning Starts After This Phase

Fine-tuning begins in Phase 11 because the trained agent needs a stable baseline to beat. Phase 10 creates those baselines and the notebook path that Phase 11 will extend into TRL/Unsloth GRPO training.

## Phase 10 Validation Gate

Phase 10 is complete. Completed validation:

- Reward comparison tests pass.
- Artifact generator writes JSON, CSV, Markdown, and SVG files.
- Judge-facing report exists under `docs/`.
- Colab bridge notebook exists under `notebooks/`.
- Full local test suite passes.

## Produced Artifacts

- Reward report: `docs/reward-comparison-report.md`
- Reward chart: `docs/reward-comparison-chart.svg`
- Colab bridge: `notebooks/AtomicVision_GRPO_Colab.ipynb`

## Fine-Tuning Start Point

Actual fine-tuning starts in Phase 11. The first Phase 11 target is to train an agent that beats the `prior_submit` baseline from the reward report while keeping scan cost low.

## Phase 11 Entry Criteria

Start Phase 11 after Phase 10 passes. Phase 11 is where actual fine-tuning starts, using either Colab, Kaggle, or another GPU runtime.
