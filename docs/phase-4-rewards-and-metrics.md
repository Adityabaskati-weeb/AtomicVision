# Phase 4 Reward Scoring And Metrics

## Purpose

Phase 4 implements transparent reward scoring and aggregate evaluation metrics. This phase does not implement the OpenEnv server, agent training, dashboard, or PyTorch model.

## Implemented Scope

- Final submission reward scoring.
- Identity precision, recall, and F1.
- Concentration mean absolute error, gated by identity F1 so blind submissions cannot earn concentration credit.
- Confidence calibration reward, gated by identity F1 so low-confidence blind submissions do not beat real investigation.
- Confidence calibration reward.
- False-positive penalty.
- Missed-defect penalty.
- Scan-cost penalty.
- Timeout penalty.
- Aggregate metrics across evaluated episodes.

## Reward Formula

```text
final_reward =
  identity_reward
  + concentration_reward
  + confidence_reward
  + false_positive_penalty
  + missed_defect_penalty
  + scan_cost_penalty
  + timeout_penalty
```

The implementation stores penalties as negative values in the reward breakdown. This makes logs easy to read: positive fields help the agent, negative fields hurt it.

## Validation Gate

Phase 4 is complete only when:

- Correct submissions score higher than wrong submissions.
- Expensive scans reduce reward.
- Missing defects reduce recall and reward.
- Overconfident wrong answers are penalized.
- Timeouts receive a timeout penalty.
- Invalid submission shapes raise clear errors.
- Aggregate metrics report mean reward and timeout rate.
- Phase 4 tests pass locally.

## Phase 5 Entry Criteria

Start Phase 5 only after reward tests pass. Phase 5 will implement the OpenEnv-compatible environment wrapper around the synthetic world and reward scorer.
