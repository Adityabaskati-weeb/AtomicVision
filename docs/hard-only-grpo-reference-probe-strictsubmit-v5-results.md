# Hard-Only GRPO Strict-Submit Contract Probe

This page records the follow-up Hugging Face Jobs probe that tested the
strict-submit contract patch from branch `codex-strict-submit-contract-probe`.

## Goal

Test whether a narrower prompt + reward change can improve strict terminal
`submit_defect_map` behavior without breaking the useful signal recovered by the
previous short hard-only GRPO probe.

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ec79d0d2c8bd8662bcd53a](https://huggingface.co/jobs/prodigyhuh/69ec79d0d2c8bd8662bcd53a)
- Hardware: `a10g-large`
- Branch: `codex-strict-submit-contract-probe`
- Commit under test: `8f3fbea582d8998957ad34ab86dc8d7980acfb78`
- Prompt focus: `reference-improvement`
- Difficulty: `hard`
- Seed band: `4000-7999`
- Output mode: no push-to-hub, metrics persisted in the HF job logs

## Final Probe Metrics

| Metric | Value |
| --- | ---: |
| `reward_std` | `0.00` |
| `frac_reward_zero_std` | `1.00` |
| `done_rate` | `0.00` |
| `normalized_tool_call_pass_rate` | `0.50` |
| `normalized_tool_call_repair_rate` | `0.50` |
| `submit_tool_rate` | `0.50` |
| `strict_tool_call_pass_rate` | `0.00` |
| `strict_submit_reward_mean` | `0.00` |
| `tools/failure_frequency` | `0.00` |
| `format_reward_mean` | `-0.05` |
| `process_shaping_reward_mean` | `-0.05` |
| `total_reward_mean` | `-0.65` |
| `train_loss` | `3.89e-10` |
| `train_runtime_s` | `77.1551` |

## Comparison Against The Previous Probe

| Metric | Previous probe | Strict-submit probe | Read |
| --- | ---: | ---: | --- |
| `reward_std` | `0.5065` | `0.00` | Regressed badly; the learning signal collapsed |
| `done_rate` | `0.75` | `0.00` | Regressed badly; the policy no longer reached terminal success |
| `normalized_tool_call_pass_rate` | `1.00` | `0.50` | Worse |
| `submit_tool_rate` | `0.25` | `0.50` | Better |
| `strict_tool_call_pass_rate` | `0.00` | `0.00` | Still unresolved |
| `tools/failure_frequency` | `0.00` | `0.00` | Stable |

## Interpretation

This probe answered a narrower question cleanly:

- good: `submit_tool_rate` improved from `0.25` to `0.50`
- good: tool failures stayed at `0.00`
- bad: the improvement came with a collapse in `done_rate`
- bad: `reward_std` fell to `0.00`, which removes useful GRPO learning signal
- bad: strict terminal submit behavior still did not land

So the strict-submit contract patch is **informative but not merge-ready as a
training default**. It did push the model toward submitting more often, but it
also pushed the short probe into a much more brittle regime.

## Promotion Decision

- Probe status: **completed**
- Probe quality: **informative regression**
- Promotion status: **not promoted**
- Main-branch action: **do not merge this branch into `main` yet**

## Recommended Next Step

The next fix should target the strict-submit gap with less behavioral pressure
than this patch. The most likely next move is to keep the repaired terminal
parsing from `main`, but soften the prompt / reward contract so the model does
not skip `ask_prior` and lose reward variance entirely.

## Stored Artifact

The machine-readable summary for this run is committed at:

- [hard-only-grpo-reference-probe-strictsubmit-v5-metrics.json](hard-only-grpo-reference-probe-strictsubmit-v5-metrics.json)
