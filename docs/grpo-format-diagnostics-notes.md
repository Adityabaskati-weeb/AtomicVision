# GRPO Format Diagnostics Notes

These notes capture the two short hard-only GRPO probes run after we stopped guessing and added explicit completion-format diagnostics.

## V10b: Strip-think on Main Baseline

- Job: [69ec8952d70108f37acde3a5](https://huggingface.co/jobs/prodigyhuh/69ec8952d70108f37acde3a5)
- Commit: `42b25cf919fedd57e3522fbc5262d04a47c0ef01`
- Metrics: [hard-only-grpo-reference-probe-stripthink-main-v10b-metrics.json](./hard-only-grpo-reference-probe-stripthink-main-v10b-metrics.json)

Key result:

- The parser fix preserved the healthier baseline behavior:
  - `done_rate = 0.75`
  - `submit_tool_rate = 0.25`
  - `normalized_tool_call_pass_rate = 1.0`
  - `strict_tool_call_pass_rate = 0.0`

Diagnostic result:

- `raw_tool_call_tag_rate = 0.0`
- `repaired_without_tool_tags_rate = 1.0`
- `stripped_think_wrapper_rate = 0.0`

Interpretation:

- The dominant strict-format failure is **not** empty Qwen `<think>` wrappers.
- The model is usually emitting **tagless repaired outputs** such as a bare tool name plus arguments.

## V11b: Stronger Penalty for Tagless Repaired Outputs

- Job: [69ec8b0fd70108f37acde3b9](https://huggingface.co/jobs/prodigyhuh/69ec8b0fd70108f37acde3b9)
- Commit: `8e2aa2b1191539c697baf914a98690a9d8c40b64`
- Metrics: [hard-only-grpo-reference-probe-tagless-penalty-v11b-metrics.json](./hard-only-grpo-reference-probe-tagless-penalty-v11b-metrics.json)

Key result:

- The stronger tagless penalty changed the diagnostic mix, but not the main gate metrics:
  - `done_rate = 0.75`
  - `submit_tool_rate = 0.25`
  - `normalized_tool_call_pass_rate = 1.0`
  - `strict_tool_call_pass_rate = 0.0`

Diagnostic result:

- `raw_tool_call_tag_rate = 0.25`
- `repaired_without_tool_tags_rate = 0.75`
- `repaired_with_tool_tags_rate = 0.25`

Interpretation:

- Penalizing tagless repaired outputs more strongly is **not enough by itself** in a 4-step probe.
- The model begins to show some wrapper usage, but it does not convert into strict final submissions yet.

## Current Conclusion

The next meaningful strict-format intervention should target **generation behavior**, not just parsing or reward accounting. The evidence so far says:

1. wrapper stripping was a correct parser hardening fix
2. the live failure mode is mostly tagless repaired output
3. a stronger penalty alone does not produce strict tool calls in a short GRPO continuation
