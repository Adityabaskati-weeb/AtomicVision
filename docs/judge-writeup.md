# AtomicVision Judge Writeup

AtomicVision is an OpenEnv environment for non-destructive multi-defect mapping.
The agent receives compact spectral evidence, chooses low-cost characterization
actions, and submits a final defect map under scan-budget pressure.

## Deliverables

- Hugging Face Space: [prodigyhuh/atomicvision-openenv](https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv)
- Judge reproducible notebook: [notebooks/AtomicVision_Judge_Repro_Colab.ipynb](../notebooks/AtomicVision_Judge_Repro_Colab.ipynb)
- NaN-safe SFT trainer: [training/train_sft_atomicvision_safe.py](../training/train_sft_atomicvision_safe.py)
- Held-out evaluator: [training/evaluate_atomicvision_adapter.py](../training/evaluate_atomicvision_adapter.py)
- Adapter publisher: [training/publish_adapter_to_hub.py](../training/publish_adapter_to_hub.py)

## Why this project fits the hackathon

AtomicVision is not a static benchmark. It is a world-modeling task with:

- step-by-step tool use,
- verifiable outcomes,
- explicit cost tradeoffs,
- and a judge-visible deployment path.

The OpenEnv environment inherits the standard environment base class, exposes
real `reset` and `step` methods, and scores the final submission with layered
reward components for identity, concentration, confidence, and penalties.

## What we learned

The hardest failure mode was not scientific reasoning. It was interface
reliability. The model often chose the right action semantically, but failed to
serialize exact tool calls on held-out seeds.

That led to the current recovery path:

1. NaN-safe SFT validation,
2. schema-aware two-step curriculum,
3. strict plus normalized held-out evaluation,
4. only then GRPO continuation.

## Evidence

The repository now includes validator-facing training evidence as committed PNG
artifacts:

- [training-loss-curve.png](training-loss-curve.png)
- [training-reward-curve.png](training-reward-curve.png)

Those plots are also embedded in the main README so the automated pass can
reach them directly from the project root.
