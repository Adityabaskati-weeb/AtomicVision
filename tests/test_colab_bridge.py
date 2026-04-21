from __future__ import annotations

import json
from pathlib import Path


def test_colab_bridge_notebook_exists_and_targets_space() -> None:
    notebook_path = Path("notebooks/AtomicVision_GRPO_Colab.ipynb")

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )

    assert payload["nbformat"] == 4
    assert "AtomicVision GRPO Colab Bridge" in source_text
    assert "https://prodigyhuh-atomicvision-openenv.hf.space" in source_text
    assert "https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv" in source_text
    assert "wss://" in source_text
    assert "run_prior_submit_episode" in source_text
    assert "Phase 11 GRPO Fine-Tuning" in source_text
    assert "training/requirements-grpo.txt" in source_text
    assert "--dry-run --difficulty easy" in source_text
    assert "--preset smoke" in source_text
    assert "--preset colab-20" in source_text
    assert "--preset qwen-1p7b-50" in source_text
    assert "Qwen/Qwen3-1.7B" in source_text
