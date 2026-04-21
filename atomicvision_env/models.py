"""Typed OpenEnv models for AtomicVision."""

from __future__ import annotations

from typing import Literal

from openenv.core import Action, Observation, State
from pydantic import BaseModel, Field


ActionType = Literal[
    "request_scan",
    "zoom_band",
    "compare_reference",
    "ask_prior",
    "submit_defect_map",
]

Resolution = Literal["low", "medium", "high"]
ScanMode = Literal["quick_pdos", "standard_pdos", "high_res_pdos", "raman_proxy"]


class ScanRecord(BaseModel):
    """One scan or lab-tool event in the episode history."""

    action_type: str
    scan_mode: str | None = None
    resolution: str | None = None
    cost: float = 0.0
    freq_min: float | None = None
    freq_max: float | None = None


class PriorPrediction(BaseModel):
    """DefectNet-lite prior shown to the agent."""

    predicted_defects: list[str] = Field(default_factory=list)
    predicted_concentrations: list[float] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "heuristic"
    checkpoint_path: str | None = None


class AtomicVisionAction(Action):
    """Action accepted by the AtomicVision lab environment."""

    action_type: ActionType
    scan_mode: ScanMode | None = None
    resolution: Resolution | None = None
    freq_min: float | None = None
    freq_max: float | None = None
    predicted_defects: list[str] = Field(default_factory=list)
    predicted_concentrations: list[float] = Field(default_factory=list)
    confidence: float | None = None


class AtomicVisionObservation(Observation):
    """Observation returned by the AtomicVision lab environment."""

    episode_id: str
    material_id: str
    difficulty: str
    host_family: str
    frequency_axis: list[float]
    current_spectrum: list[float]
    pristine_reference: list[float] | None = None
    scan_history: list[ScanRecord] = Field(default_factory=list)
    candidate_defects: list[str] = Field(default_factory=list)
    prior_prediction: PriorPrediction | None = None
    budget_remaining: float
    step_count: int
    max_steps: int
    last_reward: float = 0.0
    reward_breakdown: dict[str, float] | None = None
    message: str = ""


class AtomicVisionState(State):
    """Internal environment state exposed through OpenEnv state()."""

    seed: int | None = None
    difficulty: str | None = None
    material_id: str | None = None
    done: bool = False
    budget_remaining: float = 0.0
    total_scan_cost: float = 0.0
    max_steps: int = 0
    reward_history: list[float] = Field(default_factory=list)
