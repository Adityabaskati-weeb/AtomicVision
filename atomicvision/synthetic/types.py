"""Typed data structures for the synthetic materials world."""

from __future__ import annotations

from dataclasses import dataclass


CANDIDATE_DEFECTS: tuple[str, ...] = (
    "B",
    "C",
    "N",
    "O",
    "Al",
    "Si",
    "P",
    "S",
    "Ge",
    "Ga",
    "Mg",
    "Zn",
)


@dataclass(frozen=True)
class DifficultyConfig:
    """Difficulty settings for one synthetic characterization episode."""

    name: str
    min_defects: int
    max_defects: int
    min_concentration: float
    max_concentration: float
    noise_scale: float
    budget: float
    max_steps: int
    distractors: int = 0


DIFFICULTY_CONFIGS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        name="easy",
        min_defects=1,
        max_defects=1,
        min_concentration=0.08,
        max_concentration=0.25,
        noise_scale=0.012,
        budget=10.0,
        max_steps=5,
    ),
    "medium": DifficultyConfig(
        name="medium",
        min_defects=2,
        max_defects=3,
        min_concentration=0.02,
        max_concentration=0.20,
        noise_scale=0.024,
        budget=9.0,
        max_steps=6,
    ),
    "hard": DifficultyConfig(
        name="hard",
        min_defects=4,
        max_defects=6,
        min_concentration=0.005,
        max_concentration=0.15,
        noise_scale=0.040,
        budget=8.0,
        max_steps=7,
    ),
    "expert": DifficultyConfig(
        name="expert",
        min_defects=4,
        max_defects=6,
        min_concentration=0.002,
        max_concentration=0.12,
        noise_scale=0.050,
        budget=6.0,
        max_steps=7,
        distractors=2,
    ),
}


@dataclass(frozen=True)
class Defect:
    """One hidden substitutional defect in a synthetic material."""

    species: str
    concentration: float


@dataclass(frozen=True)
class MaterialCase:
    """Complete hidden synthetic material case."""

    seed: int
    material_id: str
    difficulty: str
    host_family: str
    frequency_axis: tuple[float, ...]
    pristine_spectrum: tuple[float, ...]
    defective_spectrum: tuple[float, ...]
    defects: tuple[Defect, ...]
    candidate_defects: tuple[str, ...]
    budget: float
    max_steps: int


@dataclass(frozen=True)
class ScanResult:
    """Observed scan returned to an agent."""

    scan_mode: str
    resolution: str
    cost: float
    frequency_axis: tuple[float, ...]
    spectrum: tuple[float, ...]
    noise_scale: float
    freq_min: float | None = None
    freq_max: float | None = None

