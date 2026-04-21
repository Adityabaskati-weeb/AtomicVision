"""Synthetic materials world for AtomicVision."""

from atomicvision.synthetic.generator import generate_case, simulate_scan
from atomicvision.synthetic.types import (
    CANDIDATE_DEFECTS,
    DIFFICULTY_CONFIGS,
    Defect,
    DifficultyConfig,
    MaterialCase,
    ScanResult,
)

__all__ = [
    "CANDIDATE_DEFECTS",
    "DIFFICULTY_CONFIGS",
    "Defect",
    "DifficultyConfig",
    "MaterialCase",
    "ScanResult",
    "generate_case",
    "simulate_scan",
]

