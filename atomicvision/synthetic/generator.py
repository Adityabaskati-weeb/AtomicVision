"""Synthetic PDoS-like spectra for AtomicVision.

The generator is intentionally lightweight and deterministic. It creates
physically inspired spectra with Gaussian-like vibrational peaks, then injects
defect signatures as subtle shifts, broadening, and additional local peaks.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

from atomicvision.synthetic.types import (
    CANDIDATE_DEFECTS,
    DIFFICULTY_CONFIGS,
    Defect,
    DifficultyConfig,
    MaterialCase,
    ScanResult,
)


HOST_FAMILIES: tuple[str, ...] = (
    "silicon_family",
    "wide_bandgap",
    "oxide_ceramic",
    "thermoelectric",
)

SCAN_COSTS: dict[str, float] = {
    "quick_pdos": 1.0,
    "standard_pdos": 2.0,
    "high_res_pdos": 4.0,
    "raman_proxy": 2.5,
}

RESOLUTION_NOISE_MULTIPLIER: dict[str, float] = {
    "low": 1.40,
    "medium": 1.00,
    "high": 0.55,
}

RESOLUTION_SMOOTHING_WINDOW: dict[str, int] = {
    "low": 5,
    "medium": 3,
    "high": 1,
}


def generate_case(
    seed: int,
    difficulty: str = "medium",
    points: int = 128,
    max_frequency: float = 20.0,
) -> MaterialCase:
    """Generate one deterministic hidden material case."""

    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    if points < 16:
        raise ValueError("points must be at least 16")

    config = DIFFICULTY_CONFIGS[difficulty]
    rng = random.Random(seed)
    host_family = rng.choice(HOST_FAMILIES)
    frequency_axis = _frequency_axis(points, max_frequency)
    pristine = _normalize(_host_spectrum(frequency_axis, host_family, rng))
    defects = _sample_defects(rng, config)
    defective = _normalize(_inject_defects(frequency_axis, pristine, defects))
    candidate_defects = _candidate_set(rng, defects, config)

    return MaterialCase(
        seed=seed,
        material_id=f"synthetic-{difficulty}-{seed}",
        difficulty=difficulty,
        host_family=host_family,
        frequency_axis=tuple(frequency_axis),
        pristine_spectrum=tuple(pristine),
        defective_spectrum=tuple(defective),
        defects=tuple(defects),
        candidate_defects=tuple(candidate_defects),
        budget=config.budget,
        max_steps=config.max_steps,
    )


def simulate_scan(
    case: MaterialCase,
    scan_mode: str = "quick_pdos",
    resolution: str = "low",
    freq_min: float | None = None,
    freq_max: float | None = None,
    seed_offset: int = 0,
) -> ScanResult:
    """Return a deterministic noisy scan for a generated material case."""

    if scan_mode not in SCAN_COSTS:
        raise ValueError(f"Unknown scan mode: {scan_mode}")
    if resolution not in RESOLUTION_NOISE_MULTIPLIER:
        raise ValueError(f"Unknown resolution: {resolution}")
    if (freq_min is None) != (freq_max is None):
        raise ValueError("freq_min and freq_max must be provided together")
    if freq_min is not None and freq_max is not None and freq_min >= freq_max:
        raise ValueError("freq_min must be less than freq_max")

    axis = list(case.frequency_axis)
    spectrum = list(case.defective_spectrum)

    if freq_min is not None and freq_max is not None:
        paired = [
            (freq, value)
            for freq, value in zip(axis, spectrum, strict=True)
            if freq_min <= freq <= freq_max
        ]
        if not paired:
            raise ValueError("frequency band does not overlap the spectrum")
        axis = [freq for freq, _ in paired]
        spectrum = [value for _, value in paired]

    config = DIFFICULTY_CONFIGS[case.difficulty]
    mode_multiplier = 0.75 if scan_mode == "high_res_pdos" else 1.0
    if scan_mode == "raman_proxy":
        spectrum = _raman_proxy_transform(spectrum)
        mode_multiplier = 1.15

    noise_scale = (
        config.noise_scale
        * RESOLUTION_NOISE_MULTIPLIER[resolution]
        * mode_multiplier
    )
    smoothed = _moving_average(
        spectrum,
        RESOLUTION_SMOOTHING_WINDOW[resolution],
    )
    rng = random.Random(case.seed + 10_000 + seed_offset)
    observed = _clip_nonnegative(
        value + rng.gauss(0.0, noise_scale) for value in smoothed
    )
    observed = _normalize(observed)

    return ScanResult(
        scan_mode=scan_mode,
        resolution=resolution,
        cost=SCAN_COSTS[scan_mode],
        frequency_axis=tuple(axis),
        spectrum=tuple(observed),
        noise_scale=noise_scale,
        freq_min=freq_min,
        freq_max=freq_max,
    )


def _frequency_axis(points: int, max_frequency: float) -> list[float]:
    step = max_frequency / (points - 1)
    return [round(i * step, 6) for i in range(points)]


def _host_spectrum(
    frequency_axis: Iterable[float],
    host_family: str,
    rng: random.Random,
) -> list[float]:
    templates = {
        "silicon_family": ((3.0, 0.7, 0.9), (8.5, 1.1, 0.75), (14.0, 1.4, 0.5)),
        "wide_bandgap": ((4.5, 0.8, 0.8), (11.0, 1.0, 0.85), (17.0, 1.1, 0.45)),
        "oxide_ceramic": ((2.0, 0.6, 0.55), (9.0, 1.5, 0.7), (16.2, 1.0, 0.95)),
        "thermoelectric": ((1.5, 0.8, 0.8), (6.5, 1.3, 0.85), (12.5, 1.7, 0.6)),
    }
    peaks = templates[host_family]
    values: list[float] = []
    for freq in frequency_axis:
        baseline = 0.025 + 0.015 * math.sin(freq * 0.8)
        value = baseline
        for center, width, amplitude in peaks:
            jittered_center = center + rng.uniform(-0.08, 0.08)
            value += amplitude * _gaussian(freq, jittered_center, width)
        values.append(max(value, 0.0))
    return values


def _sample_defects(rng: random.Random, config: DifficultyConfig) -> list[Defect]:
    count = rng.randint(config.min_defects, config.max_defects)
    species = rng.sample(list(CANDIDATE_DEFECTS), count)
    defects = []
    for symbol in species:
        concentration = rng.uniform(
            config.min_concentration,
            config.max_concentration,
        )
        defects.append(Defect(symbol, round(concentration, 5)))
    return defects


def _candidate_set(
    rng: random.Random,
    defects: list[Defect],
    config: DifficultyConfig,
) -> list[str]:
    required = [defect.species for defect in defects]
    remaining = [symbol for symbol in CANDIDATE_DEFECTS if symbol not in required]
    distractor_count = max(config.distractors, 3)
    distractors = rng.sample(remaining, min(distractor_count, len(remaining)))
    combined = required + distractors
    rng.shuffle(combined)
    return combined


def _inject_defects(
    frequency_axis: Iterable[float],
    pristine: list[float],
    defects: list[Defect],
) -> list[float]:
    defective = list(pristine)
    for defect in defects:
        signature = _species_signature(defect.species)
        strength = 2.8 * defect.concentration
        for index, freq in enumerate(frequency_axis):
            added_peak = strength * signature["amplitude"] * _gaussian(
                freq,
                signature["center"],
                signature["width"],
            )
            local_softening = 0.45 * strength * _gaussian(
                freq,
                signature["soften_center"],
                signature["soften_width"],
            )
            broadening = 0.18 * strength * _gaussian(
                freq,
                signature["broad_center"],
                signature["broad_width"],
            )
            defective[index] = max(
                defective[index] + added_peak + broadening - local_softening,
                0.0,
            )
    return defective


def _species_signature(species: str) -> dict[str, float]:
    code = sum((i + 1) * ord(ch) for i, ch in enumerate(species))
    center = 1.2 + (code % 170) / 10.0
    soften_center = 1.0 + ((code * 7) % 180) / 10.0
    broad_center = 2.0 + ((code * 11) % 150) / 10.0
    return {
        "center": min(center, 19.2),
        "width": 0.22 + (code % 5) * 0.08,
        "amplitude": 0.75 + (code % 7) * 0.09,
        "soften_center": min(soften_center, 19.0),
        "soften_width": 0.45 + (code % 4) * 0.12,
        "broad_center": min(broad_center, 18.5),
        "broad_width": 0.9 + (code % 6) * 0.18,
    }


def _gaussian(freq: float, center: float, width: float) -> float:
    return math.exp(-0.5 * ((freq - center) / width) ** 2)


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    radius = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def _raman_proxy_transform(values: list[float]) -> list[float]:
    transformed = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else value
        right = values[index + 1] if index < len(values) - 1 else value
        transformed.append(abs(right - left) + 0.65 * value)
    return transformed


def _clip_nonnegative(values: Iterable[float]) -> list[float]:
    return [max(value, 0.0) for value in values]


def _normalize(values: Iterable[float]) -> list[float]:
    materialized = list(values)
    max_value = max(materialized) if materialized else 0.0
    if max_value <= 0.0:
        return [0.0 for _ in materialized]
    return [round(value / max_value, 6) for value in materialized]

