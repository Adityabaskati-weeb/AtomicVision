from __future__ import annotations

import pytest

from atomicvision.synthetic import DIFFICULTY_CONFIGS, generate_case, simulate_scan


def test_generate_case_is_deterministic() -> None:
    first = generate_case(seed=42, difficulty="medium")
    second = generate_case(seed=42, difficulty="medium")

    assert first == second


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
def test_difficulty_contract_bounds(difficulty: str) -> None:
    case = generate_case(seed=7, difficulty=difficulty)
    config = DIFFICULTY_CONFIGS[difficulty]

    assert config.min_defects <= len(case.defects) <= config.max_defects
    assert case.budget == config.budget
    assert case.max_steps == config.max_steps
    for defect in case.defects:
        assert config.min_concentration <= defect.concentration <= config.max_concentration


def test_defects_change_the_pristine_spectrum() -> None:
    case = generate_case(seed=11, difficulty="easy")

    total_delta = sum(
        abs(defective - pristine)
        for pristine, defective in zip(
            case.pristine_spectrum,
            case.defective_spectrum,
            strict=True,
        )
    )

    assert total_delta > 0.1


def test_simulated_scan_is_deterministic_for_same_seed_offset() -> None:
    case = generate_case(seed=99, difficulty="hard")

    first = simulate_scan(
        case,
        scan_mode="standard_pdos",
        resolution="medium",
        seed_offset=3,
    )
    second = simulate_scan(
        case,
        scan_mode="standard_pdos",
        resolution="medium",
        seed_offset=3,
    )

    assert first == second


def test_zoom_band_returns_only_requested_frequency_range() -> None:
    case = generate_case(seed=123, difficulty="medium")
    scan = simulate_scan(
        case,
        scan_mode="high_res_pdos",
        resolution="high",
        freq_min=5.0,
        freq_max=8.0,
    )

    assert scan.frequency_axis
    assert min(scan.frequency_axis) >= 5.0
    assert max(scan.frequency_axis) <= 8.0
    assert len(scan.frequency_axis) == len(scan.spectrum)


def test_invalid_difficulty_raises() -> None:
    with pytest.raises(ValueError, match="Unknown difficulty"):
        generate_case(seed=1, difficulty="impossible")


def test_invalid_scan_band_raises() -> None:
    case = generate_case(seed=5, difficulty="easy")

    with pytest.raises(ValueError, match="freq_min must be less"):
        simulate_scan(case, freq_min=8.0, freq_max=2.0)


def test_high_resolution_scan_has_lower_noise_than_quick_scan() -> None:
    case = generate_case(seed=77, difficulty="medium")

    quick = simulate_scan(case, scan_mode="quick_pdos", resolution="low")
    high_res = simulate_scan(case, scan_mode="high_res_pdos", resolution="high")

    assert high_res.noise_scale < quick.noise_scale
    assert high_res.cost > quick.cost

