# Phase 3 Synthetic Materials World

## Purpose

Phase 3 implements the deterministic synthetic materials world required by the OpenEnv environment. This phase does not implement the OpenEnv server, reward scoring, PyTorch model, or training loop.

## Implemented Scope

- Candidate defect species list.
- Difficulty configurations for `easy`, `medium`, `hard`, and `expert`.
- Deterministic material case generation by seed.
- PDoS-like pristine spectra with host-family peak templates.
- Defective spectra with species-specific peak shifts, broadening, and local softening.
- Scan simulation for `quick_pdos`, `standard_pdos`, `high_res_pdos`, and `raman_proxy`.
- Resolution-dependent noise and smoothing.
- Frequency-band zoom scans.
- Tests for determinism, difficulty bounds, defect signal changes, scan behavior, and invalid inputs.

## Validation Gate

Phase 3 is complete only when:

- Synthetic cases are deterministic by seed.
- Difficulty configurations obey the Phase 2 contract.
- Defects measurably change the pristine spectrum.
- Scan simulation is deterministic for a fixed seed offset.
- Zoom scans obey requested frequency bounds.
- Invalid difficulty and invalid scan bands raise clear errors.
- Phase 3 tests pass locally.

## Phase 4 Entry Criteria

Start Phase 4 only after Phase 3 tests pass. Phase 4 will implement reward scoring and environment-facing metrics before the OpenEnv server is built.
