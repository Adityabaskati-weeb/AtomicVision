# Phase 0 Scope Lock

## Project Identity

**Title:** AtomicVision: An Autonomous AI Agent for Non-Destructive Multi-Defect Mapping

**Theme:** Theme #3.1, World Modeling / Professional Tasks

**One-line pitch:** AtomicVision is a simulated scientific characterization lab where an AI agent learns to map hidden atomic defects from non-invasive vibrational spectra while balancing accuracy, uncertainty, and scan cost.

Locked identifiers:

- Title: AtomicVision
- Full title: AtomicVision: An Autonomous AI Agent for Non-Destructive Multi-Defect Mapping
- Theme: Theme #3.1, World Modeling / Professional Tasks
- One-line pitch: AtomicVision is a simulated scientific characterization lab where an AI agent learns to map hidden atomic defects from non-invasive vibrational spectra while balancing accuracy, uncertainty, and scan cost.

## Problem Statement

Advanced materials used in semiconductors, batteries, solar cells, alloys, and quantum devices often depend on carefully controlled atomic-scale defects. These defects can improve conductivity, strength, thermal behavior, and other functional properties. The difficulty is verification: once a material is produced, its internal defect profile is hard to measure without damaging the sample or relying on incomplete indirect measurements.

AtomicVision turns this into an agentic world-modeling task. The true defect state is hidden. The agent only sees indirect spectral evidence and must decide which characterization actions to take before submitting a final defect map.

## Product Goal

Build an OpenEnv-compliant environment that demonstrates autonomous non-destructive defect characterization in a synthetic but physically inspired materials lab. The system should be credible to AI/ML judges, understandable to non-specialists, and complete enough to show measurable reward improvement.

## Hackathon Requirements Alignment

| Requirement | Phase 0 Decision |
| --- | --- |
| Use latest OpenEnv release | Target `openenv-core` latest verified release at planning time: `0.2.3` from PyPI. Re-check before implementation. |
| OpenEnv-compliant environment | Build around `reset()`, `step(action)`, and `state()` with typed action and observation models. |
| Minimal training script in Colab | Provide a TRL or Unsloth GRPO notebook after the environment is stable. |
| Hugging Face Space hosting | Deploy as an OpenEnv/HF Space, likely Docker-based because OpenEnv serves an environment backend. |
| Mini-blog or mini-video | Prepare a short HF blog or under-2-minute video after MVP metrics are ready. |

## Judging Strategy

| Judging Criterion | Weight | AtomicVision Strategy |
| --- | ---: | --- |
| Environment Innovation | 40% | A scientific lab environment where an agent actively chooses scans and infers hidden atomic defects from indirect evidence. |
| Storytelling | 30% | Explain the problem as: inspect expensive high-tech materials without destroying them. |
| Reward Improvement | 20% | Show random or heuristic baseline versus trained agent reward curves and before/after behavior. |
| Reward and Training Pipeline | 10% | Use a transparent reward formula based on defect identity accuracy, concentration error, confidence, and scan cost. |

## MVP Scope

The MVP must include:

- Synthetic PDoS-like spectral data generator.
- Hidden material state with 0-6 possible defect species.
- Agent actions for requesting scans, zooming into spectral bands, comparing references, asking for a model prior, and submitting a final defect map.
- Reward logic that makes the agent balance accuracy and measurement cost.
- OpenEnv server/client package.
- Minimal PyTorch spectral predictor or heuristic prior.
- TRL or Unsloth-compatible training notebook.
- Visible reward improvement.
- Hugging Face Space deployment.
- README, short pitch, and blog/video artifact.

## Explicit Non-Goals

AtomicVision will not claim to solve full industrial atomic defect characterization during the hackathon.

AtomicVision will not claim access to the original DefectNet training dataset unless that dataset is actually integrated and licensed for use.

AtomicVision will not depend on neutron-scattering facility data for the MVP.

AtomicVision will not start as a plain classifier demo. The primary deliverable is an OpenEnv agent environment.

## Scientific Positioning

AtomicVision is research-inspired, simulation-backed, and agentic. The inspiration is non-destructive defect identification from vibrational spectra, especially PDoS-based defect inference. The hackathon version will simulate spectra and defect effects while preserving the core challenge: inferring hidden defects from indirect, noisy measurements.

## Agentic Environment Concept

Each episode represents one unknown material sample.

The agent observes:

- Current vibrational spectrum.
- Optional pristine reference spectrum.
- Host material metadata.
- Scan history.
- Noise and resolution metadata.
- Remaining scan budget.

The agent can act:

- Request a scan.
- Request a higher-resolution scan.
- Zoom into a frequency band.
- Compare against a pristine reference.
- Ask for a DefectNet-lite prior.
- Submit defect identities, concentrations, and confidence.

The episode ends when:

- The agent submits a final defect map.
- The step limit is reached.
- The scan budget is exhausted.

## Reward Philosophy

The reward must teach scientific investigation, not just final guessing.

Reward should increase for:

- Correct defect identity.
- Accurate concentration estimates.
- Correctly calibrated confidence.
- Efficient use of scans.

Reward should decrease for:

- False-positive defects.
- Missed defects.
- Large concentration error.
- Unnecessary expensive scans.
- Failing to submit before the episode limit.

## Success Metrics

Primary metrics:

- Mean episode reward.
- Defect identity F1.
- Concentration mean absolute error.
- Scan cost per solved episode.

Demo metrics:

- Baseline reward versus trained reward.
- Before/after agent action trace.
- Easy/medium/hard difficulty comparison.

## Phase 0 Completion Gate

Phase 0 is complete only when:

- Project title is locked.
- Theme is locked.
- One-line pitch is locked.
- MVP scope is locked.
- Non-goals are documented.
- Judging criteria are mapped to product decisions.
- Risk register is documented.
- Source assumptions are documented.
- Repo contains a readable Phase 0 artifact.
- Repo status is clean except intentional Phase 0 documentation changes.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Project becomes only a classifier | High | Keep OpenEnv actions and reward loop central. |
| Synthetic data looks fake | High | Use physically inspired peak shifts, broadening, noise, and overlapping defect signatures. |
| Reward does not improve | High | Include a simple heuristic baseline and curriculum before LLM training. |
| OpenEnv API changes | Medium | Re-check latest OpenEnv docs before Phase 5 implementation. |
| HF Space deployment friction | Medium | Keep CPU-compatible MVP and use Docker Space only if needed. |
| Pitch becomes too technical | Medium | Lead with manufacturing problem, then show spectra and reward improvement. |
| Time pressure | High | Build in MVP-first order; push stretch features only after deployment works. |

## Source Assumptions Checked

- OpenEnv latest PyPI release checked on April 21, 2026: `openenv-core 0.2.3`.
- OpenEnv docs describe the environment interface around `reset()`, `step()`, and `state()`.
- TRL docs describe OpenEnv integration for stateful multi-turn environment training.
- Hugging Face Docker Spaces support backend apps outside standard Gradio/Streamlit.
- DefectNet-style research reports PDoS-based inference of multiple substitutional point defects from vibrational spectra.

## Phase 1 Entry Criteria

Start Phase 1 only after this document is reviewed and accepted as the project contract. Phase 1 will design the full system architecture, module boundaries, and data flow before any environment or model code is written.
