"""OpenEnv client for AtomicVision."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from atomicvision_env.models import (
    AtomicVisionAction,
    AtomicVisionObservation,
    AtomicVisionState,
)


class AtomicVisionEnv(EnvClient[AtomicVisionAction, AtomicVisionObservation, AtomicVisionState]):
    """Client for a remote AtomicVision OpenEnv server."""

    def _step_payload(self, action: AtomicVisionAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[AtomicVisionObservation]:
        return StepResult(
            observation=AtomicVisionObservation(**payload.get("observation", {})),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> AtomicVisionState:
        return AtomicVisionState(**payload)
