"""FastAPI app entrypoint for the AtomicVision OpenEnv server."""

from __future__ import annotations

from openenv.core import create_app

from atomicvision_env.models import AtomicVisionAction, AtomicVisionObservation
from atomicvision_env.server.environment import AtomicVisionEnvironment


app = create_app(
    lambda: AtomicVisionEnvironment(),
    AtomicVisionAction,
    AtomicVisionObservation,
    env_name="atomicvision_env",
    max_concurrent_envs=32,
)

