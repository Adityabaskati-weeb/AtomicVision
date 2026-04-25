"""FastAPI app entrypoint for the AtomicVision OpenEnv server."""

from __future__ import annotations

from fastapi.responses import HTMLResponse
from openenv.core import create_app

from atomicvision_env.models import AtomicVisionAction, AtomicVisionObservation
from atomicvision_env.server.environment import AtomicVisionEnvironment
from atomicvision_env.server.frontend import render_home_html


app = create_app(
    lambda: AtomicVisionEnvironment(),
    AtomicVisionAction,
    AtomicVisionObservation,
    env_name="atomicvision_env",
    max_concurrent_envs=32,
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Browser workbench for the Hugging Face Space App tab."""

    return render_home_html()
