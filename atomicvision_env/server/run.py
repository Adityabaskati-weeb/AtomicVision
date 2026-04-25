"""Local server runner for the AtomicVision OpenEnv Space."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the AtomicVision OpenEnv app with environment-configurable settings."""

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    workers = int(os.environ.get("WORKERS", "1"))
    reload = os.environ.get("RELOAD", "").lower() in {"1", "true", "yes", "on"}

    uvicorn.run(
        "atomicvision_env.server.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    main()
