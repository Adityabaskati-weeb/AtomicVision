# Phase 9 Hugging Face Space Deployment

## Purpose

Phase 9 prepares and deploys AtomicVision as a Hugging Face Docker Space. The hosted Space is the judge-facing OpenEnv environment endpoint.

## Space Target

- Namespace: `prodigyhuh`
- Space name: `atomicvision-openenv`
- Space id: `prodigyhuh/atomicvision-openenv`
- SDK: Docker
- App port: `7860`

## Implemented Scope

- Hugging Face Space YAML metadata in `README.md`.
- Dockerfile for the OpenEnv FastAPI app.
- Installable package metadata in `pyproject.toml`.
- Reproducible dependency lockfile in `uv.lock`.
- Runtime dependencies in `requirements.txt`.
- OpenEnv app health-route smoke test.
- Deployment target documented.

## Runtime Command

```text
server
```

Equivalent direct command:

```text
uvicorn atomicvision_env.server.app:app --host 0.0.0.0 --port 7860
```

## Space Components

AtomicVision now exposes the three HF Space components highlighted in the
OpenEnv submission deck:

1. **Server endpoint**
   - Public host: `https://prodigyhuh-atomicvision-openenv.hf.space`
   - Key routes: `/health`, `/docs`, `/ws`, `/reset`, `/step`, `/state`
2. **Installable repository package**
   - Source package metadata: `pyproject.toml`
   - Public install command:
     `pip install git+https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv`
3. **Registry image**
   - Provided automatically by the Docker Space
   - The exact `registry.hf.space/...` image string should be copied from the
     Space's `Run with Docker` button

## Local Development

Clone and run the Space locally:

```text
git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv
cd atomicvision-openenv
uv sync --frozen
uv run server
```

The checked-in `uv.lock` is the pinned dependency graph for this local flow.

Or run the server entry point directly:

```text
uvicorn atomicvision_env.server.app:app --host 0.0.0.0 --port 7860 --workers 4
```

Example client usage from the installable Space package:

```python
import asyncio

from atomicvision_env import AtomicVisionAction, AtomicVisionEnv


async def main() -> None:
    async with AtomicVisionEnv(
        base_url="https://prodigyhuh-atomicvision-openenv.hf.space"
    ) as client:
        await client.reset()
        result = await client.step(AtomicVisionAction(action_type="ask_prior"))
        print(result)


asyncio.run(main())
```

Local container flow from the current repo:

```text
docker build -t atomicvision-openenv:latest .
docker run -d -p 7860:7860 --name atomicvision-openenv atomicvision-openenv:latest
```

## CLI Deployment

Typical OpenEnv CLI flow:

```text
openenv init my_env
cd my_env
openenv push
openenv push --repo-id prodigyhuh/atomicvision-openenv
```

## Validation Gate

Phase 9 is complete. Completed validation:

- The OpenEnv app imports locally.
- `/health` returns a successful response locally through FastAPI test client.
- Full test suite passes locally.
- The Hugging Face Space repository exists.
- The current repo uploads to the Space successfully.
- The Space build completed and reached `RUNNING`.
- Public `/health` returns `200`.
- Public `/reset` returns a real AtomicVision observation.

## Deployment Links

- Space: https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv
- Public app host: https://prodigyhuh-atomicvision-openenv.hf.space
- Last deployed commit: `d2f89fe31e2fe9aed7dd42beab5638493e9d6c40`

## Phase 10 Entry Criteria

Start Phase 10 only after deployment is confirmed. Phase 10 should create judge-facing reward comparison assets and the TRL/Unsloth Colab bridge.
