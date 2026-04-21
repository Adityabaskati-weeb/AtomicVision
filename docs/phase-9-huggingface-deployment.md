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
- Runtime dependencies in `requirements.txt`.
- OpenEnv app health-route smoke test.
- Deployment target documented.

## Runtime Command

```text
uvicorn atomicvision_env.server.app:app --host 0.0.0.0 --port 7860
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
