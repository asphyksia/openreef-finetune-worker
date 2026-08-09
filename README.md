# openreef-finetune-worker

Public OpenReef **provider finetune worker** sources and GHCR image builds.

| Item | Value |
|------|--------|
| Images | `ghcr.io/asphyksia/finetune-worker:cuda-*` / `rocm-*` |
| Product monorepo | [asphyksia/OpenReef](https://github.com/asphyksia/OpenReef) (private app/deploy) |
| ADR | OpenReef `docs/worker-repo-split.md` |

This repository exists so CUDA/ROCm image builds can use **public Actions** minutes without opening the product monorepo, and so providers can audit the worker that runs on their GPUs.

## What this is / is not

**Is:** Dockerfiles + Python worker that trains LoRA/QLoRA adapters for OpenReef jobs (OGPU network source).

**Is not:** OpenReef backend, frontend, Stripe, admin, operator vault, or any secrets. Configure runtime with environment variables only (see compose examples).

## Images

| Tag | Meaning |
|-----|---------|
| `cuda-<git-sha>` | Immutable CUDA build (preferred for pins) |
| `rocm-<git-sha>` | Immutable ROCm build |
| `cuda-latest` / `rocm-latest` | Rolling (only updated from protected `main` after smoke) |

**Pin after smoke:**

```bash
export FINETUNE_IMAGE=ghcr.io/asphyksia/finetune-worker@sha256:<digest>
```

## Build locally

```bash
# NVIDIA
./build_image.sh --cuda
# AMD
./build_image.sh --rocm
```

No GPU required to **build**; training needs a matching GPU at runtime.

## CI policy

- **Pull requests:** build CUDA image, **do not** push to GHCR.
- **Push to `main`:** push `cuda-<sha>` (and `rocm-<sha>` when enabled).  
  `:cuda-latest` is published only when workflow input `publish_latest=true` (or after cutover policy is flipped).
- Branch protection should require the PR smoke check before merge.

## License

MIT — see [LICENSE](LICENSE). Runtime depends on third-party packages (PyTorch, Unsloth/Axolotl, OGPU SDK, etc.) under their own licenses.

## Security

- Never commit `.env`, private keys, or tokens.
- `PROVIDER_PRIVATE_KEY` belongs in the **signer** sidecar (or provider-app env), not in this image.
- Report security issues privately to the OpenReef operators.
