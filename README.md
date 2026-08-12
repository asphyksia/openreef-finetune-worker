# openreef-finetune-worker

Public OpenReef **provider finetune worker** sources and GHCR image builds.

| Item | Value |
|------|--------|
| Images | `cuda-*` (Unsloth), `rocm-*` (Axolotl), `cuda-multigpu-*` (Axolotl) |
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
| `rocm-<git-sha>` | Immutable AMD ROCm Axolotl build |
| `cuda-multigpu-<git-sha>` | Immutable NVIDIA Axolotl multi-GPU build |
| `cuda-latest` / `rocm-latest` / `cuda-multigpu-latest` | Rolling; explicit promotion only after matching GPU smoke |

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

# NVIDIA multi-GPU (Axolotl, explicit opt-in)
./build_image.sh --cuda-multigpu
```

No GPU required to **build**; training needs a matching GPU at runtime.

## Engine policy

| Hardware | Engine | Image |
|----------|--------|-------|
| NVIDIA, one GPU | Unsloth | `cuda-*` |
| AMD ROCm | Axolotl | `rocm-*` |
| NVIDIA, two or more GPUs | Axolotl through Accelerate | `cuda-multigpu-*` |

User modes are independent from provider hardware:

| Mode | Provider eligibility | Contract |
|------|----------------------|----------|
| Fast | AMD or NVIDIA | Fixed 1-epoch iteration preset |
| Balanced | AMD or NVIDIA | Fixed 2-epoch recommended baseline |
| Custom | NVIDIA, exactly one GPU | Validated expert controls; not a quality tier |

Custom exposes LoRA, optimization, evaluation and checkpoint controls through
Unsloth core. It fails closed on AMD and multi-GPU providers. The retired
Quality experiment is historical evidence only and cannot be created.

AMD Unsloth remains a local experiment. It is not installed in the ROCm
product image. Multi-GPU is explicit through
`docker-compose-nvidia-multigpu.yml`; the normal NVIDIA compose reserves one GPU.

RDNA4 `gfx1200` has a narrow local safety rule: the standard `balanced`
r32/alpha64 shape is resolved to r64/alpha128 while keeping balanced epochs and
learning rate. Crossed local evidence showed quality+r32 page-faulting at step
51 while balanced+r64 completed 502/502. A second r64 pass on the final local
image also completed 502/502; both adapters passed a three-prompt serve smoke
with finite loss. This does not waive the consecutive-run gate required before
publication.

## CI and publication policy

- **Pull requests and pushes to `main`:** run tests and build affected images,
  but never push to GHCR.
- **Manual dispatch:** may publish selected immutable SHA tags.
- **Rolling tags:** require the additional `publish_latest=true` input after a
  real smoke on the same source revision, plus `gpu_smoke_commit` equal to the
  exact dispatched commit SHA.
- **Local release script:** `--push` requires `--tag <family>-<git-sha>`;
  rolling tags additionally require `--publish-latest --confirm-gpu-smoke`.
- Test-only and compose-only changes run unit tests without multi-GB image builds.
- Branch protection should require `unit-tests`; image jobs may be skipped when
  no Docker image input changed.

## Dependency updates

- Dependabot proposes GitHub Actions, Docker base and all exact requirement updates.
- `refresh-unsloth-pins.yml` resolves the official notebook-compatible CUDA
  stack weekly and opens a PR.
- Build tooling and shared worker I/O packages are exact-pinned in
  `requirements-build.txt` and `requirements-worker-io.txt`.
- Update PRs never publish. CUDA/ROCm/multi-GPU image builds are validation only.
- A real matching GPU mini-train is required before merge/promotion.
- Published images include an SBOM, third-party notices and a deterministic
  Python distribution/license inventory under `/licenses`.

The repository does not auto-merge dependency PRs. This removes hand-maintained
version work without promoting a new GPU stack before hardware qualification.

## Provider image updater

Linux providers can install the optional six-hour updater:

```bash
sudo ./scripts/install-provider-updater.sh \
  --compose-file /opt/openreef-provider/docker-compose-nvidia.yml \
  --project-directory /opt/openreef-provider
```

It creates an `updating` drain marker, refuses to interrupt `training.active`,
pulls only the compose-selected rolling/digest images, checks signer and worker
HTTP readiness, and recreates the prior locally tagged images on failure. It
retains the latest three rollback generations per service. Digest-pinned
providers remain pinned until an operator changes their configured digest.

## Checkpoints and metrics

Unsloth jobs save bounded checkpoints in the provider workspace and resume the
latest checkpoint after a same-provider process restart. With a validation
split they select the best checkpoint by eval loss and support early stopping.
The final API result includes train/eval loss, epoch and global step. Checkpoints
are deleted only after the final adapter has been delivered successfully.

## Extension gates

Adapter ZIP is the only enabled export. Merged-model/GGUF exports wait for
storage pricing, model-license checks and a supported conversion smoke matrix.
Dataset recipes are disabled until LLM credentials, cost limits, moderation and
dataset-evaluation contracts exist. NeMo Curator is the preferred Apache-2.0
candidate for a separate future service; it will not be added to provider images.

## License

MIT - see [LICENSE](LICENSE). Runtime dependencies retain their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). OpenReef calls the
Apache-2.0 Unsloth core API. `unsloth_zoo` is LGPL-3.0 and the separate Studio
source tree is AGPL-3.0; OpenReef does not copy or import Studio/CLI code.

## Security

- Never commit `.env`, private keys, or tokens.
- `PROVIDER_PRIVATE_KEY` belongs in the **signer** sidecar (or provider-app env), not in this image.
- Report security issues privately to the OpenReef operators.
