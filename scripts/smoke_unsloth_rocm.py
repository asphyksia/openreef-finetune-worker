"""Local ROCm smoke for the Unsloth engine (no OGPU/claim plumbing).

Run inside the ROCm worker image with GPU access, e.g.:

    docker run --rm \
      --device=/dev/kfd --device=/dev/dri \
      --security-opt seccomp=unconfined \
      --group-add video --group-add 991 \
      --ipc=host --shm-size=16g \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
      -e OPENREEF_PROVIDER_ENV=amd_rocm \
      openreef-finetune-worker:rocm-unsloth-exp \
      python /app/scripts/smoke_unsloth_rocm.py

Writes a tiny chat dataset under /tmp and trains 1 epoch of LoRA on
Llama-3.2-3B-Instruct (already in the house HF cache), then asserts the
PEFT adapter files exist.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/app")

_ROWS = [
    {"messages": [
        {"role": "user", "content": "What does OpenReef do?"},
        {"role": "assistant", "content": "OpenReef fine-tunes open LLMs on decentralized GPUs."},
    ]},
    {"messages": [
        {"role": "user", "content": "Which adapters does it export?"},
        {"role": "assistant", "content": "It exports PEFT LoRA adapters as a zip."},
    ]},
    {"messages": [
        {"role": "user", "content": "Say OK."},
        {"role": "assistant", "content": "OK"},
    ]},
] * 4  # 12 rows


def main() -> int:
    import torch

    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    if not torch.cuda.is_available():
        print("SMOKE FAIL: no ROCm device visible in container")
        return 1
    print(f"device={torch.cuda.get_device_name(0)}")

    from unsloth_train import run_unsloth_sft, unsloth_available

    if not unsloth_available():
        print("SMOKE FAIL: unsloth not importable")
        return 1

    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "tiny.jsonl"
        out = Path(td) / "out"
        ds.write_text(
            "\n".join(json.dumps(r) for r in _ROWS), encoding="utf-8"
        )
        run_unsloth_sft(
            base_model="unsloth/Llama-3.2-3B-Instruct",
            dataset_path=ds,
            output_dir=out,
            preset="fast",
            adapter="lora",
            device="amd_rocm",
            num_dataset_rows=len(_ROWS),
        )
        for fname in ("adapter_model.safetensors", "adapter_config.json"):
            if not (out / fname).exists():
                print(f"SMOKE FAIL: missing {fname}")
                return 1

    print("SMOKE OK: unsloth-rocm trained and exported a PEFT adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
