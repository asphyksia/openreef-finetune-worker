"""Canonical fine-tuning hyperparameters for the OGPU worker image.

MIRROR of backend/app/services/training_config.py — keep both files identical
in logic when changing presets, clamps, or Axolotl defaults. The worker image
cannot import the FastAPI package, so this copy lives in the source tree.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SEQUENCE_LEN = 2048
MAX_TOKENS_PER_EXAMPLE = SEQUENCE_LEN

# Multipack sample packing needs enough sequences to fill ≥1 batch.
# Below this row count Axolotl's multipack sampler can emit zero batches and
# crash with IndexError on batches[-1] (seen on 3-row smoke datasets).
MIN_ROWS_FOR_SAMPLE_PACKING = 16

# AMD ROCm: packing multiplies effective activations. Enable only when the
# provider reports enough free-card headroom (consumer 16GB stays off).
MIN_VRAM_GB_AMD_SAMPLE_PACKING = 24.0
# NVIDIA: packing is default, but skip on tiny cards + large models.
MIN_VRAM_GB_CUDA_SAMPLE_PACKING = 10.0

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PRESET_PARAMS: dict[str, dict[str, Any]] = {
    "fast": {
        "num_epochs": 1,
        "learning_rate": 2e-4,
        "base_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "val_set_size": 0.0,
    },
    "balanced": {
        "num_epochs": 2,
        "learning_rate": 1e-4,
        "base_batch_size": 2,
        "lora_r": 32,
        "lora_alpha": 64,
        "val_set_size": 0.05,
    },
    "quality": {
        "num_epochs": 3,
        "learning_rate": 5e-5,
        "base_batch_size": 1,
        "lora_r": 64,
        "lora_alpha": 128,
        "val_set_size": 0.05,
    },
}


def clamp_micro_batch(
    base_batch: int,
    *,
    param_count: int = 0,
    device: str = "nvidia_cuda",
    adapter: str = "lora",
) -> int:
    batch = max(1, int(base_batch))
    adapter = (adapter or "lora").lower()
    device = (device or "nvidia_cuda").lower()

    if param_count >= 70:
        batch = 1
    elif param_count >= 13:
        batch = min(batch, 1)
    elif param_count >= 7:
        batch = min(batch, 2)

    # AMD ROCm (esp. consumer 16GB): activations + fp16 logits float-up blow VRAM
    if device == "amd_rocm":
        batch = 1

    if adapter == "qlora" and param_count >= 13:
        batch = 1

    return max(1, batch)


def should_enable_sample_packing(
    *,
    device: str,
    num_dataset_rows: int | None = None,
    micro_batch_size: int = 1,
    vram_gb: float | None = None,
    param_count: int = 0,
) -> bool:
    """Return whether Axolotl multipack sample packing is safe for this job.

    Shared rules:
    - Off when the dataset is too small to fill a packed batch (empty multipack).

    Device rules:
    - NVIDIA: on by default when rows are enough; off on very small VRAM + large models.
    - AMD ROCm: **dynamic** — on only when provider VRAM is known and ≥
      MIN_VRAM_GB_AMD_SAMPLE_PACKING (so MI300 / 48GB+ / 24GB+ pro cards can pack;
      16GB consumer stays safe). Unknown VRAM → off (conservative).
    """
    device = (device or "nvidia_cuda").lower()
    rows_ok = True
    if num_dataset_rows is not None:
        rows = max(0, int(num_dataset_rows))
        min_rows = max(MIN_ROWS_FOR_SAMPLE_PACKING, int(micro_batch_size) * 4)
        rows_ok = rows >= min_rows
    if not rows_ok:
        return False

    if device == "amd_rocm":
        if vram_gb is None:
            return False  # unknown card → keep 16GB-safe default
        return float(vram_gb) + 1e-6 >= MIN_VRAM_GB_AMD_SAMPLE_PACKING

    # nvidia_cuda / other
    if vram_gb is not None and float(vram_gb) < MIN_VRAM_GB_CUDA_SAMPLE_PACKING and param_count >= 13:
        return False
    # Unknown row count on CUDA: historical default on
    return True


def resolve_training_hyperparams(
    preset: str,
    *,
    param_count: int = 0,
    device: str = "nvidia_cuda",
    adapter: str = "lora",
    num_dataset_rows: int | None = None,
    vram_gb: float | None = None,
) -> dict[str, Any]:
    base = deepcopy(PRESET_PARAMS.get(preset) or PRESET_PARAMS["balanced"])
    micro_batch = clamp_micro_batch(
        int(base["base_batch_size"]),
        param_count=param_count,
        device=device,
        adapter=adapter,
    )
    # Tiny smoke datasets: keep batch=1 so dataloader always has full steps.
    if num_dataset_rows is not None and int(num_dataset_rows) < max(8, micro_batch * 4):
        micro_batch = 1
    gradient_accumulation_steps = max(1, 4 // micro_batch)
    needs_gc = param_count >= 7 or device == "amd_rocm"
    # Longer context on ROCm only when packing is viable (more VRAM).
    sequence_len = SEQUENCE_LEN
    if device == "amd_rocm":
        if vram_gb is not None and float(vram_gb) + 1e-6 >= MIN_VRAM_GB_AMD_SAMPLE_PACKING:
            sequence_len = SEQUENCE_LEN
        else:
            sequence_len = 1024
    sample_packing = should_enable_sample_packing(
        device=device,
        num_dataset_rows=num_dataset_rows,
        micro_batch_size=micro_batch,
        vram_gb=vram_gb,
        param_count=param_count,
    )

    return {
        "num_epochs": int(base["num_epochs"]),
        "learning_rate": float(base["learning_rate"]),
        "batch_size": micro_batch,
        "micro_batch_size": micro_batch,
        "base_batch_size": int(base["base_batch_size"]),
        "lora_r": int(base["lora_r"]),
        "lora_alpha": int(base["lora_alpha"]),
        "lora_dropout": 0.05,
        "sequence_len": sequence_len,
        "sample_packing": sample_packing,
        "val_set_size": float(base["val_set_size"]),
        "warmup_ratio": 0.05,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": needs_gc,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 1,
        "train_on_inputs": False,
        "param_count": int(param_count or 0),
        "preset": preset if preset in PRESET_PARAMS else "balanced",
        "adapter": adapter,
    }


def device_precision(device: str) -> dict[str, Any]:
    if device == "amd_rocm":
        return {
            "bf16": False,
            "fp16": True,
            "tf32": False,
            "attn_implementation": "sdpa",
            "optimizer": "adamw_torch",
        }
    if device == "nvidia_cuda":
        # adamw_torch is the same stable path as AMD; adamw_bnb_8bit needs
        # bitsandbytes and has caused opaque train exits on some provider hosts.
        return {
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "attn_implementation": "sdpa",
            "optimizer": "adamw_torch",
        }
    return {
        "bf16": False,
        "fp16": False,
        "tf32": False,
        "attn_implementation": "sdpa",
        "optimizer": "adamw_torch",
    }


def build_axolotl_config_dict(
    *,
    base_model: str,
    dataset_path: str,
    device: str,
    adapter: str,
    output_dir: str,
    preset: str = "balanced",
    param_count: int = 0,
    dataset_prepared_path: str | None = None,
    optimizer_override: str | None = None,
    num_dataset_rows: int | None = None,
    vram_gb: float | None = None,
) -> dict[str, Any]:
    adapter = (adapter or "lora").lower()
    if adapter == "qlora" and device == "amd_rocm":
        adapter = "lora"

    hp = resolve_training_hyperparams(
        preset,
        param_count=param_count,
        device=device,
        adapter=adapter,
        num_dataset_rows=num_dataset_rows,
        vram_gb=vram_gb,
    )
    precision = device_precision(device)
    optimizer = optimizer_override or precision["optimizer"]

    # When packing is off, pad_to_sequence_len still works but wastes compute on
    # tiny jobs; keep it True for stable shapes on larger non-packed runs.
    config: dict[str, Any] = {
        "base_model": base_model,
        "base_model_config": base_model,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "datasets": [
            {
                "path": dataset_path,
                "type": {
                    "field_instruction": "instruction",
                    "field_output": "output",
                    "format": "{instruction}",
                    "no_input_format": "{instruction}",
                },
            }
        ],
        "dataset_prepared_path": dataset_prepared_path or "/workspace/prepared",
        "val_set_size": hp["val_set_size"],
        "output_dir": output_dir,
        "sequence_len": hp["sequence_len"],
        "sample_packing": hp["sample_packing"],
        "eval_sample_packing": False,
        "pad_to_sequence_len": bool(hp["sample_packing"]),
        "adapter": adapter,
        "lora_r": hp["lora_r"],
        "lora_alpha": hp["lora_alpha"],
        "lora_dropout": hp["lora_dropout"],
        "lora_target_linear": True,
        "lora_target_modules": list(LORA_TARGET_MODULES),
        # Axolotl 0.17 auto-enables lora_*_kernel=true (Triton SwiGLU/LoRA) when
        # unset. Those kernels need a C compiler (triton CudaUtils) on first use.
        # Our CUDA image is nvidia/cuda *-runtime* (no gcc) by design — leave
        # kernels OFF so training uses eager PEFT/torch. Slightly less throughput;
        # far more reliable on marketplace provider images. See docs.axolotl.ai
        # lora_optims + OpenReef smoke 841e5c59 (Makre, C compiler RuntimeError).
        "lora_mlp_kernel": False,
        "lora_qkv_kernel": False,
        "lora_o_kernel": False,
        "lora_embedding_kernel": False,
        "gradient_accumulation_steps": hp["gradient_accumulation_steps"],
        "micro_batch_size": hp["micro_batch_size"],
        "num_epochs": hp["num_epochs"],
        "optimizer": optimizer,
        "lr_scheduler": "cosine",
        "learning_rate": hp["learning_rate"],
        "train_on_inputs": hp["train_on_inputs"],
        "max_grad_norm": hp["max_grad_norm"],
        "weight_decay": hp["weight_decay"],
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "tf32": precision["tf32"],
        "attn_implementation": precision["attn_implementation"],
        "logging_steps": hp["logging_steps"],
        "save_steps": hp["save_steps"],
        "save_total_limit": hp["save_total_limit"],
        "warmup_ratio": hp["warmup_ratio"],
        "gradient_checkpointing": hp["gradient_checkpointing"],
    }
    # Axolotl requires load_in_4bit for QLoRA (bitsandbytes); without it train
    # aborts at config validation before any GPU work.
    if adapter == "qlora":
        config["load_in_4bit"] = True
        config["load_in_8bit"] = False
    if hp["gradient_checkpointing"]:
        config["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
    return config
