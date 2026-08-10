"""NVIDIA Unsloth SFT engine for OpenReef finetune workers.

Primary path on CUDA images and (experiment) AMD/ROCm. ROCm rules: LoRA only
(QLoRA stays off AMD: bitsandbytes 4-bit decode is not safe there yet) and
adamw_torch instead of the bnb-backed adamw_8bit. Exports a standard PEFT adapter directory
(``adapter_model.safetensors`` + ``adapter_config.json``) so packaging is shared.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("openreef.unsloth_train")

# Default LoRA targets for Llama/Mistral/Qwen-style dense Transformers.
_DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Liquid LFM2.x hybrid modules (from Unsloth LFM2.5 guide).
_LFM_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "in_proj",
    "w1",
    "w2",
    "w3",
]


def unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401

        return True
    except Exception:
        return False


def lora_target_modules_for_model(base_model: str) -> list[str]:
    name = (base_model or "").lower()
    if "lfm" in name or "liquid" in name:
        return list(_LFM_LORA_TARGETS)
    return list(_DEFAULT_LORA_TARGETS)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_num}: expected JSON object")
            rows.append(obj)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def _example_to_text(example: dict[str, Any], tokenizer: Any) -> str:
    """Build a single training string aligned with chat template when possible."""
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    instruction = str(example.get("instruction") or "").strip()
    input_text = str(example.get("input") or "").strip()
    output = str(example.get("output") or "").strip()
    if input_text and instruction:
        user = f"{instruction}\n\n{input_text}"
    else:
        user = instruction or input_text
    if not output:
        return ""

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user or "Continue."},
                    {"role": "assistant", "content": output},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    if user:
        return f"### Instruction:\n{user}\n\n### Response:\n{output}"
    return output


def run_unsloth_sft(
    *,
    base_model: str,
    dataset_path: Path,
    output_dir: Path,
    preset: str = "balanced",
    adapter: str = "lora",
    device: str = "nvidia_cuda",
    param_count: int = 0,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    max_seq_length: int = 2048,
    num_dataset_rows: int | None = None,
    log_path: Path | None = None,
    phase: Callable[..., None] | None = None,
) -> Path:
    """Train with Unsloth and write PEFT adapter files under ``output_dir``.

    Returns ``output_dir`` on success. Raises on failure.
    """
    from training_config import resolve_training_hyperparams

    def _phase(name: str, **fields: object) -> None:
        if phase is not None:
            phase(name, **fields)
        else:
            logger.info("PHASE=%s %s", name, fields)

    if not unsloth_available():
        raise RuntimeError("unsloth is not installed in this image")

    # Late imports so Axolotl-only images never import unsloth at module load.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    adapter = (adapter or "lora").lower()
    device = (device or "nvidia_cuda").strip().lower()
    if adapter == "qlora" and device == "amd_rocm":
        raise RuntimeError(
            "QLoRA on AMD/ROCm is disabled: bitsandbytes 4-bit decode is not "
            "safe on AMD yet (use adapter=lora on ROCm)"
        )
    load_in_4bit = adapter == "qlora"
    hp = resolve_training_hyperparams(
        preset,
        param_count=param_count,
        device=device,
        adapter=adapter,
        num_dataset_rows=num_dataset_rows,
    )
    r = int(lora_r if lora_r is not None else hp["lora_r"])
    alpha = int(lora_alpha if lora_alpha is not None else hp["lora_alpha"])
    micro = int(hp["micro_batch_size"])
    grad_accum = int(hp["gradient_accumulation_steps"])
    epochs = int(hp["num_epochs"])
    lr = float(hp["learning_rate"])
    max_seq = int(max_seq_length or hp.get("sequence_len") or 2048)
    targets = lora_target_modules_for_model(base_model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _phase(
        "unsloth_load_start",
        base_model=base_model[:80],
        load_in_4bit=load_in_4bit,
        max_seq=max_seq,
    )
    # ROCm RDNA4 (gfx1201): bf16 GEMMs page-fault in rocBLAS/Tensile
    # (rocm-libraries#7992, open upstream). fp16 is the validated path on AMD;
    # verified full 135-step train on RX 9060 XT (2026-08-10).
    import torch as _torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq,
        dtype=_torch.float16 if device == "amd_rocm" else None,
        load_in_4bit=load_in_4bit,
    )
    # Unsloth kernel path is optimized for lora_dropout=0 (see Unsloth LoRA guide).
    dropout = 0.0

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=targets,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    _phase("unsloth_peft_ok", r=r, alpha=alpha, targets=len(targets))

    rows = _load_jsonl_rows(Path(dataset_path))
    texts: list[str] = []
    for row in rows:
        text = _example_to_text(row, tokenizer)
        if text and text.strip():
            texts.append(text)
    if not texts:
        raise RuntimeError("No train strings after formatting dataset")
    train_ds = Dataset.from_dict({"text": texts})
    _phase("unsloth_dataset_ok", rows=len(texts), max_seq=max_seq)

    # val split only when enough rows (mirror Axolotl val_set_size gates)
    eval_ds = None
    val_frac = float(hp.get("val_set_size") or 0)
    if val_frac > 0 and len(texts) >= 32:
        split = train_ds.train_test_split(test_size=val_frac, seed=3407)
        train_ds = split["train"]
        eval_ds = split["test"]

    # bf16 disabled on AMD: see rocm-libraries#7992 (RDNA4 Tensile page fault).
    bf16 = is_bfloat16_supported() and device != "amd_rocm"
    # Logging file: redirect print/trainer via default logging
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer SFTConfig (TRL ≥0.22): dataset_* / packing live on the config object.
    sft_args = SFTConfig(
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=float(hp.get("warmup_ratio") or 0.05),
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=max(1, int(hp.get("logging_steps") or 10)),
        optim="adamw_8bit" if device == "nvidia_cuda" else "adamw_torch",
        weight_decay=float(hp.get("weight_decay") or 0.0),
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=str(output_dir / "trainer_state"),
        report_to="none",
        save_strategy="no",
        max_grad_norm=float(hp.get("max_grad_norm") or 1.0),
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_seq_length=max_seq,
        packing=False,
        dataset_num_proc=1,
    )

    _phase(
        "train_start",
        engine="unsloth",
        epochs=epochs,
        batch=micro,
        grad_accum=grad_accum,
        lr=lr,
        rows=len(train_ds),
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": sft_args,
    }
    # TRL 0.22 still accepts tokenizer=; newer uses processing_class=.
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)

    # Optional tee of trainer logs
    if log_path is not None:
        try:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            logging.getLogger().addHandler(fh)
        except Exception:
            pass

    trainer.train()

    # Save PEFT adapter + tokenizer next to what Axolotl would produce.
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _phase("unsloth_save_ok", output_dir=str(output_dir))
    return output_dir
