"""NVIDIA Unsloth SFT engine for OpenReef finetune workers.

Primary path on CUDA images and (experiment) AMD/ROCm. ROCm rules: LoRA only
(QLoRA stays off AMD: bitsandbytes 4-bit decode is not safe there yet) and
adamw_torch instead of the bnb-backed adamw_8bit. Exports a standard PEFT adapter directory
(``adapter_model.safetensors`` + ``adapter_config.json``) so packaging is shared.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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


def _hash_messages(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _example_to_text(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    sft_profile: str = "chat",
) -> tuple[str, str]:
    """Build training string via SFT contract v1. Returns (text, renderer_id)."""
    from sft_format import normalize_messages, render_training_text

    messages = normalize_messages(example)
    if not messages:
        return "", "empty"
    return render_training_text(messages, tokenizer, sft_profile=sft_profile)


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
    sft_profile: str = "chat",
    serve_smoke: bool = True,
    custom_config: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    holdout_hashes: list[str] | None = None,
) -> Path:
    """Train with Unsloth and write PEFT adapter files under ``output_dir``.

    Returns ``output_dir`` on success. Raises on failure.
    """
    from training_config import (
        clamp_schedule_steps,
        estimate_train_steps,
        resolve_training_hyperparams,
    )

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
        custom_config=custom_config,
    )
    r = int(lora_r if lora_r is not None else hp["lora_r"])
    alpha = int(lora_alpha if lora_alpha is not None else hp["lora_alpha"])
    micro = int(hp["micro_batch_size"])
    grad_accum = int(hp["gradient_accumulation_steps"])
    epochs = int(hp["num_epochs"])
    lr = float(hp["learning_rate"])
    max_seq = int(max_seq_length or hp.get("sequence_len") or 2048)
    targets = lora_target_modules_for_model(base_model)
    dropout = float(hp.get("lora_dropout") or 0.0)
    seed = int(hp.get("seed") or 3407)

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
        trust_remote_code=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=targets,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        use_gradient_checkpointing=(
            "unsloth" if hp.get("gradient_checkpointing", True) else False
        ),
        random_state=seed,
        use_rslora=bool(hp.get("use_rslora", False)),
    )
    _phase("unsloth_peft_ok", r=r, alpha=alpha, targets=len(targets))

    profile = (sft_profile or "chat").strip().lower()
    rows = _load_jsonl_rows(Path(dataset_path))
    holdout = {h for h in (holdout_hashes or []) if h}
    train_texts: list[str] = []
    eval_texts: list[str] = []
    eval_rows: list[dict[str, Any]] = []
    renderer_ids: list[str] = []
    for row in rows:
        text, rid = _example_to_text(row, tokenizer, sft_profile=profile)
        if not (text and text.strip()):
            continue
        renderer_ids.append(rid)
        digest = _hash_messages(row.get("messages") or [])
        if holdout and digest in holdout:
            eval_texts.append(text)
            eval_rows.append(row)
        else:
            train_texts.append(text)
    if not train_texts:
        raise RuntimeError("No train strings after formatting dataset")
    # Dominant renderer (should be uniform; mixed → experimental inconsistency)
    from collections import Counter

    renderer = Counter(renderer_ids).most_common(1)[0][0]
    if profile == "chat" and renderer != "chat_template":
        logger.warning(
            "sft_profile=chat but renderer=%s (missing/broken chat_template?) — "
            "smoke/eval must use the same renderer",
            renderer,
        )

    train_ds = Dataset.from_dict({"text": train_texts})
    _phase(
        "unsloth_dataset_ok",
        rows=len(train_texts),
        max_seq=max_seq,
        sft_profile=profile,
        renderer=renderer,
        has_chat_template=bool(getattr(tokenizer, "chat_template", None)),
        holdout_rows=len(eval_texts),
    )

    # Lab-pinned holdout wins. Otherwise val split when the preset asks for it.
    eval_ds = None
    if eval_texts:
        eval_ds = Dataset.from_dict({"text": eval_texts})
    else:
        val_frac = float(hp.get("val_set_size") or 0)
        if val_frac > 0 and len(train_texts) >= 32:
            split = train_ds.train_test_split(test_size=val_frac, seed=seed)
            train_ds = split["train"]
            eval_ds = split["test"]

    # bf16 disabled on AMD: see rocm-libraries#7992 (RDNA4 Tensile page fault).
    bf16 = is_bfloat16_supported() and device != "amd_rocm"
    # Logging file: redirect print/trainer via default logging
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    trainer_state_dir = Path(checkpoint_dir or (output_dir / "trainer_state"))
    trainer_state_dir.mkdir(parents=True, exist_ok=True)
    est_steps = estimate_train_steps(
        num_dataset_rows=len(train_ds),
        num_epochs=epochs,
        micro_batch_size=micro,
        gradient_accumulation_steps=grad_accum,
        val_set_size=0.0,
    )
    save_steps = clamp_schedule_steps(int(hp.get("save_steps") or 100), est_steps)
    logging_steps = clamp_schedule_steps(int(hp.get("logging_steps") or 10), est_steps)
    has_eval = eval_ds is not None

    # Prefer SFTConfig (TRL >=0.22): dataset_* / packing live on the config object.
    sft_kwargs: dict[str, Any] = dict(
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=float(hp.get("warmup_ratio") or 0.05),
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=logging_steps,
        optim=(
            str(hp.get("optimizer") or "adamw_8bit")
            if device == "nvidia_cuda"
            else "adamw_torch"
        ),
        weight_decay=float(hp.get("weight_decay") or 0.0),
        lr_scheduler_type=str(hp.get("lr_scheduler_type") or "cosine"),
        seed=seed,
        output_dir=str(trainer_state_dir),
        report_to="none",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=max(1, int(hp.get("save_total_limit") or 1)),
        max_grad_norm=float(hp.get("max_grad_norm") or 1.0),
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_seq_length=max_seq,
        packing=False,
        dataset_num_proc=1,
    )
    if has_eval:
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    sft_args = SFTConfig(**sft_kwargs)

    _phase(
        "train_start",
        engine="unsloth",
        epochs=epochs,
        batch=micro,
        grad_accum=grad_accum,
        lr=lr,
        rows=len(train_ds),
    )

    from transformers import TrainerCallback

    class _OpenReefProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            values = logs or {}
            fields: dict[str, Any] = {
                "step": int(getattr(state, "global_step", 0) or 0),
                "max_steps": int(getattr(state, "max_steps", 0) or 0),
            }
            for key in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                value = values.get(key)
                if isinstance(value, (int, float)):
                    fields[key] = float(value)
            _phase("train_metrics", **fields)

    callbacks: list[Any] = [_OpenReefProgressCallback()]
    patience = int(hp.get("early_stopping_patience") or 0)
    if has_eval and patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": sft_args,
        "callbacks": callbacks,
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

    checkpoints: list[tuple[int, Path]] = []
    for candidate in trainer_state_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
        if match and candidate.is_dir():
            checkpoints.append((int(match.group(1)), candidate))
    resume_checkpoint = max(checkpoints, default=(0, None))[1]
    if resume_checkpoint is not None:
        _phase("checkpoint_resume", path=str(resume_checkpoint))
    train_out = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    train_loss = None
    try:
        metrics = getattr(train_out, "metrics", None) or {}
        train_loss = metrics.get("train_loss")
    except Exception:
        train_loss = None

    trainer_state = getattr(trainer, "state", None)
    best_checkpoint = getattr(trainer_state, "best_model_checkpoint", None)
    best_metric = getattr(trainer_state, "best_metric", None)
    log_history = list(getattr(trainer_state, "log_history", None) or [])
    training_metrics = {
        "train_loss": train_loss,
        "eval_loss": best_metric if has_eval else None,
        "best_checkpoint": Path(best_checkpoint).name if best_checkpoint else None,
        "resumed_from_checkpoint": resume_checkpoint.name if resume_checkpoint else None,
        "global_step": int(getattr(trainer_state, "global_step", 0) or 0),
        "epoch": getattr(trainer_state, "epoch", None),
        "log_history": log_history[-100:],
    }
    (output_dir / "openreef_training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2, default=str),
        encoding="utf-8",
    )

    # Save PEFT adapter + tokenizer next to what Axolotl would produce.
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _phase("unsloth_save_ok", output_dir=str(output_dir))

    smoke_report: dict[str, Any] = {"enabled": bool(serve_smoke), "ok": True, "items": []}
    if serve_smoke:
        smoke_report = _run_serve_smoke(
            model=model,
            tokenizer=tokenizer,
            rows=eval_rows or rows,
            renderer=renderer,
            phase=_phase,
        )
        if not smoke_report.get("ok"):
            raise RuntimeError(
                "SFT serve_smoke failed after train: "
                + "; ".join(
                    f"{it.get('reason')}" for it in smoke_report.get("items") or [] if not it.get("ok")
                )
            )

    from sft_format import write_train_manifest

    write_train_manifest(
        output_dir / "openreef_train_manifest.json",
        {
            "schema": "openreef.sft_manifest.v1",
            "base_model": base_model,
            "sft_profile": profile,
            "prompt_renderer": renderer,
            "engine": "unsloth",
            "device": device,
            "adapter": adapter,
            "preset": preset,
            "lora_r": r,
            "lora_alpha": alpha,
            "max_seq_length": max_seq,
            "num_train_rows": len(texts),
            "train_loss": train_loss,
            "eval_loss": best_metric if has_eval else None,
            "best_checkpoint": Path(best_checkpoint).name if best_checkpoint else None,
            "resumed_from_checkpoint": resume_checkpoint.name if resume_checkpoint else None,
            "dtype": "float16" if device == "amd_rocm" or not bf16 else "bfloat16",
            "smoke": smoke_report,
        },
    )
    _phase("unsloth_manifest_ok", renderer=renderer, smoke_ok=smoke_report.get("ok"))
    return output_dir


def _run_serve_smoke(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    renderer: str,
    phase: Callable[..., None],
) -> dict[str, Any]:
    """Generate on a few train prompts with the same renderer; reject garbage."""
    import torch
    from sft_format import is_garbage_generation, pick_smoke_prompts, render_inference_prompt

    pairs = pick_smoke_prompts(rows, n=3)
    report: dict[str, Any] = {"enabled": True, "ok": True, "items": [], "renderer": renderer}
    if not pairs:
        report["ok"] = False
        report["items"].append({"ok": False, "reason": "no_smoke_prompts"})
        phase("serve_smoke_failed", reason="no_smoke_prompts")
        return report

    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    phase("serve_smoke_start", n=len(pairs), renderer=renderer)
    for user, gold in pairs:
        prompt = render_inference_prompt(user, tokenizer, renderer=renderer)
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None)
                    or getattr(tokenizer, "pad_token_id", None),
                )
            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            ).strip()
        except Exception as exc:
            report["ok"] = False
            report["items"].append(
                {"ok": False, "reason": f"generate_error:{type(exc).__name__}", "user": user[:80]}
            )
            continue
        bad, reason = is_garbage_generation(gen)
        item = {
            "ok": not bad,
            "reason": reason,
            "user": user[:120],
            "gold_preview": gold[:80],
            "gen_preview": gen[:200],
        }
        if bad:
            report["ok"] = False
        report["items"].append(item)

    if report["ok"]:
        phase("serve_smoke_ok", n=len(report["items"]))
    else:
        phase("serve_smoke_failed", n=len(report["items"]))
    return report
