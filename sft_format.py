"""SFT contract v1 — dataset normalize + render + serve-smoke heuristics.

See monorepo docs/sft-contract-v1.md. Internal SoT is always ``messages``;
profile chooses the renderer (chat_template vs fixed completion template).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Fixed completion renderer (OpenReef) — used when profile=completion or no chat_template.
COMPLETION_INSTRUCTION = "### Instruction:\n"
COMPLETION_RESPONSE = "\n\n### Response:\n"


def normalize_messages(obj: dict[str, Any]) -> list[dict[str, str]] | None:
    """Return role/content messages or None if row cannot be normalized."""
    raw = obj.get("messages")
    if isinstance(raw, list) and raw:
        out: list[dict[str, str]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip().lower()
            content = str(m.get("content") or "").strip()
            if role in ("system", "user", "assistant", "human", "gpt") and content:
                if role == "human":
                    role = "user"
                if role == "gpt":
                    role = "assistant"
                out.append({"role": role, "content": content})
        if out:
            return out

    instruction = str(obj.get("instruction") or obj.get("prompt") or obj.get("question") or "").strip()
    input_text = str(obj.get("input") or obj.get("context") or "").strip()
    output = str(obj.get("output") or obj.get("completion") or obj.get("response") or "").strip()
    if not output:
        return None
    if input_text and instruction:
        user = f"{instruction}\n\n{input_text}"
    else:
        user = instruction or input_text or "Continue."
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def user_assistant_from_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    user_parts: list[str] = []
    assistant = ""
    for m in messages:
        if m["role"] == "user":
            user_parts.append(m["content"])
        elif m["role"] == "assistant":
            assistant = m["content"]
    return ("\n\n".join(user_parts).strip() or "Continue."), assistant


def render_training_text(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    sft_profile: str = "chat",
) -> tuple[str, str]:
    """Return (text, renderer_id). renderer_id is recorded in the train manifest."""
    profile = (sft_profile or "chat").strip().lower()
    has_tpl = bool(getattr(tokenizer, "chat_template", None))

    if profile == "completion" or (profile == "experimental" and not has_tpl):
        user, assistant = user_assistant_from_messages(messages)
        if not assistant:
            return "", "empty"
        text = f"{COMPLETION_INSTRUCTION}{user}{COMPLETION_RESPONSE}{assistant}"
        return text, "completion_v1"

    # chat (default) and experimental-with-template
    if has_tpl:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if text and str(text).strip():
                return str(text), "chat_template"
        except Exception:
            pass

    # Fallback: fixed completion so train still runs, but mark renderer clearly.
    user, assistant = user_assistant_from_messages(messages)
    if not assistant:
        return "", "empty"
    text = f"{COMPLETION_INSTRUCTION}{user}{COMPLETION_RESPONSE}{assistant}"
    return text, "completion_v1_fallback"


def render_inference_prompt(
    user_text: str,
    tokenizer: Any,
    *,
    renderer: str,
) -> str:
    """Mirror train renderer for smoke/eval."""
    user_text = (user_text or "").strip() or "Continue."
    if renderer == "chat_template" and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{COMPLETION_INSTRUCTION}{user_text}{COMPLETION_RESPONSE}"


_REP_RE = re.compile(r"(.{8,}?)\1{4,}", re.DOTALL)


def is_garbage_generation(text: str, *, min_chars: int = 8) -> tuple[bool, str]:
    """Cheap heuristics for post-train smoke (no GPU quality model)."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return True, "too_short"
    # Extreme character repetition
    if _REP_RE.search(t):
        return True, "repetition"
    # Mostly the same short token
    words = t.split()
    if len(words) >= 12 and len(set(words)) <= 2:
        return True, "token_loop"
    # Tool-call / think junk that LFM base often emits under wrong format
    bad_markers = ("<|tool_call", "</think>", "<minimax:", "STEM1-1-1-1")
    if any(m in t for m in bad_markers):
        return True, "junk_markers"
    return False, "ok"


def pick_smoke_prompts(rows: list[dict[str, Any]], *, n: int = 3) -> list[tuple[str, str]]:
    """Return up to n (user, gold_assistant) pairs from normalized rows."""
    pairs: list[tuple[str, str]] = []
    for obj in rows:
        msgs = normalize_messages(obj)
        if not msgs:
            continue
        user, assistant = user_assistant_from_messages(msgs)
        if user and assistant and len(assistant) >= 12:
            pairs.append((user, assistant))
        if len(pairs) >= n:
            break
    return pairs


def write_train_manifest(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
