import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_format import (
    is_garbage_generation,
    normalize_messages,
    render_training_text,
)


class _Tok:
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"{m['role']}:{m['content']}")
        return "\n".join(parts) + ("\nassistant:" if add_generation_prompt else "")


def test_normalize_alpaca():
    msgs = normalize_messages({"instruction": "What is OpenReef?", "input": "", "output": "A lab."})
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "OpenReef" in msgs[0]["content"]


def test_render_chat():
    msgs = normalize_messages({"instruction": "Hi", "output": "Hello"})
    text, rid = render_training_text(msgs, _Tok(), sft_profile="chat")
    assert rid == "chat_template"
    assert "user:Hi" in text
    assert "assistant:Hello" in text


def test_render_completion():
    msgs = normalize_messages({"instruction": "Hi", "output": "Hello"})
    text, rid = render_training_text(msgs, _Tok(), sft_profile="completion")
    assert rid == "completion_v1"
    assert "### Instruction:" in text
    assert "### Response:" in text


def test_garbage():
    bad, reason = is_garbage_generation("hi")
    assert bad is True and reason == "too_short"
    bad2, r2 = is_garbage_generation("STEM1-1-1-1-1-1-1-1-1-1-1-1")
    assert bad2 and r2 == "junk_markers"
    ok, r3 = is_garbage_generation(
        "OpenReef is a fine-tuning platform on decentralized GPU compute using OGPU."
    )
    assert ok is False and r3 == "ok"
