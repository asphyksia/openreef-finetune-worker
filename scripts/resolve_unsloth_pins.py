#!/usr/bin/env python3
"""Resolve OpenReef CUDA finetune pins from Unsloth notebook + PyPI metadata.

Policy (Qoder Level B adjustments):
  1. trl / transformers / datasets — PRIMARY source is the official Unsloth
     notebook install cell (unslothai/notebooks). Metadata ranges alone can
     allow combos Unsloth has not smoke-tested (e.g. trl 0.24 vs notebook 0.22.2).
  2. peft / accelerate / bitsandbytes / unsloth_zoo — max version on PyPI that
     satisfies unsloth's requires_dist SpecifierSet (notebook installs "latest").
  3. unsloth itself — latest (or --unsloth-version) CalVer pin; full lock written.
  4. torch / xformers — platform pins: torch + CUDA channel fixed; xformers
     matched to torch line via notebook map / Unsloth cu126-torch* extras.
  5. Output is a complete lock (pins-cuda.env). Nothing should float at build.

Usage:
  python3 resolve_unsloth_pins.py              # print to stdout
  python3 resolve_unsloth_pins.py --write      # write ../pins-cuda.env
  python3 resolve_unsloth_pins.py --check      # exit 1 if file would change
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

FINETUNE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = FINETUNE_DIR / "pins-cuda.env"

# Platform line for OpenReef NVIDIA image (change rarely; not every Unsloth bump).
DEFAULT_TORCH = "2.9.1"
DEFAULT_CUDA_CHANNEL = "cu126"
DEFAULT_OGPU = "0.2.1"

# Fallback when notebook scrape fails (last known official combo).
NOTEBOOK_FALLBACK = {
    "transformers": "4.56.2",
    "trl": "0.22.2",
    "datasets": "4.3.0",
}

# Notebook + Unsloth extras map torch major.minor → xformers pin.
# 2.9.1 uses Unsloth extra cu126onlytorch291 (post2); notebook maps 2.9 → post1.
XFORMERS_BY_TORCH = {
    "2.8": "0.0.32.post2",
    "2.9": "0.0.33.post1",
    "2.9.1": "0.0.33.post2",
    "2.10": "0.0.34",
}

NOTEBOOK_CANDIDATES = [
    "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb",
    "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Qwen3_(14B)-Conversational.ipynb",
    "https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb",
]

PYPI_JSON = "https://pypi.org/pypi/{pkg}/json"
OGPU_COMPAT_SPEC = SpecifierSet(">=0.2.3,<0.3")

# Packages resolved to "max in SpecifierSet" (notebook leaves these floating).
RANGE_PKGS = ("peft", "accelerate", "bitsandbytes", "unsloth_zoo")

# Packages taken from notebook install cells when present.
NOTEBOOK_PKGS = ("transformers", "trl", "datasets")


def _http_json(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "OpenReef-pin-resolver/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _http_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "OpenReef-pin-resolver/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def pypi_info(pkg: str) -> dict[str, Any]:
    return _http_json(PYPI_JSON.format(pkg=pkg))


def pypi_versions(pkg: str) -> list[str]:
    data = pypi_info(pkg)
    vers: list[Version] = []
    for v in data.get("releases") or {}:
        try:
            ver = Version(v)
        except InvalidVersion:
            continue
        if ver.is_prerelease or ver.is_devrelease:
            continue
        vers.append(ver)
    vers.sort()
    return [str(v) for v in vers]


def base_requirement_specs(pkg: str, dep_name: str) -> SpecifierSet:
    """Union of unconditional requires_dist specs for dep_name from pkg metadata."""
    data = pypi_info(pkg)
    specs = SpecifierSet()
    for raw in data.get("info", {}).get("requires_dist") or []:
        if not raw:
            continue
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.name.lower() != dep_name.lower():
            continue
        # Skip marker-only extras that are clearly non-linux (darwin arm etc.) when
        # marker is present and evaluates false on linux x86_64 — best-effort.
        if req.marker is not None:
            try:
                if not req.marker.evaluate(
                    {
                        "sys_platform": "linux",
                        "platform_machine": "x86_64",
                        "python_version": "3.10",
                        "extra": "",
                    }
                ):
                    continue
            except Exception:
                pass
        specs &= req.specifier
    return specs


def max_in_spec(pkg: str, spec: SpecifierSet) -> str:
    if not spec:
        # No constraint from unsloth — take latest.
        vers = pypi_versions(pkg)
        if not vers:
            raise RuntimeError(f"No versions for {pkg}")
        return vers[-1]
    candidates = [v for v in pypi_versions(pkg) if Version(v) in spec]
    if not candidates:
        raise RuntimeError(f"No {pkg} version satisfies {spec}")
    return candidates[-1]


def scrape_notebook_pins() -> dict[str, str]:
    """Extract transformers/trl/datasets==X from Unsloth notebook install cells."""
    found: dict[str, str] = {}
    # Match both raw and JSON-escaped notebook cells ("datasets==4.3.0" / \"datasets==4.3.0\").
    pat = re.compile(
        r"""(?:pip\s+install|!pip\s+install)[^\n"']*?\b(transformers|trl|datasets)==([0-9][0-9a-zA-Z.\-]+)""",
        re.IGNORECASE,
    )
    pat_q = re.compile(
        r"""(?:\\?["'])(transformers|trl|datasets)==([0-9][0-9a-zA-Z.\-]+)(?:\\?["'])""",
        re.IGNORECASE,
    )
    for url in NOTEBOOK_CANDIDATES:
        try:
            raw = _http_text(url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"notebook scrape skip {url}: {exc}", file=sys.stderr)
            continue
        # ipynb is JSON; still search raw text for speed
        for m in pat.finditer(raw):
            found[m.group(1).lower()] = m.group(2)
        for m in pat_q.finditer(raw):
            found[m.group(1).lower()] = m.group(2)
        if all(k in found for k in NOTEBOOK_PKGS):
            print(f"notebook pins from {url}", file=sys.stderr)
            return {k: found[k] for k in NOTEBOOK_PKGS}
    if found:
        print(f"notebook pins partial: {found}", file=sys.stderr)
        out = dict(NOTEBOOK_FALLBACK)
        out.update(found)
        return out
    print("notebook scrape failed; using NOTEBOOK_FALLBACK", file=sys.stderr)
    return dict(NOTEBOOK_FALLBACK)


def xformers_for_torch(torch_version: str) -> str:
    if torch_version in XFORMERS_BY_TORCH:
        return XFORMERS_BY_TORCH[torch_version]
    major_minor = ".".join(torch_version.split(".")[:2])
    if major_minor in XFORMERS_BY_TORCH:
        return XFORMERS_BY_TORCH[major_minor]
    raise RuntimeError(
        f"No xformers map for torch {torch_version}; extend XFORMERS_BY_TORCH"
    )


def assert_in_spec(pkg: str, version: str, spec: SpecifierSet) -> None:
    if not spec:
        return
    if Version(version) not in spec:
        raise RuntimeError(
            f"{pkg}=={version} outside unsloth requires_dist {spec} — refuse pin"
        )


def resolve(
    *,
    unsloth_version: str | None = None,
    torch_version: str = DEFAULT_TORCH,
    cuda_channel: str = DEFAULT_CUDA_CHANNEL,
    ogpu_version: str | None = None,
) -> dict[str, str]:
    u_info = pypi_info("unsloth")
    unsloth_v = unsloth_version or u_info["info"]["version"]

    # When pinning a non-latest unsloth, still validate against that release's metadata.
    if unsloth_v != u_info["info"]["version"]:
        u_info = _http_json(f"https://pypi.org/pypi/unsloth/{unsloth_v}/json")

    nb = scrape_notebook_pins()

    pins: dict[str, str] = {
        "UNSLOTH_VERSION": unsloth_v,
        "TORCH_VERSION": torch_version,
        "TORCH_CUDA_CHANNEL": cuda_channel,
        "XFORMERS_VERSION": xformers_for_torch(torch_version),
        "XFORMERS_INDEX_URL": f"https://download.pytorch.org/whl/{cuda_channel}",
        "TORCH_INDEX_URL": f"https://download.pytorch.org/whl/{cuda_channel}",
        "OGPU_VERSION": ogpu_version or max_in_spec("ogpu", OGPU_COMPAT_SPEC),
        "TRANSFORMERS_VERSION": nb["transformers"],
        "TRL_VERSION": nb["trl"],
        "DATASETS_VERSION": nb["datasets"],
    }

    # Validate notebook pins against unsloth metadata ranges when present.
    for name, key in (
        ("transformers", "TRANSFORMERS_VERSION"),
        ("trl", "TRL_VERSION"),
        ("datasets", "DATASETS_VERSION"),
    ):
        spec = base_requirement_specs("unsloth", name)
        assert_in_spec(name, pins[key], spec)

    for name in RANGE_PKGS:
        spec = base_requirement_specs("unsloth", name)
        if name == "unsloth_zoo" and not spec:
            spec = SpecifierSet(">=2026.8.6")
        ver = max_in_spec(name, spec)
        pins[f"{name.upper()}_VERSION"] = ver

    # Stable key order for diffs
    order = [
        "UNSLOTH_VERSION",
        "UNSLOTH_ZOO_VERSION",
        "TORCH_VERSION",
        "TORCH_CUDA_CHANNEL",
        "TORCH_INDEX_URL",
        "XFORMERS_VERSION",
        "XFORMERS_INDEX_URL",
        "TRANSFORMERS_VERSION",
        "TRL_VERSION",
        "DATASETS_VERSION",
        "PEFT_VERSION",
        "ACCELERATE_VERSION",
        "BITSANDBYTES_VERSION",
        "OGPU_VERSION",
    ]
    return {k: pins[k] for k in order if k in pins}


def render_env(pins: dict[str, str]) -> str:
    lines = [
        "# OpenReef CUDA finetune lock — GENERATED, do not hand-edit casually.",
        "# Regenerate: python3 scripts/resolve_unsloth_pins.py --write",
        "#",
        "# Policy:",
        "#   - trl/transformers/datasets: Unsloth official notebook install pins",
        "#   - peft/accelerate/bitsandbytes/unsloth_zoo: max version in unsloth SpecifierSet",
        "#   - torch/xformers: platform pins (cu126 wheel index, not generic PyPI)",
        "#",
        "# Merge gate (Level B):",
        "#   1) CI: docker build CUDA + SFTConfig smoke (no push to :cuda-latest)",
        "#   2) Human: 1-step mini-train on house GPU or Vast before merge to main",
        "# Publication remains a separate manual workflow_dispatch after the GPU gate.",
        "#",
        f"# resolved_for_unsloth={pins.get('UNSLOTH_VERSION', '?')}",
        "",
    ]
    for k, v in pins.items():
        lines.append(f"{k}={v}")
    lines.append("")
    return "\n".join(lines)


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help=f"Write {DEFAULT_OUT}")
    ap.add_argument("--check", action="store_true", help="Exit 1 if pins-cuda.env is stale")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--unsloth-version", default=None, help="Pin specific unsloth (default: latest)")
    ap.add_argument("--torch-version", default=DEFAULT_TORCH)
    ap.add_argument("--cuda-channel", default=DEFAULT_CUDA_CHANNEL)
    args = ap.parse_args(argv)

    pins = resolve(
        unsloth_version=args.unsloth_version,
        torch_version=args.torch_version,
        cuda_channel=args.cuda_channel,
    )
    text = render_env(pins)

    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        if current.strip() != text.strip():
            print("pins-cuda.env is stale; run resolve_unsloth_pins.py --write", file=sys.stderr)
            return 1
        print("pins-cuda.env is up to date")
        return 0

    if args.write:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
