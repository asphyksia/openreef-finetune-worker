#!/usr/bin/env python3
"""Write deterministic metadata for Python distributions installed in an image."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path


def distribution_record(dist: metadata.Distribution) -> dict[str, object]:
    meta = dist.metadata
    license_files = sorted(
        str(path)
        for path in (dist.files or [])
        if any(part.lower().startswith(("license", "copying", "notice")) for part in path.parts)
    )
    return {
        "name": meta.get("Name") or dist.name,
        "version": dist.version,
        "license": meta.get("License-Expression") or meta.get("License") or None,
        "project_url": meta.get("Home-page") or None,
        "license_files": license_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/licenses/python-distributions.json")
    args = parser.parse_args()

    records = sorted(
        (distribution_record(dist) for dist in metadata.distributions()),
        key=lambda item: (str(item["name"]).lower(), str(item["version"])),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
