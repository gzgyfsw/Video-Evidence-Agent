#!/usr/bin/env python3
"""Verify that the curated four-stage source map resolves inside the repo."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "stages" / "source_manifest.json"
EXPECTED_IDS = [
    "01_full_video_model",
    "02_tool_workflow",
    "03_autonomous_agent_loop",
    "04_policy_configurable_agent",
]


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stages = payload.get("stages", [])
    ids = [stage.get("id") for stage in stages]
    errors: list[str] = []

    if ids != EXPECTED_IDS:
        errors.append(f"stage order mismatch: {ids!r}")

    for stage in stages:
        stage_id = stage.get("id", "<missing-id>")
        stage_dir = ROOT / "stages" / stage_id
        if not (stage_dir / "README.md").is_file():
            errors.append(f"{stage_id}: missing README.md")
        for group in ("entrypoints", "shared", "policy", "training"):
            for relative in stage.get(group, []):
                path = ROOT / relative
                if path.is_file():
                    print(f"PRESENT\t{stage_id}\t{group}\t{relative}")
                else:
                    errors.append(f"{stage_id}: missing {group} path {relative}")

    if errors:
        for error in errors:
            print(f"ERROR\t{error}")
        return 1
    print("OK\tfour-stage layout is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
