#!/usr/bin/env python3
"""Derive structured-output GRPO prompts without overwriting source data."""

import argparse
import json
import os
from pathlib import Path

try:
    from scripts.video_safety_schema import build_category_prompt
except ModuleNotFoundError:
    from video_safety_schema import build_category_prompt


# Historical runs keep their legacy prompt by default.  The current V12 GRPO
# run explicitly requests ``answer_only`` through RL_TARGET_MODE.
TARGET_MODE = os.environ.get("RL_TARGET_MODE", "legacy_reason_confidence")
TARGET_PROMPT = build_category_prompt(TARGET_MODE)


def replace_prompt(sample: dict) -> dict:
    messages = json.loads(sample["message"])
    repo_root = Path(__file__).resolve().parents[1]
    user_message = next(message for message in messages if message.get("role") == "user")
    for part in user_message["content"]:
        if part.get("type") in {"image", "video"}:
            for key in ("image", "video"):
                if key in part:
                    part[key] = rewrite_media_path(part[key], repo_root)
    text_parts = [part for part in user_message["content"] if part.get("type") == "text"]
    if not text_parts:
        raise ValueError(f"sample {sample.get('id', '<unknown>')} has no user text part")
    text_parts[-1]["text"] = TARGET_PROMPT

    result = dict(sample)
    result["message"] = json.dumps(messages, ensure_ascii=False)
    return result


def rewrite_media_path(path: str, repo_root: Path) -> str:
    """Map dataset paths from previous machines to this checkout."""
    prefixes = (
        "/root/MM-EUREKA-qwen",
        "/home/ubuntu/MM-EUREKA-qwen",
        "/home/user/MM-EUREKA-qwen",
        "/mnt/HDD1/LLM",
        "/mnt/HDD1",
    )
    uri_prefix = ""
    raw_path = path
    if raw_path.startswith("file://"):
        uri_prefix = "file://"
        raw_path = raw_path[len(uri_prefix):]

    # Do not rewrite an absolute path that already points inside this checkout.
    # Without this guard, a current path such as
    # /mnt/HDD1/LLM/MM-EUREKA-qwen/data/... matches the broader
    # /mnt/HDD1/LLM prefix below and becomes .../MM-EUREKA-qwen/MM-EUREKA-qwen.
    current_root = str(repo_root.resolve())
    if raw_path == current_root or raw_path.startswith(current_root + "/"):
        return uri_prefix + raw_path

    for prefix in prefixes:
        if raw_path.startswith(prefix + "/"):
            raw_path = str(repo_root) + raw_path[len(prefix):]
            break
    return uri_prefix + raw_path


def prepare_file(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                sample = replace_prompt(json.loads(line))
            except Exception as exc:
                raise ValueError(f"failed to prepare {source}:{line_number}: {exc}") from exc
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = prepare_file(args.source, args.destination)
    print(f"Prepared {count} structured prompts: {args.destination}")


if __name__ == "__main__":
    main()
