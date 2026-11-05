#!/usr/bin/env python3
"""Resumable dataset evaluation for the prompted tool-calling Agent."""

import argparse
import gc
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from agent.policy_agent import PolicyAgentConfig, ToolCallingVideoSafetyAgent
from agent.policy_agent import AgentRunError
from agent.qwen_backend import QwenTokenLikelihoodClassifier, canonical_category
from agent.tools import MotionEventLocator, RiskSemanticLocator, TemporalEvidenceExpander
from scripts.prepare_video_safety_rl_data import rewrite_media_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--id", action="append", dest="sample_ids",
                        help="run only the specified sample id; may be repeated")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--selection", choices=["first", "random"], default="first",
                        help="balanced selection within each label; random is deterministic with --seed")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--selection-scores", type=Path,
                        help="optional prior JSONL; lowest raw_confidence per class are diagnostic hard cases")
    parser.add_argument("--initial-frames", type=int, default=0,
                        help="0 selects duration-adaptive 4/8/12/16 initial frames")
    parser.add_argument("--expansion-frames", type=int, default=4)
    parser.add_argument("--max-total-frames", type=int, default=24)
    parser.add_argument("--max-frames-per-action", type=int, default=8)
    parser.add_argument("--locator-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--locator-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--locator-candidate-frames", type=int, default=0,
                        help="0 selects duration-adaptive 2-second sampling")
    parser.add_argument("--max-pixels", type=int, default=224 * 224,
                        help="maximum pixels per frame sent to the VLM")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def video_path(row: dict) -> str:
    if row.get("local_path"):
        path = Path(row["local_path"])
        return str(path if path.is_absolute() else REPO_ROOT / path)
    messages = json.loads(row["message"], strict=False)
    for message in messages:
        for part in message.get("content", []):
            if isinstance(part, dict) and "video" in part:
                return rewrite_media_path(part["video"], REPO_ROOT)
    raise ValueError(f"no video in sample {row.get('id')}")


def select_rows(rows: list[dict], per_class: int, maximum: int, scores_path: Path | None,
                selection: str = "first", seed: int = 20260805) -> list[dict]:
    score = {}
    if scores_path:
        score = {row["id"]: float(row["raw_confidence"]) for row in read_jsonl(scores_path)}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("answer", row.get("category"))].append(row)
    chosen = []
    for category in sorted(grouped):
        candidates = sorted(grouped[category], key=lambda row: row["id"])
        if scores_path:
            candidates.sort(key=lambda row: (score.get(row["id"], 1.0), row["id"]))
        elif selection == "random":
            random.Random(f"{seed}:{category}").shuffle(candidates)
        existing = []
        for row in candidates:
            path = Path(video_path(row))
            if path.is_file() and path.stat().st_size > 0:
                existing.append(row)
            if len(existing) == per_class:
                break
        chosen.extend(existing)
    return chosen[:maximum]


def metrics(records: list[dict]) -> dict:
    answered = [row for row in records if row["final_action"] == "ANSWER"]
    correct = [row for row in answered if row["pred"] == row["label"]]
    labels = sorted({row["label"] for row in records} | {row.get("pred") for row in answered})
    confusion = {
        label: {pred: sum(row["label"] == label and row.get("pred") == pred for row in answered)
                for pred in labels}
        for label in labels
    }
    harmful = [row for row in records if row["label"] != "Harmless"]
    harmless = [row for row in records if row["label"] == "Harmless"]
    per_class_frames = {
        label: sum(row["total_frames"] for row in records if row["label"] == label)
        / sum(row["label"] == label for row in records)
        for label in sorted({row["label"] for row in records})
    }
    return {
        "n": len(records),
        "answered": len(answered),
        "accuracy": len(correct) / len(records) if records else None,
        "mean_frames": sum(row["total_frames"] for row in records) / len(records) if records else None,
        "tool_calls": dict(Counter(tool for row in records for tool in row["tool_calls"])),
        "labels": dict(Counter(row["label"] for row in records)),
        "harmful_recall": (
            sum(row.get("pred") == row["label"] for row in harmful) / len(harmful)
            if harmful else None
        ),
        "harmless_false_positive_rate": (
            sum(row.get("pred") != "Harmless" for row in harmless) / len(harmless)
            if harmless else None
        ),
        "mean_frames_by_label": per_class_frames,
        "confusion_matrix": confusion,
    }


def main():
    args = parse_args()
    source_rows = read_jsonl(args.data)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        rows = [row for row in source_rows if row.get("id") in wanted]
        missing = wanted - {row.get("id") for row in rows}
        if missing:
            raise SystemExit(f"sample ids not found: {sorted(missing)}")
    else:
        rows = select_rows(source_rows, args.per_class, args.max_samples,
                           args.selection_scores, args.selection, args.seed)
    cached = {row["id"]: row for row in read_jsonl(args.output)} if args.output.exists() else {}
    print(f"selected={len(rows)} cached={sum(row['id'] in cached for row in rows)}", flush=True)

    classifier = QwenTokenLikelihoodClassifier(args.model_path, max_pixels=args.max_pixels)
    locator = RiskSemanticLocator(
        args.locator_model, args.locator_candidate_frames or None, args.locator_device
    )
    agent = ToolCallingVideoSafetyAgent(
        classifier, locator, TemporalEvidenceExpander(),
        PolicyAgentConfig(
            initial_frames=args.initial_frames or None,
            default_action_frames=args.expansion_frames,
            max_total_frames=args.max_total_frames,
            max_frames_per_action=args.max_frames_per_action,
        ),
        motion_locator=MotionEventLocator(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows, 1):
        if row["id"] in cached:
            records.append(cached[row["id"]])
            print(f"[{index}/{len(rows)}] cached {row['id']}", flush=True)
            continue
        try:
            result = agent.run(video_path(row))
            payload = result.to_dict()
            action = payload["final_action"]
            record = {
                "id": row["id"],
                "label": canonical_category(row.get("answer", row.get("category"))),
                "final_action": action["action"],
                "pred": canonical_category(action.get("category")) if action["action"] == "ANSWER" else None,
                "severity": action.get("severity"),
                "moderation_action": action.get("moderation_action"),
                "total_frames": payload["total_frames"],
                "tool_calls": [entry["tool"] for entry in payload["transcript"] if "tool" in entry],
                "result": payload,
            }
        except AgentRunError as exc:
            record = {"id": row["id"],
                      "label": canonical_category(row.get("answer", row.get("category"))),
                      "error": f"{type(exc).__name__}: {exc}", "partial_result": exc.to_dict()}
        except Exception as exc:
            record = {"id": row["id"],
                      "label": canonical_category(row.get("answer", row.get("category"))),
                      "error": f"{type(exc).__name__}: {exc}"}
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(f"[{index}/{len(rows)}] {record['id']} label={record['label']} "
              f"action={record.get('final_action')} pred={record.get('pred')} "
              f"frames={record.get('total_frames')} error={record.get('error')}", flush=True)
        gc.collect()
        if classifier.torch.cuda.is_available():
            classifier.torch.cuda.empty_cache()

    valid = [row for row in records if "error" not in row]
    summary = metrics(valid)
    summary["errors"] = len(records) - len(valid)
    summary["selection"] = (
        "lowest prior 4B confidence per class" if args.selection_scores
        else f"{args.selection} per class (seed={args.seed})"
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved {args.output} and {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
