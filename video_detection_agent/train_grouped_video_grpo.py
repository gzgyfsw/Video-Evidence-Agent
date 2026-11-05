#!/usr/bin/env python3
"""Historical whole-conversation GRPO baseline, disabled unless explicitly opted in.

Independent-request policies must use request_policy_replay for likelihoods.
Its exact scoring path is separate from this historical optimizer; this module
must not silently concatenate native requests into a new conditioning context.

Rollouts are generated beforehand by ``evaluate_causal_video_agent.py`` so
SigLIP retrieval and temporal frame expansion are real.  This trainer rebuilds
the exact causal messages/images, assigns log-probability only to model-written
Assistant JSON tokens, subtracts each prompt group's mean reward without
dividing by its standard deviation, and updates only the existing LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from grpo_reward import RewardBreakdown, RewardInput, compute_reward, reward_input_from_record
from train_sft_coldstart import ResponseOnlyCollator, save_adapter, set_seed


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_weight(path: Path) -> Path:
    weight = path.resolve() / "adapter_model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(f"missing adapter weight: {weight}")
    return weight


def action_stage(index: int, count: int, turn: dict | None = None) -> str:
    explicit = str((turn or {}).get("action_stage") or "")
    if explicit in {"initial_route", "temporal_route", "final_answer", "post_tool_assessment"}:
        return explicit
    if index == 0:
        return "initial_route"
    if index == count - 1:
        return "final_answer"
    parsed = (turn or {}).get("assistant_parsed") or {}
    if str(parsed.get("action") or "").upper() == "TEMPORAL_EXPAND":
        return "temporal_route"
    return "post_tool_assessment"


def make_trajectory(
    meta: dict,
    record: dict,
    reward: float,
    logprob_scope: str = "all_assistant",
) -> dict:
    if logprob_scope not in {
        "all_assistant", "final_assistant", "stagewise_causal", "stagewise_causal_v2",
    }:
        raise ValueError(f"unsupported logprob scope: {logprob_scope}")
    all_model_turns = record.get("model_turns") or []
    # A stage-specialized frozen answer model may run in a separate clean
    # visual context. Its output determines the environment reward but is not
    # a causal turn of the trainable route policy and must not be linearized
    # into that policy's transcript.
    model_turns = [
        turn for turn in all_model_turns
        if not turn.get("exclude_from_policy_trajectory")
    ]
    if record.get('request_replay_version') or record.get('context_layout') == 'independent_requests' or any(
            t.get('independent_context') or t.get('request_input_manifest') for t in model_turns):
        raise ValueError('Independent requests must use request_policy_replay; concatenation changes the policy conditioning context')
    if not model_turns:
        raise ValueError(f"{record.get('id')}: missing model_turns")
    turns = []
    for index, turn in enumerate(model_turns):
        raw = turn.get("assistant_raw")
        timestamps = turn.get("frame_timestamps")
        prompt = turn.get("user_prompt")
        if not isinstance(raw, str) or not isinstance(prompt, str) or not isinstance(timestamps, list):
            raise ValueError(f"{record.get('id')}: malformed model turn {index}")
        stage = action_stage(index, len(model_turns), turn)
        policy_owned = turn.get("policy_owned", True) is not False
        turns.append({"role": "user", "text": prompt, "timestamps": timestamps})
        turns.append({
            "role": "assistant",
            "content": raw,
            # A terminal trajectory reward must not reinforce an earlier
            # incorrect answer merely because the Python safety guard later
            # rescued the trajectory.  The final-assistant scope keeps all
            # causal history visible but assigns task-reward log-probability
            # only to the terminal model decision.
            "supervise": policy_owned and (
                logprob_scope == "all_assistant"
                or index == len(model_turns) - 1
                or (logprob_scope == "stagewise_causal" and index == 0)
                or (
                    logprob_scope == "stagewise_causal_v2"
                    and stage in {"initial_route", "temporal_route", "final_answer"}
                )
            ),
            "action_stage": stage,
            "policy_owned": policy_owned,
        })
    path = Path(meta["local_path"])
    if not path.is_absolute():
        path = ROOT / path
    return {
        "id": record["id"],
        "source_id": record.get("source_id") or meta.get("source_id") or meta["id"],
        "rollout_index": record.get("rollout_index"),
        "video_path": str(path),
        "turns": turns,
        "image_count": sum(len(turn.get("frame_timestamps") or []) for turn in model_turns),
        "reward": reward,
        "category": meta["category"],
        "severity": meta["severity"],
        "daa_group": meta.get("daa_group"),
        "logprob_scope": logprob_scope,
        "modular_temporal_only": bool(
            record.get("initial_route_model")
            and record.get("initial_route_model") != record.get("model")
        ),
        "has_temporal_stage": any(
            turn.get("role") == "assistant" and turn.get("action_stage") == "temporal_route"
            for turn in turns
        ),
    }


def causal_terminal_return(breakdown: RewardBreakdown) -> float:
    """Return assigned to the terminal answer, excluding earlier route cost/bonus."""
    if breakdown.task is None:
        raise ValueError("infrastructure reward has no causal terminal return")
    if not breakdown.exact:
        return float(breakdown.task)
    return round(
        float(breakdown.task + breakdown.expand + breakdown.grounding), 12,
    )


def causal_route_return(item: RewardInput, breakdown: RewardBreakdown) -> float:
    """Return-to-go for the model's first routing decision.

    A model-initiated scan and a policy-compliant direct L2 answer inherit the
    trajectory return.  Any premature/invalid ANSWER that only succeeds after
    the Python safety guard intervenes is an invalid first action and receives
    the frozen invalid-action reward.  This prevents framework rescue from
    turning a bad route into a positive training example.
    """
    if breakdown.total is None:
        raise ValueError("infrastructure reward has no causal route return")
    if item.model_initiated_joint_scan:
        return float(breakdown.total)
    if item.valid_direct_answer and not item.framework_rescued:
        return float(breakdown.total)
    return -5.0


def temporal_route_contract_valid(record: dict) -> bool:
    """Check whether a post-scan stop decision is supported by complete assessments.

    This is deliberately a route-policy contract, not a task-label heuristic.
    TEMPORAL_EXPAND is legal after runtime schema validation.  ANSWER is legal
    only when every risk is resolved as PRESENT or ABSENT; stopping while any
    risk remains INSUFFICIENT is an invalid temporal action even if a frozen
    answer helper later happens to recover the correct label.
    """
    turns = [
        turn for turn in (record.get("model_turns") or [])
        if turn.get("action_stage") == "temporal_route"
        and turn.get("policy_owned", True) is not False
    ]
    if not turns:
        return True
    if len(turns) != 1:
        return False
    parsed = turns[0].get("assistant_parsed") or {}
    action = str(parsed.get("action") or "").upper()
    if action == "TEMPORAL_EXPAND":
        return True
    if action != "ANSWER":
        return False
    assessments = parsed.get("assessments")
    if not isinstance(assessments, dict):
        return False
    required = {
        "Sexual Content",
        "Violence & Dangerous",
        "Sensitive Figures & Ideologies",
    }
    if not required.issubset(assessments):
        return False
    return all(
        str((assessments.get(risk) or {}).get("status") or "").upper()
        in {"PRESENT", "ABSENT"}
        for risk in required
    )


def causal_temporal_return(
    breakdown: RewardBreakdown,
    contract_valid: bool = True,
) -> float:
    """Return-to-go for an autonomous post-scan temporal decision.

    It excludes the already-taken initial-route bonus but retains task outcome,
    grounding, the fail-closed temporal bonus, and total tool/frame cost.  A
    useful expansion therefore outranks an unnecessary one without making
    tool cost large enough to dominate safety correctness.
    """
    if breakdown.total is None or breakdown.task is None:
        raise ValueError("infrastructure reward has no causal temporal return")
    if not contract_valid:
        return -5.0
    return round(
        float(breakdown.task + breakdown.expand + breakdown.grounding - breakdown.cost),
        12,
    )


def temporal_comparison_rows(values: list[dict]) -> tuple[list[dict], bool]:
    """Return rollouts eligible for the post-scan group baseline.

    In the modular v2 architecture the initial router is frozen and only the
    post-scan policy is optimized.  A rollout that exits directly from the
    frozen initial router never visits a state controlled by the temporal
    policy.  Including that exogenous branch in the temporal group mean would
    assign its reward difference to a policy that could not have changed it.
    Compare only siblings that actually reached the post-scan decision.  The
    legacy shared-policy path continues to use every rollout in the group.
    """
    flags = [bool(row.get("modular_temporal_only")) for row in values]
    if any(flags) and not all(flags):
        raise ValueError("a prompt group mixes modular and shared-policy rollouts")
    modular = bool(flags and all(flags))
    if modular:
        return [row for row in values if row.get("has_temporal_stage")], True
    return values, False


def build_dataset(
    manifest: Path,
    rollouts: Path,
    group_size: int,
    logprob_scope: str = "all_assistant",
    route_advantage_weight: float = 1.0,
    terminal_advantage_weight: float = 1.0,
    temporal_advantage_weight: float = 0.0,
    route_credit_scope: str = "initial_payload",
    temporal_credit_scope: str = "action",
    enforce_temporal_answer_contract: bool = False,
) -> tuple[list[dict], dict]:
    if route_advantage_weight < 0 or terminal_advantage_weight < 0 or temporal_advantage_weight < 0:
        raise ValueError("causal advantage weights must be non-negative")
    if logprob_scope in {"stagewise_causal", "stagewise_causal_v2"} and not (
        route_advantage_weight or terminal_advantage_weight or temporal_advantage_weight
    ):
        raise ValueError("stagewise causal training requires at least one non-zero advantage weight")
    if route_credit_scope not in {"action", "initial_payload"}:
        raise ValueError(f"unsupported route credit scope: {route_credit_scope}")
    if temporal_credit_scope not in {"action", "payload"}:
        raise ValueError(f"unsupported temporal credit scope: {temporal_credit_scope}")
    meta_by_id = {row["id"]: row for row in read_jsonl(manifest)}
    records = read_jsonl(rollouts)
    duplicate_ids = len(records) - len({row.get("id") for row in records})
    if duplicate_ids:
        raise ValueError(f"rollout file contains {duplicate_ids} duplicate ids")
    record_by_id = {row["id"]: row for row in records}
    missing = sorted(set(meta_by_id) - set(record_by_id))
    unexpected = sorted(set(record_by_id) - set(meta_by_id))
    if missing or unexpected:
        raise ValueError(f"rollout mismatch: missing={len(missing)} unexpected={len(unexpected)}")

    rows = []
    reward_reasons = Counter()
    reward_details = []
    for sample_id, meta in meta_by_id.items():
        record = record_by_id[sample_id]
        item = reward_input_from_record(meta, record)
        breakdown = compute_reward(item)
        if breakdown.total is None:
            raise ValueError(f"{sample_id}: infrastructure failures cannot enter GRPO")
        reward_reasons[breakdown.reason] += 1
        trajectory = make_trajectory(
            meta, record, float(breakdown.total), logprob_scope=logprob_scope,
        )
        contract_valid = temporal_route_contract_valid(record)
        trajectory["route_return"] = causal_route_return(item, breakdown)
        trajectory["temporal_return"] = causal_temporal_return(
            breakdown,
            contract_valid=(contract_valid or not enforce_temporal_answer_contract),
        )
        trajectory["terminal_return"] = causal_terminal_return(breakdown)
        trajectory["route_credit_scope"] = route_credit_scope
        trajectory["temporal_credit_scope"] = temporal_credit_scope
        trajectory["temporal_route_contract_valid"] = contract_valid
        rows.append(trajectory)
        reward_details.append({
            "id": sample_id,
            "source_id": record.get("source_id") or meta.get("source_id") or meta["id"],
            "reward_input": item.__dict__,
            "reward": breakdown.to_dict(),
            "temporal_route_contract_valid": contract_valid,
            "temporal_answer_contract_enforced": enforce_temporal_answer_contract,
        })

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    wrong_sizes = {key: len(values) for key, values in groups.items() if len(values) != group_size}
    if wrong_sizes:
        raise ValueError(f"incomplete prompt groups: {wrong_sizes}")

    informative_groups = 0
    route_informative_groups = 0
    temporal_informative_groups = 0
    terminal_informative_groups = 0
    modular_prompt_groups = 0
    temporal_reached_rollouts = 0
    temporal_comparison_rollouts = 0
    for values in groups.values():
        mean_reward = sum(row["reward"] for row in values) / len(values)
        mean_route_return = sum(row["route_return"] for row in values) / len(values)
        temporal_values, modular_group = temporal_comparison_rows(values)
        if modular_group:
            modular_prompt_groups += 1
        temporal_reached_rollouts += sum(bool(row.get("has_temporal_stage")) for row in values)
        temporal_comparison_rollouts += len(temporal_values)
        mean_temporal_return = (
            sum(row["temporal_return"] for row in temporal_values) / len(temporal_values)
            if temporal_values else 0.0
        )
        mean_terminal_return = sum(row["terminal_return"] for row in values) / len(values)
        total_varies = max(row["reward"] for row in values) - min(row["reward"] for row in values) > 1e-12
        route_varies = max(row["route_return"] for row in values) - min(row["route_return"] for row in values) > 1e-12
        temporal_varies = (
            len(temporal_values) > 1
            and max(row["temporal_return"] for row in temporal_values)
            - min(row["temporal_return"] for row in temporal_values) > 1e-12
        )
        terminal_varies = max(row["terminal_return"] for row in values) - min(row["terminal_return"] for row in values) > 1e-12
        if route_varies:
            route_informative_groups += 1
        if temporal_varies:
            temporal_informative_groups += 1
        if terminal_varies:
            terminal_informative_groups += 1
        stagewise_varies = (
            (route_advantage_weight > 0 and route_varies)
            or (temporal_advantage_weight > 0 and temporal_varies)
            or (terminal_advantage_weight > 0 and terminal_varies)
        )
        if total_varies if logprob_scope not in {"stagewise_causal", "stagewise_causal_v2"} else stagewise_varies:
            informative_groups += 1
        for row in values:
            # Deliberately no group-standard-deviation normalization.
            row["advantage"] = row["reward"] - mean_reward
            row["group_mean_reward"] = mean_reward
            row["route_advantage"] = row["route_return"] - mean_route_return
            row["temporal_advantage"] = (
                row["temporal_return"] - mean_temporal_return
                if not modular_group or row.get("has_temporal_stage") else 0.0
            )
            row["terminal_advantage"] = row["terminal_return"] - mean_terminal_return
            row["has_training_signal"] = (
                abs(row["advantage"]) > 1e-12
                if logprob_scope not in {"stagewise_causal", "stagewise_causal_v2"}
                else (
                    route_advantage_weight * abs(row["route_advantage"]) > 1e-12
                    or (
                        row.get("has_temporal_stage")
                        and temporal_advantage_weight * abs(row["temporal_advantage"]) > 1e-12
                    )
                    or terminal_advantage_weight * abs(row["terminal_advantage"]) > 1e-12
                )
            )
    audit = {
        "rollouts": len(rows),
        "prompt_groups": len(groups),
        "group_size": group_size,
        "informative_groups": informative_groups,
        "route_informative_groups": route_informative_groups,
        "temporal_informative_groups": temporal_informative_groups,
        "terminal_informative_groups": terminal_informative_groups,
        "modular_prompt_groups": modular_prompt_groups,
        "temporal_reached_rollouts": temporal_reached_rollouts,
        "temporal_comparison_rollouts": temporal_comparison_rollouts,
        "zero_variance_groups": len(groups) - informative_groups,
        "nonzero_advantage_rollouts": sum(row["has_training_signal"] for row in rows),
        "reward_reason_counts": dict(reward_reasons),
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "logprob_scope": logprob_scope,
        "route_advantage_weight": route_advantage_weight,
        "temporal_advantage_weight": temporal_advantage_weight,
        "terminal_advantage_weight": terminal_advantage_weight,
        "route_credit_scope": route_credit_scope,
        "temporal_credit_scope": temporal_credit_scope,
        "temporal_answer_contract_enforced": enforce_temporal_answer_contract,
        "temporal_route_contract_violations": sum(
            not row["temporal_route_contract_valid"] for row in rows
        ),
        "reward_details": reward_details,
    }
    return rows, audit


def model_batch(
    collator: ResponseOnlyCollator,
    row: dict,
    device: str = "cuda",
) -> tuple[dict, torch.Tensor, dict[str, torch.Tensor]]:
    batch = collator([row])
    labels = batch.pop("labels")
    route_mask = batch.pop("route_mask")
    temporal_route_mask = batch.pop("temporal_route_mask", None)
    action_spans = batch.pop("action_spans", None)
    action_stage_names = batch.pop("action_stage_names", None)
    for key in (
        "sample_id", "stage", "sequence_length", "target_tokens", "route_tokens",
    ):
        batch.pop(key, None)
    target_positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    if not len(target_positions) or int(target_positions.min()) == 0:
        raise RuntimeError(f"{row['id']}: invalid supervised token positions")
    targets = batch["input_ids"][0, target_positions].to(device)
    prediction_positions = (target_positions - 1).to(device)
    credit_masks: dict[str, torch.Tensor] = {}
    if row.get("logprob_scope") in {"stagewise_causal", "stagewise_causal_v2"}:
        if not action_spans or not action_stage_names or len(action_spans) != len(action_stage_names):
            raise RuntimeError(f"{row['id']}: stagewise trajectory has no action spans")
        modular_temporal_only = bool(row.get("modular_temporal_only"))
        if modular_temporal_only:
            # The accepted GRPO-v1 adapter generated the frozen initial turn;
            # only the post-scan temporal adapter is on-policy in GRPO-v2.
            # Keep zero route/terminal masks with the same target shape so the
            # generic stagewise objective can remain unchanged.
            credit_masks = {
                "route": torch.zeros_like(target_positions, dtype=torch.bool, device=device),
                "terminal": torch.zeros_like(target_positions, dtype=torch.bool, device=device),
            }
        else:
            try:
                first_index = action_stage_names.index("initial_route")
            except ValueError as exc:
                raise RuntimeError(f"{row['id']}: missing initial stage") from exc
            # The last persisted model response is always the terminal response for
            # reward assignment.  In a two-turn v2 trajectory, post-scan ANSWER is
            # simultaneously the temporal routing choice and the final answer, so
            # its one token span must receive both temporal and terminal credit.
            final_index = len(action_stage_names) - 1
            first_start, first_end = map(int, action_spans[first_index])
            final_start, final_end = map(int, action_spans[final_index])
            first_span = target_positions.ge(first_start) & target_positions.lt(first_end)
            final_span = target_positions.ge(final_start) & target_positions.lt(final_end)
            same_span = first_start == final_start and first_end == final_end
            # A multi-turn scan owns its entire first JSON (action plus generated
            # retrieval queries).  A single-turn direct answer already receives
            # terminal credit over the whole payload, so add route credit only to
            # the action value rather than double-weighting category/evidence text.
            action_credit = route_mask[0, target_positions]
            if same_span or row.get("route_credit_scope") == "action":
                # Malformed first actions occasionally omit the canonical
                # ``action`` key (for example, a top-level JOINT_SEMANTIC_SCAN
                # object).  There is then no action-value token to down-weight.
                # Fall back to the generated first payload only for that invalid
                # row so its negative route return can still suppress the bad
                # syntax, while valid rows remain action-token-only.
                initial_credit = action_credit if action_credit.any() else first_span
            else:
                initial_credit = first_span
            if not initial_credit.any() or not final_span.any():
                raise RuntimeError(f"{row['id']}: invalid stagewise route/final token masks")
            credit_masks = {
                "route": initial_credit.to(device),
                "terminal": final_span.to(device),
            }
        temporal_credit = torch.zeros_like(target_positions, dtype=torch.bool)
        if row.get("temporal_credit_scope", "action") == "payload":
            for stage, span in zip(action_stage_names, action_spans):
                if stage == "temporal_route":
                    temporal_start, temporal_end = map(int, span)
                    temporal_credit |= (
                        target_positions.ge(temporal_start)
                        & target_positions.lt(temporal_end)
                    )
        elif temporal_route_mask is not None:
            temporal_credit = temporal_route_mask[0, target_positions]
        # Invalid historical payloads can omit the canonical action field. In
        # that exceptional case retain a full-span negative signal so malformed
        # policy output can still be suppressed. Valid temporal decisions are
        # always action-token-only.
        if not temporal_credit.any():
            for stage, span in zip(action_stage_names, action_spans):
                if stage == "temporal_route":
                    temporal_start, temporal_end = map(int, span)
                    temporal_credit |= (
                        target_positions.ge(temporal_start)
                        & target_positions.lt(temporal_end)
                    )
        if modular_temporal_only and not temporal_credit.any():
            raise RuntimeError(f"{row['id']}: modular temporal trajectory has no temporal action")
        credit_masks["temporal"] = temporal_credit.to(device)
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    batch["logits_to_keep"] = prediction_positions
    return batch, targets, credit_masks


def assistant_logps(model, batch: dict, targets: torch.Tensor) -> torch.Tensor:
    outputs = model(**batch, use_cache=False)
    logits = outputs.logits[0]
    if logits.shape[0] != targets.shape[0]:
        raise RuntimeError(f"selected logits/targets mismatch: {logits.shape[0]} != {targets.shape[0]}")
    # The subtraction is mathematically log_softmax+gather but only retains one
    # scalar per supervised token instead of a full vocabulary probability map.
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    return chosen - torch.logsumexp(logits.float(), dim=-1)


def set_policy_adapter(model) -> None:
    model.set_adapter("default", inference_mode=False)


def set_reference_adapter(model) -> None:
    model.set_adapter("reference", inference_mode=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy-adapter", type=Path, required=True)
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--groups-per-update", type=int, default=8,
        help="Prompt groups per PPO minibatch update; one rollout batch is one clipped-policy epoch",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=384 * 384)
    parser.add_argument("--max-sequence-length", type=int, default=16384)
    parser.add_argument(
        "--save-steps", type=int, default=0,
        help="Save intermediate policy adapters every N optimizer steps; 0 disables it",
    )
    parser.add_argument(
        "--logprob-scope",
        choices=("all_assistant", "final_assistant", "stagewise_causal", "stagewise_causal_v2"),
        default="all_assistant",
        help=(
            "Assistant turns receiving the terminal trajectory advantage. "
            "final_assistant keeps earlier turns as causal context but prevents "
            "framework-rescued early mistakes from receiving positive task credit; "
            "stagewise_causal assigns independent group-relative returns to the "
            "initial route/query and terminal answer token spans."
        ),
    )
    parser.add_argument(
        "--temporal-advantage-weight", type=float, default=0.0,
        help="Multiplier for autonomous post-scan TEMPORAL_EXPAND token advantages",
    )
    parser.add_argument(
        "--route-advantage-weight", type=float, default=1.0,
        help="Multiplier for stagewise_causal initial route/query token advantages",
    )
    parser.add_argument(
        "--terminal-advantage-weight", type=float, default=1.0,
        help="Multiplier for stagewise_causal terminal answer token advantages",
    )
    parser.add_argument(
        "--route-credit-scope", choices=("action", "initial_payload"),
        default="initial_payload",
        help=(
            "Token span receiving causal route advantage: only the generated "
            "ANSWER/JOINT_SEMANTIC_SCAN value, or the complete initial JSON/query"
        ),
    )
    parser.add_argument(
        "--temporal-credit-scope", choices=("action", "payload"), default="action",
        help=(
            "Token span receiving temporal-route advantage. Use payload only "
            "for targeted schema/argument repair; action remains the default."
        ),
    )
    parser.add_argument(
        "--enforce-temporal-answer-contract", action="store_true",
        help=("Assign -5 temporal-route return when ANSWER is emitted while "
              "any post-scan risk assessment remains INSUFFICIENT"),
    )
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument('--legacy-linearized-baseline',action='store_true',
        help='Explicitly reproduce the historical whole-conversation baseline only; never an aligned independent-request GRPO run')
    args = parser.parse_args()
    if not args.legacy_linearized_baseline:
        raise RuntimeError('Legacy conversation-reconstruction optimizer is disabled by default. '
            'Use request_policy_replay for exact per-request scoring. The aligned request-level optimizer '
            'requires a separate implementation/validation; --legacy-linearized-baseline is only for historical reproduction.')
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires CUDA")
    set_seed(args.seed)
    rows, audit = build_dataset(
        args.manifest, args.rollouts, args.group_size, args.logprob_scope,
        args.route_advantage_weight, args.terminal_advantage_weight,
        args.temporal_advantage_weight,
        args.route_credit_scope,
        args.temporal_credit_scope,
        args.enforce_temporal_answer_contract,
    )
    if audit["informative_groups"] == 0:
        raise RuntimeError("all prompt groups have zero reward variance; no GRPO learning signal")
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "training_trajectories.jsonl", rows)
    (args.output / "reward_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Zero-advantage rollouts contribute exactly no policy gradient.  Build
    # the informative minibatches before model loading so reference caching
    # and multimodal decoding are not wasted on those rows.
    informative_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["has_training_signal"]:
            informative_by_source[row["source_id"]].append(row)
    prompt_batches = list(informative_by_source.values())
    random.Random(args.seed).shuffle(prompt_batches)
    prompt_batches = [
        prompt_batches[index:index + args.groups_per_update]
        for index in range(0, len(prompt_batches), args.groups_per_update)
    ]
    informative = [row for values in informative_by_source.values() for row in values]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, args.policy_adapter, is_trainable=True)
    model.load_adapter(args.reference_adapter, adapter_name="reference", is_trainable=False)
    # Loading a frozen second adapter may leave all adapters inactive/frozen in
    # some PEFT versions. Reactivate the policy explicitly before collecting
    # optimizer parameters.
    set_policy_adapter(model)
    model.enable_input_require_grads()
    model.to("cuda")
    collator = ResponseOnlyCollator(
        processor, args.max_pixels, args.max_sequence_length,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("policy adapter exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    # Cache old-policy and frozen-reference log-probabilities before updating.
    old_reference_equal = args.policy_adapter.resolve() == args.reference_adapter.resolve()
    cached = {}
    model.eval()
    start = time.time()
    for index, row in enumerate(informative, 1):
        batch, targets, _ = model_batch(collator, row)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            set_policy_adapter(model)
            old_logps = assistant_logps(model, batch, targets).cpu()
            if old_reference_equal:
                reference_logps = old_logps.clone()
            else:
                set_reference_adapter(model)
                reference_logps = assistant_logps(model, batch, targets).cpu()
        cached[row["id"]] = {"old": old_logps, "reference": reference_logps}
        del batch, targets
        if index == 1 or index % 5 == 0:
            print(
                f"reference={index}/{len(informative)} id={row['id']} "
                f"elapsed={time.time()-start:.1f}s", flush=True,
            )
    # Gradient checkpointing in Transformers is active only in training mode.
    # Enable it for long multimodal trajectories, but force every dropout to
    # zero so current/old-policy ratios remain comparable to inference.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    set_policy_adapter(model)
    scalar_metrics = []
    grad_norms = []
    trained_rollouts = 0
    skipped_no_credit_rollouts = 0
    skipped_empty_updates = 0
    for optimizer_step, group_batch in enumerate(prompt_batches, 1):
        minibatch = [row for values in group_batch for row in values]
        optimizer.zero_grad(set_to_none=True)
        step_trained_rollouts = 0
        for row in minibatch:
            batch, targets, credit_masks = model_batch(collator, row)
            old_logps = cached[row["id"]]["old"].to("cuda")
            reference_logps = cached[row["id"]]["reference"].to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                current_logps = assistant_logps(model, batch, targets)
                ratio = torch.exp(current_logps - old_logps)
                if args.logprob_scope in {"stagewise_causal", "stagewise_causal_v2"}:
                    advantage = (
                        credit_masks["route"].float()
                        * float(row["route_advantage"])
                        * args.route_advantage_weight
                        + credit_masks["temporal"].float()
                        * float(row["temporal_advantage"])
                        * args.temporal_advantage_weight
                        + credit_masks["terminal"].float()
                        * float(row["terminal_advantage"])
                        * args.terminal_advantage_weight
                    )
                else:
                    advantage = torch.tensor(row["advantage"], device="cuda", dtype=torch.float32)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) * advantage
                if args.logprob_scope in {"stagewise_causal", "stagewise_causal_v2"}:
                    credited = advantage.ne(0)
                    if not credited.any():
                        skipped_no_credit_rollouts += 1
                        scalar_metrics.append({
                            "id": row["id"],
                            "source_id": row["source_id"],
                            "optimizer_step": optimizer_step,
                            "reward": row["reward"],
                            "advantage": row["advantage"],
                            "route_advantage": row["route_advantage"],
                            "temporal_advantage": row["temporal_advantage"],
                            "terminal_advantage": row["terminal_advantage"],
                            "assistant_tokens": int(targets.numel()),
                            "route_credit_tokens": int(credit_masks.get("route", torch.tensor([])).sum()),
                            "temporal_credit_tokens": int(credit_masks.get("temporal", torch.tensor([])).sum()),
                            "terminal_credit_tokens": int(credit_masks.get("terminal", torch.tensor([])).sum()),
                            "policy_loss": None,
                            "kl": None,
                            "skipped_no_credit_tokens": True,
                        })
                        del batch, targets, credit_masks, old_logps, reference_logps, current_logps, ratio
                        continue
                    policy_loss = -torch.minimum(unclipped, clipped)[credited].mean()
                else:
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                delta = reference_logps - current_logps
                kl = (torch.exp(delta) - delta - 1.0).mean()
                loss = (policy_loss + args.beta * kl) / len(minibatch)
            loss.backward()
            trained_rollouts += 1
            step_trained_rollouts += 1
            scalar_metrics.append({
                "id": row["id"],
                "source_id": row["source_id"],
                "optimizer_step": optimizer_step,
                "reward": row["reward"],
                "advantage": row["advantage"],
                "route_advantage": row["route_advantage"],
                "temporal_advantage": row["temporal_advantage"],
                "terminal_advantage": row["terminal_advantage"],
                "assistant_tokens": int(targets.numel()),
                "route_credit_tokens": int(credit_masks.get("route", torch.tensor([])).sum()),
                "temporal_credit_tokens": int(credit_masks.get("temporal", torch.tensor([])).sum()),
                "terminal_credit_tokens": int(credit_masks.get("terminal", torch.tensor([])).sum()),
                "policy_loss": float(policy_loss.detach()),
                "kl": float(kl.detach()),
            })
            del batch, targets, credit_masks, old_logps, reference_logps, current_logps, ratio, loss
        if step_trained_rollouts == 0:
            skipped_empty_updates += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"update={optimizer_step}/{len(prompt_batches)} skipped_empty_update "
                f"groups={len(group_batch)} rollouts={trained_rollouts}/{len(informative)} "
                f"skipped_no_credit={skipped_no_credit_rollouts} elapsed={time.time()-start:.1f}s",
                flush=True,
            )
            continue
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
        grad_norms.append(grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.save_steps and optimizer_step % args.save_steps == 0:
            checkpoint_state = {
                "complete": False,
                "optimizer_step": optimizer_step,
                "rollouts": len(rows),
                "prompt_groups": audit["prompt_groups"],
                "informative_groups": audit["informative_groups"],
                "nonzero_advantage_rollouts": len(informative),
                "elapsed_seconds": time.time() - start,
                "logprob_scope": args.logprob_scope,
                "route_credit_scope": args.route_credit_scope,
                "temporal_credit_scope": args.temporal_credit_scope,
                "route_advantage_weight": args.route_advantage_weight,
                "temporal_advantage_weight": args.temporal_advantage_weight,
                "terminal_advantage_weight": args.terminal_advantage_weight,
                "test_split_used": False,
            }
            save_adapter(
                model, processor,
                args.output / f"checkpoint-{optimizer_step}" / "adapter",
                checkpoint_state,
            )
        last_trained_metric = next(
            item for item in reversed(scalar_metrics)
            if item.get("policy_loss") is not None
        )
        print(
            f"update={optimizer_step}/{len(prompt_batches)} groups={len(group_batch)} "
            f"rollouts={trained_rollouts}/{len(informative)} grad_norm={grad_norm:.6f} "
            f"last_policy_loss={last_trained_metric['policy_loss']:.6f} "
            f"last_kl={last_trained_metric['kl']:.6f} "
            f"skipped_no_credit={skipped_no_credit_rollouts} elapsed={time.time()-start:.1f}s",
            flush=True,
        )

    trained_metrics = [item for item in scalar_metrics if item.get("policy_loss") is not None]
    if not trained_metrics:
        raise RuntimeError("all selected stagewise rows had no credited tokens; no GRPO update was applied")
    state = {
        "complete": True,
        "optimizer_steps": len(prompt_batches),
        "applied_optimizer_steps": len(grad_norms),
        "skipped_empty_updates": skipped_empty_updates,
        "skipped_no_credit_rollouts": skipped_no_credit_rollouts,
        "rollouts": len(rows),
        "prompt_groups": audit["prompt_groups"],
        "informative_groups": audit["informative_groups"],
        "nonzero_advantage_rollouts": len(informative),
        "mean_reward": audit["mean_reward"],
        "mean_policy_loss": sum(item["policy_loss"] for item in trained_metrics) / len(trained_metrics),
        "mean_kl": sum(item["kl"] for item in trained_metrics) / len(trained_metrics),
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "max_grad_norm_observed": max(grad_norms),
        "elapsed_seconds": time.time() - start,
        "scale_rewards": "group_mean_only_no_std",
        "policy_advantage_reduction": (
            "nonzero_credit_tokens_mean"
            if args.logprob_scope in {"stagewise_causal", "stagewise_causal_v2"}
            else "all_selected_tokens_mean"
        ),
        "test_split_used": False,
        "temporal_credit_scope": args.temporal_credit_scope,
    }
    save_adapter(model, processor, args.output / "adapter", state)
    write_jsonl(args.output / "update_metrics.jsonl", scalar_metrics)
    config = {
        **{key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_sha256": sha256(args.manifest),
        "rollouts_sha256": sha256(args.rollouts),
        "old_reference_equal": old_reference_equal,
        "objective": "clipped_group_relative_policy_gradient_with_forward_kl",
        "assistant_only_logprob": True,
        "framework_and_tool_tokens_in_logprob": False,
        "sha256": {
            "trainer": sha256(Path(__file__)),
            "reward": sha256(Path(__file__).with_name("grpo_reward.py")),
            "collator": sha256(Path(__file__).with_name("train_sft_coldstart.py")),
            "policy_adapter": sha256(adapter_weight(args.policy_adapter)),
            "reference_adapter": sha256(adapter_weight(args.reference_adapter)),
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "peft")
        },
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "COMPLETE").write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
    print("GRPO_UPDATE_COMPLETE " + json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
