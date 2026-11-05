#!/usr/bin/env python3
"""Compare paired current-SFT/GRPO Test receipts against frozen weak references."""
import argparse
import json
from pathlib import Path

from frozen_test_manifest import load_frozen_test
from hf_frozen_test_policy import read_test_item


def load(path):
    return json.loads(Path(path).read_text())


def collect(root, manifest):
    plan = load(root / "PLAN.json")
    rows = {}
    prompts = {}
    for entry in manifest:
        ident = entry["source"]["merged_id"]
        receipt_path = root / "rollouts" / ident / "receipt.json"
        if not receipt_path.exists():
            raise ValueError(f"Missing Test receipt: {ident}")
        row = load(receipt_path)
        if (row.get("source_id") != ident or row.get("source_partition") != "test"
                or row.get("seed") != entry["seed"]
                or row.get("model_binding") != plan["model_binding"]):
            raise ValueError(f"Foreign or mismatched Test receipt: {ident}")
        if not row.get("reference_labels_hidden"):
            raise ValueError(f"Test reference exposure is not explicitly false: {ident}")
        requests = row.get("requests", [])
        if not requests:
            raise ValueError(f"Test receipt has no model request: {ident}")
        _, _, response = read_test_item(requests[0]["item_path"])
        prompts[ident] = response["prompt_tensor_receipt"]
        rows[ident] = row
    return plan, rows, prompts


def summarize(rows, refs):
    total = len(rows)
    category = severity = both = terminal = 0
    tools = requests = frames = 0
    for ident, row in rows.items():
        answer = row.get("answer") or {}
        ref = refs[ident]
        terminal += row.get("status") == "MODEL_ANSWER_UNREVIEWED"
        cat_ok = answer.get("category") == ref["weak_category"]
        sev_ok = answer.get("severity") == ref["weak_severity"]
        category += cat_ok
        severity += sev_ok
        both += cat_ok and sev_ok
        tools += len(row.get("actual_tools", []))
        requests += len(row.get("requests", []))
        frames += sum(len(call.get("returned_timestamps", [])) for call in row.get("actual_tools", []))
    return {
        "videos": total,
        "terminal_reports": terminal,
        "weak_category_agreement": category / total,
        "weak_severity_agreement": severity / total,
        "weak_category_and_severity_agreement": both / total,
        "actual_tool_calls": tools,
        "returned_new_frames": frames,
        "model_requests": requests,
        "mean_tool_calls": tools / total,
        "mean_new_frames": frames / total,
        "mean_model_requests": requests / total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    _, manifest, expected = load_frozen_test(a.root)
    refs_list = load(a.root / "WEAK_REFERENCES.json")
    refs = {row["source_id"]: row for row in refs_list}
    ids = {row["source"]["merged_id"] for row in manifest}
    if set(refs) != ids or len(refs) != expected:
        raise ValueError("Frozen Test weak-reference coverage differs")
    left_plan, left, left_prompts = collect(a.root / "sft_v4", manifest)
    right_plan, right, right_prompts = collect(a.root / "candidate", manifest)
    for field in ["manifest_sha256", "policy_sha256", "budget", "sampling", "environment", "version"]:
        if left_plan[field] != right_plan[field]:
            raise ValueError(f"Paired Test contract differs: {field}")
    if any(left_prompts[i] != right_prompts[i] for i in ids):
        raise ValueError("Paired Test initial inputs differ")
    improved = worsened = changed = 0
    for ident in ids:
        la, ra, ref = left[ident].get("answer") or {}, right[ident].get("answer") or {}, refs[ident]
        l_ok = (la.get("category"), la.get("severity")) == (ref["weak_category"], ref["weak_severity"])
        r_ok = (ra.get("category"), ra.get("severity")) == (ref["weak_category"], ref["weak_severity"])
        improved += r_ok and not l_ok
        worsened += l_ok and not r_ok
        changed += (la.get("category"), la.get("severity")) != (ra.get("category"), ra.get("severity"))
    baseline = summarize(left, refs)
    candidate = summarize(right, refs)
    result = {
        "status": "FULL_PAIRED_TEST_COMPLETE_WEAK_REFERENCE_METRICS",
        "count_per_role": expected,
        "baseline_current_sft": baseline,
        "candidate_grpo53": candidate,
        "delta_candidate_minus_sft": {k: candidate[k] - baseline[k] for k in baseline if isinstance(baseline[k], float)},
        "paired_changes": {"changed_category_or_severity": changed, "improved_vs_weak_reference": improved,
                           "worsened_vs_weak_reference": worsened},
        "same_initial_inputs_verified": True,
        "test_labels_hidden_from_models": True,
        "optimizer_steps_during_test": 0,
        "weak_reference_only_not_independent_gold": True,
    }
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
