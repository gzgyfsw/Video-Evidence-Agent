"""Draft terminal reward: accurate L2, useful REVIEW, then visual-input cost.

No inference or optimizer. References are weak Train labels. The cost scale is
soft and never restricts acquisition, decisions, or tool calls.
"""
import math

VERSION = "outcome-cost-v1-draft"
LABELS = {"Harmless", "Sexual Content", "Violence & Dangerous",
          "Sensitive Figures & Ideologies"}
LEVELS = {"L0", "L1", "L2"}


def reward(*, reference_category, reference_severity, predicted_category,
           predicted_severity, visual_inputs, boundary=False,
           infrastructure_error=False, valid=True):
    if infrastructure_error:
        return {"status": "HOLD", "total": None, "reason": "INFRASTRUCTURE_ERROR"}
    if reference_category not in LABELS or reference_severity not in LEVELS:
        return {"status": "HOLD", "total": None, "reason": "UNKNOWN_REFERENCE"}
    if (reference_category == "Harmless") != (reference_severity == "L0"):
        return {"status": "HOLD", "total": None, "reason": "REFERENCE_CONFLICT"}
    if boundary and reference_category == "Harmless":
        return {"status": "HOLD", "total": None, "reason": "BOUNDARY_REFERENCE_CONFLICT"}
    if not valid or predicted_category not in LABELS or predicted_severity not in LEVELS:
        return {"status": "SCORED", "task": -1., "cost": 0., "total": -1.,
                "reason": "INVALID_OUTPUT"}
    if (predicted_category == "Harmless") != (predicted_severity == "L0"):
        return {"status": "SCORED", "task": -1., "cost": 0., "total": -1.,
                "reason": "INCONSISTENT_CATEGORY_LEVEL"}
    if not isinstance(visual_inputs, int) or isinstance(visual_inputs, bool) or visual_inputs < 1:
        return {"status": "HOLD", "total": None, "reason": "MISSING_VISUAL_ACCOUNTING"}
    correct = predicted_category == reference_category
    if boundary:
        if predicted_severity == "L1":
            task, reason = (.95 if correct else .90), "BOUNDARY_REVIEW"
        elif predicted_severity == "L2":
            task, reason = (.85 if correct else .40), "BOUNDARY_BLOCK"
        else:
            task, reason = 0., "BOUNDARY_UNSAFE_ALLOW"
    elif reference_severity == "L2":
        if predicted_severity == "L2":
            task, reason = (1. if correct else .40), "L2_BLOCK"
        elif predicted_severity == "L1":
            task, reason = .60, "L2_DEFER_TO_REVIEW"
        else:
            task, reason = 0., "L2_UNSAFE_ALLOW"
    elif reference_severity == "L1":
        if predicted_severity == "L1":
            task, reason = (.90 if correct else .85), "L1_REVIEW"
        elif predicted_severity == "L2":
            task, reason = (.50 if correct else .20), "L1_OVERBLOCK"
        else:
            task, reason = 0., "L1_UNSAFE_ALLOW"
    else:
        task = 1. if predicted_severity == "L0" else (.30 if predicted_severity == "L1" else 0.)
        reason = "L0_ALLOW" if predicted_severity == "L0" else "L0_FALSE_ALARM"
    # All model-visible image occurrences, including repeated images and initial
    # frames. This measures frame presentations, not visual tokens or KV work.
    eligible = correct and (boundary and predicted_severity in {"L1", "L2"}
                            or not boundary and predicted_severity == reference_severity)
    cost = .02 * visual_inputs / (visual_inputs + 16.) if eligible else 0.
    total = task - cost
    assert math.isfinite(total)
    return {"status": "SCORED", "task": task, "cost": cost, "total": total,
            "reason": reason, "category_correct": correct,
            "cost_eligible": eligible, "visual_input_occurrences": visual_inputs}
