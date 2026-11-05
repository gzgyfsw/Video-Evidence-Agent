# Stage 04 — Policy-configurable evidence Agent

## Definition

The current system separates three responsibilities:

1. evidence acquisition: semantic scan and temporal expansion;
2. evidence sufficiency and stopping: the unified Student loop;
3. policy mapping: a versioned external policy maps visible facts to category,
   severity, and disposition.

```text
video -> evidence Agent -> evidence ledger / terminal report
                              + selected policy version
                              -> ALLOW / REVIEW / BLOCK
```

## Canonical implementation

- [`unified_student_loop.py`](../../video_detection_agent/unified_student_loop.py)
- [`unified_student_contract.py`](../../video_detection_agent/unified_student_contract.py)
- [`hf_unified_rollout.py`](../../video_detection_agent/hf_unified_rollout.py)
- [`agent/moderation_policy.py`](../../video_detection_agent/agent/moderation_policy.py)
- [`policies/`](../../video_detection_agent/policies/)
- [`outcome_cost_reward_v1.py`](../../video_detection_agent/outcome_cost_reward_v1.py)
- [`run_frozen_dev_manifest.py`](../../video_detection_agent/run_frozen_dev_manifest.py)
- [`run_frozen_test_manifest.py`](../../video_detection_agent/run_frozen_test_manifest.py)

## Training and evaluation identity

- Unified Student SFT: 1,743 videos, 2,829 supervised states, 400 optimizer
  steps.
- Outcome-cost GRPO candidate: 938 groups, 1,688 selected trajectories, 53
  optimizer steps.
- Paired evaluation: 1,915 Dev and 1,924 Test videos with the same sources,
  seeds, tools, budgets, processing, and schema for both roles.

## Claim boundary

Policies can be replaced and versioned without silently modifying frozen code
or historical runs. This is a policy-configurable architecture. Performance on
an unseen policy is a separate empirical generalization claim and requires its
own frozen evaluation.

