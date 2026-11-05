# GRPO v2 completion audit

Audit date: 2026-09-08 UTC

| Requirement | Authoritative evidence | Result |
|---|---|---|
| Multi-stage real-tool rollout | `../grpo_v2/iteration_002/rollout_audit.json` | PASS: 8,192 trajectories, 2,048 four-rollout groups, 216 model temporal calls, 0 framework temporal calls |
| Smoke audit and training | `../grpo_v2/iteration_002/smoke64/rollout_audit.json`, `smoke64/update_audit.json` | PASS |
| Formal GRPO update | `../grpo_v2/iteration_002/update_audit.json` | PASS: 42/42 optimizer steps |
| Failure recovery and resume | formal `rollout_failures.jsonl`, final `rollouts.jsonl`, and Dev parent `retry.jsonl` | PASS: all 7 unique rollout runtime failures and the one Dev retry were recovered; final result sets contain no runtime errors |
| Frozen Dev64 selection | `../grpo_v2/iteration_002/dev64_composite/selection.json` | PASS: checkpoint-25 selected |
| Exact frozen Dev1679 evaluation | `../grpo_v2/iteration_002/full_dev1679_composite/acceptance.json` | PASS: 1,679 unique parent and candidate results |
| Safety/quality acceptance | same acceptance file | PASS: every hard gate true and meaningful reward/strict improvements true |
| Test isolation | acceptance partition evidence and `SPLIT_LOCK.json` | PASS: zero Test-ID intersection; Test examples not evaluated |
| Reproducible checkpoint | `BEST_GRPO_V2.json` and the three adapter directories | PASS |
| Artifact integrity | 58 hashes in `BEST_GRPO_V2.json` independently recomputed | PASS: 0 missing, 0 mismatch |
| Adapter integrity | safetensors load and source/frozen SHA comparison | PASS: three adapters, 496 tensors each, rank 16/alpha 32 |
| Regression suite | `python -m pytest video_detection_agent/tests -q` | PASS: 99/99 |
| Independent post-freeze verification | `POST_FREEZE_VERIFICATION.json` | PASS |

The accepted model is a three-adapter Agent bundle, not a single merged
checkpoint. Reproduction therefore requires the exact base model plus the
initial, temporal, and final-answer adapters documented in `BEST_GRPO_V2.json`.

Known measured trade-off: category accuracy, Macro-F1, strict accuracy,
Harmless false-positive rate, evidence utilization, frame cost, and reward all
improved. Harmful misses changed from 32/1,045 to 34/1,045 (+0.1914 percentage
points), which passed the pre-frozen +0.3-point non-regression gate but should
remain visible in any paper or deployment report.
