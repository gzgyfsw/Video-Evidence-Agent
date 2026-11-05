# Video Agent evolution and Git preservation policy

This directory records the retrospective development history of the video
safety Agent.  It was imported into Git on 2026-11-05; the commits do **not**
pretend that Git existed at the original experiment dates.  Original outputs,
failures, logs, checkpoints, and datasets remain in place and are not rewritten.

## Why artifacts are referenced instead of committed

The workspace contains hundreds of gigabytes of videos, model weights, caches,
and experiment outputs.  Git tracks source code, policies, prompts, tests,
documentation, and a small set of authoritative freeze/audit manifests.  Large
artifacts stay outside Git and are identified by path plus SHA-256 in
`ARTIFACT_MANIFEST.tsv`.

Run the verifier from the repository root:

```bash
scripts/verify_agent_version_manifest.sh
```

`PRESENT` means the file exists and its bytes match the recorded digest.
`MISSING` means the referenced local artifact is no longer available.  A missing
large artifact does not alter the Git history; it makes the preservation gap
explicit.

## Version timeline

| Version | Period | Agent design | Training/evaluation identity | Status and claim boundary |
|---|---|---|---|---|
| V0: fixed-frame classifier | 2026-06 to 2026-07 | Uniform frames, direct four-way answer | Qwen3-VL-4B SFT and early rule-reward GRPO | Historical precursor only. The 239-item evaluation was too small, reason text was templated, confidence was constant, and prompt tokens were not masked correctly. |
| V1: calibrated classifier | 2026-07 | Answer-only four-way classifier with probability loss and selective prediction | Frozen 600-item test; selected SFT reached 94.0% weak-label accuracy | Useful motivation for uncertainty and review, but not the current Agent or current paper claim. |
| V2: first tool Agent | 2026-08 | Initial frames, semantic search, temporal expansion, then answer | Teacher pilots, deterministic FSM, evidence audits, Qwen3.5 Student prototypes | Established real tool execution and evidence receipts. Many runs were probes rather than frozen models. |
| V3: routed multi-stage Agent | 2026-09 | Learned initial route plus post-tool decisions; category, severity, and action outputs | GRPO v1 routing; frozen GRPO v2 package; severity-repair GRPO v3 | V2 is a three-adapter package, not a single LoRA. Later v3/action-aware models are separate lineages and must not inherit v2 metrics. |
| V3.5: legacy closeout | 2026-10-07 to 2026-10-13 | Thick guarded Agent and matched fixed-frame/thin controls | 1,679 internal weak-reference examples and 448 THVL examples | Internal set was exposed during development. THVL interval hits verify timestamp coverage, not semantic truth. Thick/thin changes include prompt and policy, not one isolated guard. |
| V4: evidence-aligned unified Student | 2026-10-17 to 2026-10-27 | One Qwen3.5-4B policy chooses semantic scan, temporal expansion, or terminal report | 1,743 Train videos, 2,829 supervised states, 400 optimizer steps | Actual run starts from the base model (`adapter=null`), despite an earlier plan mentioning an inherited adapter. It is the current SFT baseline. |
| V4.1: bounded GRPO feasibility | 2026-10-19 to 2026-10-30 | Real tool rollouts with request-level probability replay and report/evidence reward experiments | Zero-update checks, one-update trials, fixed 120-item paired holdout | Demonstrated training mechanics and exposed reward/auditor limitations. These candidates were not promoted as final models. |
| V4.2: outcome-cost GRPO candidate | 2026-11-02 to 2026-11-04 | Reward terminal safety outcome first and apply a small observation-cost penalty; no reward merely for calling a tool | 938 groups, 3,752 attempts, 1,688 selected trajectories, 53 optimizer steps | Training completed and reload passed, but the checkpoint remains `NOT_PROMOTED` until paired evaluation is independently validated. Optimizer state was reset once on resume after step 20. |
| V4.3: paired current evaluation | from 2026-11-04 | Same Agent runtime for current SFT and the 53-step GRPO candidate | Frozen 1,915-video Dev manifest, same sources, seed schedule, budgets, tools, policy, and schema | Running at the time of the retrospective import. Dev is not Test; no improvement claim is recorded before completion. |

## Lineage graph

```text
V0 fixed-frame classifier
  -> V1 calibrated answer-only classifier
  -> V2 real-tool Agent prototype
      -> V3 routed multi-stage Agent
          -> V3.5 legacy closeout + THVL framework study
          -> data/trajectory audit and evidence alignment
              -> V4 unified Student SFT
                  -> V4.1 bounded GRPO feasibility
                  -> V4.2 outcome-cost GRPO candidate
                      -> V4.3 paired 1,915-video Dev evaluation
```

This is conceptual lineage, not checkpoint inheritance.  Checkpoint inheritance
must be read from each frozen `run_config.json`, model binding, or completion
audit.

## Source snapshots retained in Git

In addition to the live source tree, Git force-tracks the small source snapshots
saved by the following runs:

- V3 severity-repair GRPO: the actual update `run_config.json`.
- V4 unified Student SFT: trainer, field-supervision, request-content, and suffix
  trimming snapshots.
- V4.1 bounded GRPO: the independent-RNG trial plan and semantic trainer
  snapshot, plus the later one-update source bundle.
- V4.2 outcome-cost GRPO: the complete saved `source_snapshot/` bundle and
  training code snapshot.

These files intentionally remain under their original output paths so the
recorded run configurations continue to resolve them without a second naming
scheme.  Large rollout tensors, optimizer states, adapters, and media remain
excluded.

## Authoritative reading order

1. `video_detection_agent/docs/HISTORICAL_EXPERIMENT_INDEX_20261102.md`
2. `video_detection_agent/docs/DATA_AND_EXPERIMENT_LOG.md`
3. The manifests listed in `ARTIFACT_MANIFEST.tsv`

OpenSpec documents are historical references only and are not part of the
current execution or approval process.

## Git tags

The annotated tag `agent-history-baseline-20261105` identifies the first Git
snapshot of the retrospectively imported history.  Future promoted Agent
versions should receive a new annotated tag only after their model binding,
dataset identity, evaluation protocol, and acceptance decision are committed.

