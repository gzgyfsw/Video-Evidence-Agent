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

## Four-stage timeline

| Stage | Period | Control structure | Main preserved evidence | Claim boundary |
|---|---|---|---|---|
| 01: full-video model | 2026-06 to 2026-07 | A whole video is one sample; one model call returns the class | Early Qwen-VL evaluation and SFT entrypoints | Non-Agent baseline. Early small or weak-label evaluations are motivation, not the current result. |
| 02: tool workflow | 2026-08 to early 2026-09 | Real semantic and temporal tools are scheduled mainly by a Python workflow and guards | Tool receipts, deterministic workflow, Teacher pilots | Real tool use is established, but workflow-scheduled calls are not model autonomy. |
| 03: autonomous Agent loop | 2026-09 to 2026-10-13 | The model repeatedly chooses a tool, continuation, or terminal answer | Routed SFT/GRPO, legacy closeout on 1,679 internal examples, THVL 448 comparison | Framework fallbacks remain separate from model actions; internal and THVL sets have explicit interpretation limits. |
| 04: policy-configurable evidence Agent | 2026-10-17 onward | One Student controls evidence acquisition and stopping; an external versioned policy defines the decision boundary | 1,743-video SFT, 2,829 states, 400 steps; outcome-cost GRPO with 938 groups, 1,688 selected trajectories, 53 steps; paired Dev 1,915 and Test 1,924 | Policy replacement is structurally supported. Unseen-policy performance remains an empirical claim requiring a frozen cross-policy evaluation. |

Historical V0–V4.3 names remain in run directories and reports. They map into
the four stages rather than defining nine separate paper contributions:

- V0/V1 -> Stage 01;
- V2 and early guarded prototypes -> Stage 02;
- routed V3, multi-loop work, and V3.5 closeout -> Stage 03;
- unified Student V4, GRPO V4.1/V4.2, and paired evaluation V4.3 -> Stage 04.

## Lineage graph

```text
01 whole-video sample / one-shot prediction
  -> 02 deterministic tool workflow
      -> 03 autonomous multi-turn Agent loop
          -> 04 policy-configurable evidence Agent
              -> unified SFT
              -> outcome-cost GRPO
              -> paired Dev/Test evaluation
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

## Git tags

The retrospective tags identify preserved checkpoints in the imported history:
`agent-v3-legacy-retrospective`, `agent-v4-sft-retrospective`,
`agent-v4.2-grpo-retrospective`, `agent-v4.3-eval-retrospective`, and
`curated-agent-history-20261105`. Future promoted versions should receive a new
tag only after model binding, dataset identity, evaluation protocol, and the
acceptance decision are committed.

