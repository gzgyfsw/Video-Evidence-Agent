# Adaptive Video Evidence Agent

This repository preserves the research lineage of an adaptive video-safety
Agent. It is a curated experiment record, not a mirror of the former
MM-EUREKA-Qwen workspace.

The central idea is to replace fixed-frame classification with an Agent that
can inspect initial frames, request semantic or temporal evidence, and stop
only when the evidence is sufficient for a structured safety decision.

## What is preserved

- `video_detection_agent/agent/`: the tool-using Agent foundation.
- `video_detection_agent/docs/`: experiment contracts and the historical log.
- `video_detection_agent/outputs/`: only small authoritative reports,
  manifests, run configurations, and frozen source snapshots.
- `docs/agent_versions/`: the retrospective version timeline and artifact
  checksums.
- `paper/`: paper-facing notes created for the Agent line (when present).

Model weights, videos, datasets, caches, generated rollouts, optimizer states,
and bulk logs are intentionally excluded. Their original local artifacts are
not deleted; authoritative files are referenced by path and digest.

## Retrospective version history

The Git commits and tags in this repository were created retrospectively on
2026-11-05 from frozen records. They organize real experiment versions but do
not pretend that Git commits existed on the original run dates.

| Line | Meaning |
|---|---|
| V2/V3 | Real-tool and routed multi-stage Agent, followed by the frozen legacy closeout. |
| V4 | Unified Student trained from the base model with supervised Agent states. |
| V4.1 | Bounded GRPO feasibility and credit-assignment checks; not promoted. |
| V4.2 | Outcome-first, observation-cost GRPO candidate; 53 optimizer steps. |
| V4.3 | Paired current SFT-vs-GRPO evaluation with frozen runtime bindings. |

Read [`docs/agent_versions/README.md`](docs/agent_versions/README.md) for the
full lineage and claim boundaries.

## Scope boundary

This repository deliberately excludes the old MM-EUREKA/OpenRLHF framework,
legacy multimodal math evaluation code, OpenSpec process files, unrelated data
downloaders, and the early fixed-frame paper draft. Those materials were not
the implementation of the final Agent.

