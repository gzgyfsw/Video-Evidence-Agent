# GRPO v1 frozen Dev acceptance

GRPO candidates are compared with the accepted SFT-v5 full-Agent baseline on
the same frozen 1,679-video Dev split. Test remains sealed during all tuning.

## Hard safety and behavior gates

- Structural trajectory audit passes with no missing/error rows.
- Initial structured-action validity is at least 98%.
- Category accuracy and Macro-F1 may not fall by more than 0.3 percentage point.
- Category+severity accuracy may not fall by more than 0.5 percentage point.
- Harmful-to-Harmless miss rate may not rise by more than 0.3 percentage point.
- Harmless false-positive rate may not rise by more than 1 percentage point.
- Direct first-step tool rate remains at most 10%.
- Tool-evidence utilization remains at least 80%.
- Mean total frames may not exceed the SFT baseline by more than 0.5 frame.

## Selection gate

In addition to every hard gate, mean frozen GRPO reward must improve by at
least 0.003 and at least one interpretable Dev metric must improve: category or
strict accuracy by 0.2 point, harmful miss by 0.2 point, necessary-tool recall
by 1 point, or mean frames by 0.1.

## Composite evaluation protocol

GRPO v1 updates only the initial routing decision. The selected GRPO adapter
chooses `ANSWER` versus `JOINT_SEMANTIC_SCAN`; the frozen SFT-v5 adapter
generates scan queries and performs every post-tool visual judgement. Route
changes are counted by the branch that the safety framework actually executes,
not merely by the raw action string. In particular, an initial L1, Harmless,
ungrounded, or structurally invalid `ANSWER` already maps to the protected scan
branch and is not a new tool call.

Frozen-Dev evaluation is paired and counterfactual: unchanged branches reuse
the previously audited SFT-v5 result; `DIRECT -> SCAN` changes execute real
SigLIP retrieval/frame expansion and frozen SFT-v5 post-tool inference; valid
new direct answers use the candidate's cached initial result. The branch
provenance audit must pass with zero violations, all 1,679 Dev IDs, no errors,
and no Train/Test overlap. The final frozen artifact therefore contains both a
router adapter and its immutable post-tool SFT-v5 adapter.

The frozen reward does not currently award autonomous `TEMPORAL_EXPAND`: the
current runner schedules this Python action after a model assessment. It may be
enabled only after model-origin provenance is implemented and audited.
