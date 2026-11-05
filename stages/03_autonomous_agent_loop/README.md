# Stage 03 — Autonomous Agent loop

## Definition

The model receives accumulated observations and prior tool receipts in a
multi-turn loop. At each turn it can request semantic evidence, request
temporal expansion, or stop with a structured answer.

```text
observe -> model action -> real tool -> append receipt -> model action -> ...
                                                         \-> ANSWER / stop
```

The important change from Stage 02 is **control ownership**: tool choice and
stopping become policy actions rather than a fixed workflow schedule. Framework
fallbacks remain logged and are not counted as model autonomy.

## Preserved scripts

- `scripts/evaluate_multistage_video_agent.py`: model-owned retrieval decisions
  in the multi-stage protocol.
- `scripts/evaluate_causal_video_agent_multiloop.py`: cumulative conversation
  with repeatable tool decisions.

Historical frozen results are indexed in
[`docs/HISTORICAL_EXPERIMENT_INDEX_20261102.md`](../../video_detection_agent/docs/HISTORICAL_EXPERIMENT_INDEX_20261102.md).

Stage 03 still couples much of the risk policy to the Agent prompt and output
contract. Stage 04 externalizes and versions that policy boundary.

