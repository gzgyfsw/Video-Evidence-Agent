# Stage 02 — Tool workflow

## Definition

The system gains real tools—semantic risk localization and temporal evidence
expansion—but the surrounding Python workflow and safety guards still decide
much of the order and stopping behavior.

```text
initial observation -> workflow condition -> semantic/temporal tool
                    -> fixed next stage -> final decision
```

This stage proves that targeted evidence can be acquired. It must not be
described as fully autonomous tool use when a guard or workflow scheduled the
call.

## Preserved scripts and shared code

- `scripts/run_agent.py`: early rules/policy command-line entrypoint.
- `scripts/evaluate_causal_video_agent.py`: stage-wise causal evaluator.
- [`agent/core.py`](../../video_detection_agent/agent/core.py): rule-based
  tool trigger.
- [`agent/tools.py`](../../video_detection_agent/agent/tools.py): real semantic
  locator and temporal expander.
- [`agent/video.py`](../../video_detection_agent/agent/video.py): video/frame
  operations.

Stage 03 begins only when the model itself repeatedly chooses whether to call a
tool, continue, or stop.

