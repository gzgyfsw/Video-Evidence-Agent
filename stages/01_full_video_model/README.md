# Stage 01 — Full-video model

## Definition

Each full video is treated as one multimodal sample. The model receives the
video and a fixed classification prompt, then produces one answer. There is no
external evidence tool, observation ledger, or Agent loop.

```text
full video -> one VLM call -> category
```

## Preserved scripts

- `scripts/evaluate_all_videos.py`: early whole-video evaluation entrypoint.
- `scripts/sft_train_qwen3vl_4b.py`: early full-video supervised training
  entrypoint.

These are historical snapshots with their original path assumptions. They are
preserved to show the first mode, not presented as the current runnable stack.

## Why the project moved on

A single full-video call does not expose which moments support the decision,
does not let the model request targeted evidence, and makes observation cost
hard to control. This limitation motivates Stage 02.

