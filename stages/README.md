# Four research stages

The paper story is organized by **control structure**, not by every checkpoint
or training suffix. V0/V1/V2-style experiment names remain in historical
records, but the four stages below are the stable conceptual units.

| Property | 01 Full-video model | 02 Tool-using workflow Agent | 03 Autonomous Agent loop | 04 Policy-configurable Agent |
|---|---|---|---|---|
| Model input | Whole video as one sample | Initial frames plus workflow-selected evidence | Cumulative visible evidence and tool history | Evidence ledger plus an explicit policy version |
| Tool execution | None | Real semantic and temporal tools | Real tools selected inside a multi-turn loop | Same loop with frozen receipts and replay |
| Decision owner | Single model call | Python stages / guards dominate routing | Model chooses continue, tool, or stop | Unified Student chooses evidence actions; policy maps evidence to decisions |
| Stop condition | End of generation | Fixed stage, guard, or budget | Model `ANSWER`, otherwise continue until explicit historical budget | Model `ANSWER`; new runs may be uncapped while recording actual cost |
| Policy coupling | Labels embedded in prompt | Policy embedded in workflow prompt | Policy still coupled to Agent output contract | External, versioned policy file and hash |
| Paper role | Baseline and motivation | Tool availability is not yet autonomy | Core active-perception contribution | Current generalization-oriented system |

## Repository rule

- `scripts/` inside a stage contains only stage-specific historical
  entrypoints or snapshots.
- Shared code is canonical under `../video_detection_agent/`; it is not copied
  into every stage.
- `source_manifest.json` records the exact canonical files associated with each
  mode.
- Frozen outputs stay at their original paths so recorded configurations and
  hashes remain resolvable.

Stage 04's policy configuration is implemented. Claims that one frozen model
generalizes to arbitrary unseen policies require a separate cross-policy
experiment and are not inferred from code modularity alone.

