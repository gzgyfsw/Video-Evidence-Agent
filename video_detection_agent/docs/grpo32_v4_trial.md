# v4 unified Student: bounded GRPO feasibility trial

This is a training/development experiment, not a final paper evaluation.
The original diagnostic run lives at outputs/grpo32_v4_trial_20261019_v1/PLAN.json.
The current run is outputs/grpo32_v4_trial_20261019_v2_rng_independent/PLAN.json.
The original run's SUCCESSOR_TRIAL.json points to this revision.

## Independent RNG correction (before any optimizer update)

The first implementation used consecutive starting seeds and `base+call_index`
inside each trajectory. The actual group21 archive showed13 requests but only6
different seeds, so different trajectories shared random streams. Their numeric
replay remains a marginal-probability diagnostic, but the affected multi-request
groups do not satisfy the intended independent-sampling protocol.

The revision retains exactly the same32 videos, four starting seeds/video,
weights, policy, budgets and raw reward definition. Request index0 retains its
original seed; later requests use `base_seed + 10^12 * call_index`, with all base
seeds strictly below10^12. The sampler explicitly records this schedule and the
collector/trainer check actual saved seeds and global uniqueness before update1.
Legacy mode remains available to other existing baselines.

Six completed direct-answer groups (24 actual first requests) already satisfy
the new rule and have complete matching numeric/semantic receipts. The new
trial imports them read-only with explicit hashes; no request or verdict was
rewritten. Whole multi-request groups6,7,21 are retained as prior diagnostics
and sampled again under the fixed rule. This is protocol repair before learning,
not reward-based selection of better trajectories. All previous failures and
raw answers stay in their original directories. The optimizer update count is0.

- 32 original Train source groups, 4 independent seeded samples each. Never
  force different routes, reject identical samples, or invent score differences.
- One v4 adapter across all stages. Initial4 frames, maximum3 tool calls,
  maximum12 delivered frames, maximum4 decisions. Search delivers at most4
  new frames per call; after the first search at least4 frames remain available.
  The Student may spend those frames on another search. No forced expansion.
- Candidate timestamps remain hints; only actually delivered JPEGs count as
  visual evidence. All actor prompts omit source paths and audit metadata.
- Semantic acquisition credit: .5 for at least one actual viewed image whose
  visible content is supported as policy risk by isolated visual review.
- Report credit: additional .5 only when actual final citation, description,
  policy category/severity/disposition and historical stopping basis are
  supported. Acquiring the Teacher's timestamp alone earns no semantic credit.
- Review newly found risk even outside Teacher anchors. Archive raw image
  observations before the separate report assessment; use only the relevant
  historical prefix. Future images never justify an earlier stop.
- Cost .01 per actual tool and .001 per newly delivered image; confirmed
  unseen citation/out-of-range/unexecutable action and confirmed fabrication
  each incur -1. Unknown semantic judgments do not count as contradictions.
  Technical model/tool/reviewer failure stops the pipeline before an update;
  budget-unresolved trajectories remain unresolved, never Harmless/success.
- No learning from cost differences alone without verified semantic reward
  discrimination. Zero-variance groups are preserved with zero advantage.
- Before update1: actual v4 rollout, validated semantic reward, each request's
  original sampler-vs-full-causal replay (mean free-token error<=.05,
  max error<=.2), finite nonzero LoRA gradients. Bind template, image tensors,
  temperature, schema/masks and actual completion IDs; no old adapter waiver.
- At most8 optimizer steps, learning rate1e-6, clipping .2, reference KL .01.
  Step1 is assessed before further steps. New behavior samples must be bound
  to their actual generating checkpoint; never relabel old samples on-policy.
- Compare fixed development sources against unchanged v4. Preserve the
  development denominator, failures and unresolved cases. Risk-positive Train
  cases cannot establish harmless false-positive rate. Report contradicted
  visual claims separately from a true negative-video false-positive rate.
- Stop after this bound even if results disappoint; no automatic extra steps.

Implementation status: frozen selection, real sampling, bounded search,
resumable collection and independent semantic scoring are implemented.
`train_grpo32_semantic.py` now provides a separate bounded optimizer; its
bookkeeping tests pass, but no optimizer step has yet run in this trial.
Full128 collection, actual multi-step training and final development evaluation
remain unverified. The
historical one-update timestamp-proxy optimizer must not be used as the semantic
reward implementation for this trial.

## First zero-update result

Group00 completed four independent v4 samples (669 completion tokens), all
direct answers. No route was forced and no tool has yet been demonstrated by
this group. Full-causal train-mode replay passed the frozen probability limits
on two requests and failed on two (maximum errors .236068 and .274155 vs .2).
All four had finite nonzero LoRA gradients; base gradients were absent and
parameters were unchanged. This is a **failed zero-update gate**, not an
authorization to update or to relax the thresholds. Cached evaluation replay
is a separate diagnostic; its result cannot substitute for train-mode scoring.

Cached evaluation replay of all four original requests subsequently matched
their saved behavior probabilities exactly. A controlled read-only diagnostic
on the worst request reduced cached/full error from .274155 to .084706 using
an FP32 output-head projection, with the transformer still BF16. The alternate
diagnostic probabilities were never substituted for sampled probabilities.
NUMERICAL_REVISION.json records the explicit next behavior computation; actual
new sampling and independent backward checks remain required before training.
The first two original reports also completed isolated weak visual review;
these semantic results do not waive the failed probability gate.

The FP32-head actual group00 passed all four gradient/probability checks
(worst maximum error .1093). Its semantic acquisition/report reward components
were frozen as [.5,1,1,.5], with no confirmed fabrication penalty; an isolated
counterevidence check found insufficient evidence rather than a contradiction.
The subsequent real-tool group21 executed SCAN/EXPAND and completed13 model
requests, but one failed at .2694. Collection stopped with zero updates.

Full-FP32 forward on that same retained-token diagnostic reduced maximum
cached/full error to .023304. NUMERICAL_REVISION_fp32.json declares the new
computation contract. Groups21 and0 are now sampled again in separate *_fp32
directories with unchanged source videos/seeds and thresholds. Earlier
diagnostic collections, rewards and failures stay intact; none have trained a
model. New actual sampling and independent gradients must pass before reuse.

The new full-FP32 group21 (real SCAN/EXPAND) and group00 both passed all actual
probability/gradient requests with zero parameter changes. The remaining30
frozen groups are being collected. Isolated reviews remain a separate gate.

The group21 counterevidence follow-up stopped once because it reported only
the cited image ID despite receiving12 images. The original failed call stays
in `_contradictions/`. Collection scheduling was explicitly paused with
SIGSTOP (current bounded child allowed to finish), then resumed only after a
new isolated v2 review satisfied whole-image coverage. The two already valid,
exactly hash-bound responses were reused; the failed response was not accepted
or auto-filled. All three accepted counterevidence judgments were INSUFFICIENT,
so none yielded a fabrication penalty or promoted the original report.
Group21 acquired no supported risk evidence and is retained without updates:
its different costs alone do not create an eligible learning group.

## Bounded optimizer protocol

- Require all32 groups/128 samples and every actual request's unchanged
  numeric/gradient gate, completed actual-image weak reviews and trace-bound
  semantic scores before update1. Prepare-only is the default CLI behavior.
- Use one pass over this frozen v4 behavior batch, in stable source-hash order,
  at most4 informative video groups per minibatch and at most8 updates.
  Later minibatches retain the original v4 behavior denominator with clipping;
  they are explicitly **not fresh on-policy samples** from the updated model.
- Zero-variance groups remain in the report without updates. Cost-only groups
  with unverified acquisition/report are excluded. Genuine semantic/confirmed
  penalty differences, or cost differences among fully supported reports, may
  yield group-centered advantages with a fixed standard-deviation floor0.5.
  This prevents a cost-only .01 difference from receiving the same unit scale
  as a .5 evidence gain. Identical rewards still have no learning signal;
  actual reward values are unchanged. This protection was added before any
  optimizer update, recorded separately in OPTIMIZER_PROTOCOL_v2.json.
- All free generated tokens inside each actual independent request receive
  credit. Forced grammar tokens are excluded. Average within each trajectory,
  then across trajectories. Do not concatenate separate inference requests.
- Frozen reference = the zero-update full-causal v4 scores, bound to the actual
  tokens/images/schema/temperature; original sampler scores remain the ratio
  denominator. Recheck actual initializer tensors and first-batch probabilities.
- LR1e-6, clipping .2, reference KL coefficient .01, gradient norm clip1,
  weight decay0, LoRA only. Use full-FP32 forward matching actual sampling.
- After update1 (and each later update), require finite gradients/weights,
  changed LoRA tensors, improved same-minibatch clipped surrogate and mean
  sampled reference-KL estimate<=.02 before continuing. Failure quarantines
  the child checkpoint and stops; no automatic retry/additional updates.
- Same-sample surrogate improvement is an engineering gate, never proof of
  semantic improvement or generalization. Development comparison is separate.

## Acquisition-reference validation repair

Group08 stopped when a response had a properly referenced SUPPORTED sexual
finding plus two NOT_SUPPORTED categories with empty references. The schema
and prompt required positive evidence for SUPPORTED findings; the validator
incorrectly required it for every finding. The correction still rejects unseen
IDs for every verdict and requires references, facts and policy basis for any
SUPPORTED finding. Negative/unknown empty references receive no credit and
never become harmless labels.

The exact original response was deterministically revalidated against its
unchanged prompt, image list, policy and model binding. Calls remained2; no new
model review was requested. ACQUISITION_VALIDATION_REPAIR.json records this.
The coordinated stop interrupted group09 before any trajectory completed. Its
original partial folder remains untouched. GROUP_OUTPUT_OVERRIDES.json directs
only that group's fresh, same-video/same-seed recovery to a new folder; the
collector, auditor, progress monitor and trainer use the same explicit map.
Training hashes this map and still requires exactly the frozen128 identities.

Subsequent report review uses the Skill's unchanged full-core parallel runner:
4 isolated workers,20 calls per trajectory (maximum80 per group), timeout180s,
and fail-fast. Original prefix ordering and all audit tasks remain intact.
Cancellation cleanup is limited to reviewer argv under this exact trial root;
an offline fake-process test confirmed it leaves an unrelated peer running.

## Numerical-only recovery for group25

The original group25 four real rollouts completed, but its numerical replay
exited after model loading without a traceback or saved exit code. No optimizer
was created. Available session/user cgroup memory events show zero OOM kills;
the original exit cause remains undetermined. Preserve the original empty gate
folder and log, and never resample those four completed trajectories.

NUMERICAL_OUTPUT_OVERRIDES.json selects an explicit new replay directory for
this group only. All consumers use the same resolver; optimizer preparation
hashes the mapping and the actual successful gate. Replay thresholds, weights,
precision, original sampled tokens, images and behavior probabilities remain
unchanged. The collector now saves child return codes/signals, and the replay
writes atomic stage progress with fault-handler diagnostics. The first recovery
found an exclusive-progress-write implementation error before loading the
model; that failed attempt is retained separately and fixed before recovery2.
A recovery PASS is never inferred from file presence or an old failed gate.

## Interrupted group26 recovery

After an interrupted task turn, the collector, sampling child and review
supervisors no longer existed. Group26 had one complete real trajectory and
three unfinished/missing trajectories. The original complete sample0 was
retained at its original paths; only samples1..3 were generated with their
original fixed seeds in a separate recovery output. The terminal receipt,
request/response/mask hashes, pixels and causal histories were reconciled.
No pre-existing complete trajectory was regenerated or selected by reward.
GROUP_OUTPUT_OVERRIDES.json now resolves group26 to the combined four-sample
collection. Numerical, semantic and gradient gates remain mandatory; recovery
verification alone is not a training pass. Supervisor launches use separate
sessions with retained logs and PIDs; progress marks stale running records
when the corresponding process has actually exited.

## Policy-claim viewing subset correction

Group26's report review stopped because a policy-disposition review listed one
actually viewed supporting cited image, while the aggregate policy claim's
frame_ids held the entire eligible prefix. Independent blind review had already
covered that full prefix. The claim validation now distinguishes facts and
interpretations (must view and ground every original citation) from aggregate
category/severity/disposition claims (nonempty, actually viewed supporting cited
subset allowed). Unseen images, absent policy, factual dependencies and stopping
requirements are unchanged. This is a versioned validator correction/clarification,
not a changed content policy or automatic approval of a full report.

37 Skill tests passed; 1060 cached review objects passed deterministic offline
revalidation with no extra model calls. The failed report remains untouched.
A separate recovery copies exact task inputs, original image paths, raw model
answers and cumulative call ledgers, records original/new validator hashes, and
revalidates the one saved failed policy response. Only unfinished objects were
called again (worker call counts8→9,8→9,7→7,7→8). Copied stale writer locks were
excluded only after confirming their original PIDs had exited. One attempted
recovery stopped on such a copied lock before a new model call and is preserved.
REPORT_OUTPUT_OVERRIDES.json binds the selected report recovery to the same
actual group collection; preparation hashes it and its selected summary.

Fail-fast cancellation had interrupted group27 after three complete real
trajectories. Those were retained; only unfinished sample3 was generated with
its original seed. All four actual histories reconciled before restarting its
numerical and semantic gates. No optimizer update occurred during these repairs.
