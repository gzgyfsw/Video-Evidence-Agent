from __future__ import annotations

import os

from .types import ClassificationResult, VideoFrame
from .actions import ActionType, AgentAction
from .moderation_policy import (
    LEGACY_POLICY_VERSION, CATEGORY_POLICY_VERSION, CATEGORY_DISPLAY_NAMES,
    build_safety_policy, policy_sha256, load_legacy_safety_policy,
)


CATEGORIES = (
    "Harmless",
    "Sexual Content",
    "Violence & Dangerous",
    "Sensitive Figures & Ideologies",
)


def state_for_visual_review(state: dict) -> dict:
    """Hide retrieval hypotheses and scores during post-retrieval visual review.

    The controller still supplies the inspected category and any validation
    feedback, but the model must judge the candidate frames without seeing the
    query that retrieved them or their similarity scores.
    """
    if state.get("phase") != "risk_inspected":
        return state
    clean = {key: value for key, value in state.items() if key != "tool_history"}
    clean["review_feedback"] = [
        row["rejected"] for row in state.get("tool_history", []) if "rejected" in row
    ][-3:]
    return clean

# Existing checkpoints and datasets learned the former answer string. Keep its
# verbalizer during migration while exposing only the broader canonical label
# to the agent. A newly trained checkpoint can remove this compatibility map.
CATEGORY_VERBALIZERS = {
    "Sensitive Figures & Ideologies": "Sensitive Public Figure",
}


def canonical_category(value: str | None) -> str | None:
    """Normalize common controller aliases to the four project categories."""
    if value is None:
        return None
    text = str(value).strip()
    canonical = {category.lower(): category for category in CATEGORIES}
    aliases = {
        "harmless-l0": "Harmless",
        "harmless l0": "Harmless",
        "sexual": "Sexual Content",
        "violence": "Violence & Dangerous",
        "sensitive": "Sensitive Figures & Ideologies",
        "sensitive public figure": "Sensitive Figures & Ideologies",
        "sensitive topics & public figures": "Sensitive Figures & Ideologies",
    }
    return aliases.get(text.lower(), canonical.get(text.lower(), text))

LEGACY_SAFETY_POLICY = load_legacy_safety_policy()

POLICY_VERSION = os.environ.get("VIDEO_SAFETY_POLICY_VERSION", LEGACY_POLICY_VERSION)
SAFETY_POLICY = build_safety_policy(LEGACY_SAFETY_POLICY, POLICY_VERSION)
SAFETY_POLICY_SHA256 = policy_sha256(SAFETY_POLICY)

LEGACY_CATEGORY_SUMMARY = """Harmless - 不含明确色情、暴力/危险、邪教极端意识形态、知名政治人物或种族歧视风险。
Sexual Content - 性暗示、性化姿势、裸体、性行为或露骨身体暴露。
Violence & Dangerous - 打斗、武器、伤害、血腥、自伤或危险违法行为。
Sensitive Figures & Ideologies - 邪教或极端组织/符号/宣传，少量高度知名政治人物，或针对种族、族群、宗教、国籍、性别及性取向的身份歧视内容。"""

CATEGORY_SUMMARY = LEGACY_CATEGORY_SUMMARY if POLICY_VERSION == LEGACY_POLICY_VERSION else """Harmless - 完成规定范围检查后，未发现定义内三类风险且没有未解决疑点。
Sexual Content - 色情与有依据的性化呈现；普通服装、赤膊、身材本身不充分。
Violence & Dangerous - 攻击、伤害、威胁或明确人身危险；疑似物品和违法名称本身不充分。
Sensitive Figures & Ideologies - 指定敏感人物、极端组织／宣传及身份仇恨；普通辱骂和外貌猜测不充分。"""

PROMPT = f"""这个视频属于什么类别？请从以下四个一级类别中选择：

{CATEGORY_SUMMARY}

直接色情、暴力或危险证据优先于敏感公共背景。请严格输出：
<answer>类别名称（英文）</answer>

{SAFETY_POLICY}"""


class QwenTokenLikelihoodClassifier:
    """Four-way teacher-forced scorer matching the repository's confidence definition."""

    def __init__(self, model_path: str, max_pixels: int = 384 * 384):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.max_pixels = max_pixels

    def _prepared_images(self, frames: list[VideoFrame]) -> list:
        """Actually enforce the visual pixel budget before processor tokenization."""
        import math

        images = []
        for frame in frames:
            image = frame.image
            width, height = image.size
            if width * height > self.max_pixels:
                scale = math.sqrt(self.max_pixels / (width * height))
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale)))
                )
            images.append(image)
        return images

    def classify(self, frames: list[VideoFrame]) -> ClassificationResult:
        if not frames:
            raise ValueError("at least one frame is required")
        images = self._prepared_images(frames)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": PROMPT})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]
        scores = []
        with self.torch.inference_mode():
            for category in CATEGORIES:
                verbalizer = CATEGORY_VERBALIZERS.get(category, category)
                inputs = self.processor(
                    text=[prompt + f" <answer>{verbalizer}</answer>"], images=images,
                    padding=True, return_tensors="pt",
                ).to(self.model.device)
                logits = self.model(**inputs).logits[:, :-1, :].float()
                token_ids = inputs.input_ids[:, 1:]
                log_probs = logits.log_softmax(dim=-1).gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
                scores.append(float(log_probs[:, prompt_len - 1:].mean()))
                del inputs, logits, token_ids, log_probs
        probabilities = self.torch.softmax(self.torch.tensor(scores, dtype=self.torch.float64), dim=0).tolist()
        pred = CATEGORIES[max(range(len(scores)), key=scores.__getitem__)]
        return ClassificationResult(pred, dict(zip(CATEGORIES, probabilities)))

    def decide(self, frames: list[VideoFrame], state: dict,
               allowed_actions: list[ActionType]) -> AgentAction:
        """Choose an action from the actual visible frames, never from confidence."""
        import json
        import re

        allowed = [item.value for item in allowed_actions]
        schema = {
            "action": "one of " + ", ".join(allowed),
            "category": "required only for ANSWER",
            "severity": "required for ANSWER; L0, L1, or L2. Harmless requires L0; harmful categories require L1-L2",
            "risk_type": f"required for SEMANTIC_LOCATE and MARK_NO_EVIDENCE; one of {list(CATEGORIES[1:])}",
            "num_frames": "number of NEW frames, required for SAMPLE_GLOBAL or TEMPORAL_EXPAND",
            "query": "list of 1-4 short atomic English visual phrases, required only for SEMANTIC_LOCATE",
            "timestamps": "list of selected locator timestamps, optional for TEMPORAL_EXPAND",
            "window": "seconds from 0.5 to 15, required only for TEMPORAL_EXPAND",
            "rationale": "one short Chinese sentence explaining the decision",
            "evidence": [{"timestamp": "one exact value from State.frame_timestamps",
                          "description": "at most 30 Chinese characters describing only visible evidence"}],
            "evidence_strength": "required for harmful ANSWER; clear or ambiguous",
            "evidence_basis": "required for Sexual Content ANSWER; one of revealing_attire, nudity, sexual_action, provocative_pose, intimate_area_emphasis, fetishized_presentation. revealing_attire is valid only for L1",
            "identified_figure": "required only when Sensitive answer is based on a named prominent political figure; exact person name",
        }
        prompt_state = state_for_visual_review(state)
        prompt = f"""You are the visual controller of a video-safety agent. You can directly see all currently sampled frames, ordered by frame_timestamps.
Choose exactly one allowed action from visible evidence, temporal coverage, and tool observations. No confidence score is available or needed.
Allowed actions now is a hard constraint. Never output an action absent from Allowed actions now.
You are the decision-making brain; Python only executes tools, tracks state, validates evidence, and enforces budgets. If State.required_risk_type is not null, output SEMANTIC_LOCATE for exactly that risk and generate its scene-specific query; do not choose another risk or ANSWER.
Use SAMPLE_GLOBAL to increase whole-video coverage when the current temporal stride is too sparse but no specific evidence query is justified.
Use SEMANTIC_LOCATE when sampled frames may have missed brief, semantically describable evidence. Generate 1-4 atomic positive visual phrases, each directly depicting visible evidence of State.required_risk_type. Every phrase must contain a concrete risk action, object, body exposure, symbol, or identifiable figure—not merely people, clothing, proximity, setting, or ordinary scene context. Split temporal sequences such as approaching, pushing, and striking into separate phrases, but omit non-risk steps such as approaching unless they visibly contain risk evidence. Never generate benign explanations, negative phrases containing no/not/without, questions, instructions such as "check whether", or alternatives such as "interview", "conversation", and "neutral interaction". SigLIP similarity is only retrieval ranking and is never proof that a risk exists. Locator timestamps in tool history are search leads, not visible evidence; cite one only when it also appears in State.frame_timestamps after visual inspection.
When State.phase is located, SigLIP has returned candidate timestamps but those candidates are not yet visible evidence. MARK_NO_EVIDENCE is forbidden in this phase. ANSWER(Harmless) is also unavailable. If the already visible frames independently contain clear harmful evidence, ANSWER that harmful category; otherwise output TEMPORAL_EXPAND. Select candidate timestamps from State.locator_candidates, choose the smallest useful window and num_frames, and do not treat similarity scores as proof.
MARK_NO_EVIDENCE is valid only after SEMANTIC_LOCATE and TEMPORAL_EXPAND for that same risk. Initial frames alone cannot establish absence across the full video.
When State.phase is risk_inspected, perform an independent visual review of inspected_risk_type.
The retrieval query, matched text, and similarity scores are intentionally hidden. Judge only the
currently supplied frames. Retrieval selected candidates for recall and is not evidence that the
category is present. Describe concrete visible facts first; if those facts do not independently
satisfy the category policy, output MARK_NO_EVIDENCE for inspected_risk_type.
However, when the visible facts satisfy the Operational Severity L1 boundary for inspected_risk_type,
do not output MARK_NO_EVIDENCE: output ANSWER for that risk with severity=L1 and timestamped facts.
HARD FICTIONAL-GAME EXCLUSION: When the inspected scene is clearly stylized video-game gameplay and
the only violence-like cues are avatars, fantasy monsters, combat UI, health/damage numbers, glowing
attack effects, or ordinary virtual combat, these cues do NOT satisfy Violence & Dangerous. After
expansion you MUST output MARK_NO_EVIDENCE for Violence, not ANSWER(Violence & Dangerous). Override
this exclusion only for realistic graphic injury/gore, cruelty, or real-world dangerous instruction.
For Sensitive Figures & Ideologies, do not MARK_NO_EVIDENCE after identifying a clearly recognizable highly prominent political leader who is central and persistent. The absence of extremist or discriminatory material does not cancel that visible political-figure evidence.
For ANSWER(Sensitive Figures & Ideologies), evidence_basis is mandatory and must be one of
named_prominent_figure, extremist_content, identity_discrimination, or cult_content. When using
named_prominent_figure, identified_figure must contain the specific person's name, and the evidence
description must name that same person. Never guess identity from a suit, podium, flags, scenery,
gender, age, or vague facial resemblance. If you cannot confidently name the person, MARK_NO_EVIDENCE.
You may immediately ANSWER a harmful category when the visible frames contain clear evidence.
Every harmful ANSWER must include exactly 1 or 2 evidence items with exact timestamps from State.frame_timestamps and a concrete description of what is visibly present. L2 requires evidence_strength=clear. L1 may use evidence_strength=ambiguous only after useful semantic/temporal inspection, and must describe the visible boundary fact rather than speculation. Write rationale and evidence descriptions in concise Chinese; keep only SEMANTIC_LOCATE query phrases and category names in English. Each evidence description must be at most 30 Chinese characters. Never repeat equivalent evidence across many frames, cite an unseen timestamp, or infer intent that is not visible. Harmless must use severity=L0 and does not require positive evidence timestamps because its audit trail is the three completed risk checks.
For ANSWER(Sexual Content), evidence_basis is mandatory. A fully clothed anime/cartoon character,
ordinary advertisement illustration, female body shape, or fitted clothing alone is invalid evidence.
Use revealing_attire only with severity=L1 for persistent bikini/underwear-like presentation,
revealing mirror selfies, or comparable body-display framing without intimate-area emphasis.
Use intimate_area_emphasis only for a deliberate persistent close-up or composition centered on a
specific intimate area, not merely because the character's whole upper body is visible.
For Violence, proximity, approaching, raised arms, an ambiguous pose, or possible contact is not clear evidence. Require an unambiguous strike, push, kick, forceful restraint, weapon, injury, or a before/contact/after sequence showing visible forced displacement. Describe only observable facts. Words such as "appears", "may", "suggests", "indicates", "possibly", or "recoiling" without visible displacement mean the evidence is ambiguous. If evidence remains ambiguous after expansion, do not ANSWER the harmful category; mark that risk no_evidence.
For driving footage, inspect timestamp-ordered burst frames as a sequence. A vehicle rapidly entering the camera vehicle's path followed by extreme proximity, impact, abrupt displacement, or an immediate cut at the peak event is clear dangerous-driving evidence even if no single still frame shows the entire collision.
ANSWER may use only State.allowed_answer_categories. When Harmless is unavailable but the video appears harmless, do not propose ANSWER(Harmless): choose SEMANTIC_LOCATE for one item in State.unchecked_risks using a scene-specific visual query. Inspect retrieved temporal context, then mark that risk no_evidence only if unsupported. Harmless becomes available only after Sexual Content, Violence & Dangerous, and Sensitive Figures & Ideologies have each completed this retrieval-and-inspection check.
Choose the smallest useful num_frames. Respect frame_budget_remaining and all remaining call budgets in State.
Project safety policy:
{SAFETY_POLICY}
Categories: {list(CATEGORIES)}
Allowed actions now: {allowed}
Output exactly one JSON object and no markdown.
Schema: {json.dumps(schema)}
State: {json.dumps(prompt_state, ensure_ascii=False)}"""
        failures = []
        data = None
        for attempt in range(3):
            retry = ""
            if attempt:
                retry = ("\nYour previous response was malformed or truncated JSON. "
                         "Return a shorter valid JSON object. Keep rationale and each evidence "
                         "description concise; close every string, array, and object.")
            images = self._prepared_images(frames)
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt + retry})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.processor(
                text=[text], images=images,
                padding=True, return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                )
            output = self.processor.decode(
                generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            del generated, inputs
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            match = re.search(r"\{.*\}", output, re.DOTALL)
            try:
                if not match:
                    raise ValueError("no JSON object found")
                data = json.loads(match.group(0))
                break
            except (ValueError, json.JSONDecodeError) as exc:
                failures.append(f"attempt {attempt + 1}: {exc}; output={output!r}")
        if data is None:
            raise ValueError("policy returned invalid JSON after 3 attempts: " + " | ".join(failures))
        action = ActionType(str(data.get("action", "")).upper())
        raw_query = data.get("query")
        query = ([str(item) for item in raw_query] if isinstance(raw_query, list)
                 else raw_query)
        evidence = []
        for item in data.get("evidence", []) or []:
            if isinstance(item, dict) and item.get("timestamp") is not None:
                evidence.append({"timestamp": float(item["timestamp"]),
                                 "description": str(item.get("description", "")).strip()})
        return AgentAction(
            action=action, category=canonical_category(data.get("category")),
            severity=str(data.get("severity", "")).upper() or None,
            risk_type=canonical_category(data.get("risk_type")), query=query,
            timestamps=[float(x) for x in data.get("timestamps", [])] or None,
            window=float(data["window"]) if data.get("window") is not None else None,
            num_frames=int(data["num_frames"]) if data.get("num_frames") is not None else None,
            rationale=data.get("rationale"),
            evidence=evidence or None,
            evidence_strength=str(data.get("evidence_strength", "")).lower() or None,
            evidence_basis=str(data.get("evidence_basis", "")).lower() or None,
            identified_figure=str(data.get("identified_figure", "")).strip() or None,
        )
