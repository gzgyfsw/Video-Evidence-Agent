from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

from openai import OpenAI

from .actions import ActionType, AgentAction
from .qwen_backend import CATEGORIES, SAFETY_POLICY, canonical_category, state_for_visual_review
from .types import VideoFrame


class RemoteQwenVisualPolicy:
    """Qwen3.7-Plus visual controller used inside the real Python tool loop."""

    def __init__(self, model: str = "qwen3.7-plus", base_url: str = "https://api.302.ai/v1"):
        key = os.environ.get("API_KEY_302")
        if not key:
            env_file = Path(__file__).resolve().parents[2] / ".env.302"
            if env_file.is_file():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    name, separator, value = line.partition("=")
                    if separator and name.strip() == "API_KEY_302":
                        key = value.strip().strip('"').strip("'")
                        break
        if not key:
            raise RuntimeError("API_KEY_302 is required in the environment or .env.302")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.usage: list[dict] = []

    @staticmethod
    def _image_url(image, max_side: int = 768) -> str:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def decide(self, frames: list[VideoFrame], state: dict,
               allowed_actions: list[ActionType]) -> AgentAction:
        allowed = [action.value for action in allowed_actions]
        visible_state = state_for_visual_review(state)
        # The transcript is already represented by compact controller state.
        # Keep recent validator feedback so the teacher can correct a rejected
        # action instead of repeating it, while omitting bulky observations.
        validation_feedback = [step["rejected"] for step in state.get("tool_history", [])
                               if "rejected" in step][-3:]
        visible_state = {key: value for key, value in visible_state.items()
                         if key != "tool_history"}
        if validation_feedback:
            visible_state["validation_feedback"] = validation_feedback
        schema = {
            "action": "one of " + ", ".join(allowed),
            "category": "ANSWER only",
            "severity": "ANSWER only: L0/L1/L2",
            "risk_type": "SEMANTIC_LOCATE or MARK_NO_EVIDENCE only",
            "query": "SEMANTIC_LOCATE only: 1-4 atomic English visual phrases",
            "timestamps": "TEMPORAL_EXPAND only: timestamps from locator_candidates",
            "window": "TEMPORAL_EXPAND only: 0.5-15 seconds",
            "num_frames": "SAMPLE_GLOBAL/TEMPORAL_EXPAND only",
            "rationale": "one concise Chinese sentence",
            "evidence": [{"timestamp": "exact visible frame timestamp",
                          "description": "visible fact in concise Chinese"}],
            "evidence_strength": "harmful ANSWER only: clear or ambiguous",
            "evidence_basis": "Sexual: revealing_attire/nudity/sexual_action/provocative_pose/intimate_area_emphasis/fetishized_presentation; Sensitive: named_prominent_figure/extremist_content/identity_discrimination/cult_content",
            "identified_figure": "exact name for named political figure only",
        }
        prompt = f"""你是视频安全Teacher Agent的大脑。当前图片按State.frame_timestamps顺序排列。
你必须自主选择下一步动作；Python只执行工具、维护状态和预算。只能选择Allowed actions now中的动作。

决策流程：
- 当前可见帧有明确或达到L1边界的风险证据时，ANSWER对应风险类别。
- 覆盖稀疏且没有明确搜索目标时，可SAMPLE_GLOBAL，并选择最少够用的新帧数。
- 怀疑遗漏了可描述的短暂风险时，SEMANTIC_LOCATE；query必须是1-4条具体、正向、可见的英文风险画面描述，不能把普通场景写成风险。
- phase=located时，检索时间点只是候选而非证据。如果当前可见帧没有独立、明确的有害证据，唯一合法动作是TEMPORAL_EXPAND；此阶段绝对禁止ANSWER Harmless，也禁止用检索分数直接ANSWER风险。
- phase=risk_inspected时，必须独立看当前图片：有实际证据就ANSWER；没有达到L1就MARK_NO_EVIDENCE。
- SigLIP分数和query不是分类证据。危险ANSWER必须引用1-2个当前可见时间点和具体可见事实。
- Harmless只能L0，并且只有Python完成三类风险排查、Allowed actions允许时才能输出。
- 当画面无害但Harmless尚不可用时，必须对State.unchecked_risks中的下一类输出SEMANTIC_LOCATE；绝不能用“有害类别+L0”表示该风险不存在。
- L1表示真实可见但轻微/边界、需要人工复核；L2表示清晰风险。不要用L1代替缺少检查。
- 普通衣服、身材、动漫女性不是色情；模糊接触、靠近、抬手不是暴力；虚拟游戏普通战斗不是现实暴力；不认识具体政治人物姓名就不能按政治人物判敏感。

项目规则：
{SAFETY_POLICY}

Categories: {list(CATEGORIES)}
Allowed actions now: {allowed}
State: {json.dumps(visible_state, ensure_ascii=False)}
Schema: {json.dumps(schema, ensure_ascii=False)}
硬约束：ANSWER的category必须属于State.allowed_answer_categories。phase=located且没有当前可见的明确有害证据时必须TEMPORAL_EXPAND。
只输出一个JSON对象，不要Markdown。"""
        last_error = None
        for attempt in range(3):
            try:
                limit = len(frames) if attempt == 0 else 8 if attempt == 1 else 4
                if len(frames) <= limit:
                    request_frames = frames
                else:
                    indices = [min(int((index + 0.5) * len(frames) / limit), len(frames) - 1)
                               for index in range(limit)]
                    request_frames = [frames[index] for index in indices]
                max_side = 768 if attempt == 0 else 640 if attempt == 1 else 512
                content = [{"type": "text", "text": prompt}]
                content.extend({
                    "type": "image_url",
                    "image_url": {"url": self._image_url(frame.image, max_side)},
                } for frame in request_frames)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                )
                if response.usage:
                    self.usage.append(response.usage.model_dump())
                data = json.loads(response.choices[0].message.content)
                action = ActionType(str(data.get("action", "")).upper())
                raw_query = data.get("query")
                query = ([str(item) for item in raw_query] if isinstance(raw_query, list)
                         else raw_query)
                evidence = []
                for item in data.get("evidence", []) or []:
                    if isinstance(item, dict) and item.get("timestamp") is not None:
                        evidence.append({"timestamp": float(item["timestamp"]),
                                         "description": str(item.get("description", "")).strip()})
                print(
                    f"remote_round phase={state.get('phase')} action={action.value} "
                    f"risk={data.get('risk_type')} category={data.get('category')}",
                    flush=True,
                )
                return AgentAction(
                    action=action,
                    category=canonical_category(data.get("category")),
                    severity=str(data.get("severity", "")).upper() or None,
                    risk_type=canonical_category(data.get("risk_type")),
                    query=query,
                    timestamps=[float(value) for value in (data.get("timestamps") or [])] or None,
                    window=float(data["window"]) if data.get("window") is not None else None,
                    num_frames=int(data["num_frames"]) if data.get("num_frames") is not None else None,
                    rationale=data.get("rationale"),
                    evidence=evidence or None,
                    evidence_strength=str(data.get("evidence_strength", "")).lower() or None,
                    evidence_basis=str(data.get("evidence_basis", "")).lower() or None,
                    identified_figure=str(data.get("identified_figure", "")).strip() or None,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"remote policy failed after 3 attempts: {last_error}")
