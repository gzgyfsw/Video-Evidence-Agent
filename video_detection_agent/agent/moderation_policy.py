"""版本化分类政策；旧运行器默认仍读取冻结的旧政策。"""
import hashlib
from pathlib import Path


LEGACY_POLICY_VERSION = "legacy-safety-policy-v1"
LEGACY_POLICY_FILE = Path(__file__).resolve().parents[1] / "policies/legacy_safety_policy_v1.md"
LEGACY_POLICY_SHA256 = "45fb654deceb9edea3681dc87f2d352e164495b6a28165d84a27afd332bd6040"
CATEGORY_POLICY_VERSION = "default-category-policy-v2"
CATEGORY_POLICY_FILE = Path(__file__).resolve().parents[1] / "policies/default_category_policy_v2.md"
CATEGORY_DISPLAY_NAMES = {
    "Harmless": "当前政策范围内无害",
    "Sexual Content": "色情与性化呈现",
    "Violence & Dangerous": "暴力与危险行为",
    "Sensitive Figures & Ideologies": "敏感人物、意识形态与身份仇恨",
}


def load_legacy_safety_policy(path: Path | None = None) -> str:
    text = (path or LEGACY_POLICY_FILE).read_text(encoding="utf-8")
    if policy_sha256(text) != LEGACY_POLICY_SHA256:
        raise ValueError("Frozen legacy policy content changed; use a new policy version instead")
    return text


def build_safety_policy(legacy_policy: str, version: str) -> str:
    if version == LEGACY_POLICY_VERSION:
        return legacy_policy
    if version != CATEGORY_POLICY_VERSION:
        raise ValueError(f"Unknown video safety policy version: {version}")
    category_rules = CATEGORY_POLICY_FILE.read_text(encoding="utf-8").strip()
    # 此次只更新分类：L1/未知分离的输出合同尚未实施，不改旧等级协议。
    _, separator, severity_rules = legacy_policy.partition("Operational severity")
    if not separator:
        raise ValueError("Legacy severity contract is missing")
    return (category_rules + "\n\n运行协议说明：此兼容入口仍使用legacy-severity-v1，"
            "其中L1仍包含边界或未解决情况。它未实现risk-evidence-labels-v2的未知等级分离；"
            "不能把旧L1自动解释为确认轻微风险。\n\n"
            + separator + severity_rules)


def policy_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
