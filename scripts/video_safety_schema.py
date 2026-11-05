"""Shared label schema for video-safety SFT and reinforcement learning."""

# 一级类别（训练用，4类）
OFFICIAL_CATEGORIES = [
    "Harmless",
    "Sexual Content",
    "Violence & Dangerous",
    "Sensitive Figures & Ideologies",
]

CATEGORY_ALIASES = {
    # 色情旧标签 → Sexual Content
    "Borderline Sexual Content": "Sexual Content",
    "Explicit Sexual Content": "Sexual Content",
    "Sexual Conetent": "Sexual Content",
    "Borderline Pornography": "Sexual Content",
    "Pornography": "Sexual Content",
    # 暴力类旧别名
    "Sentive Figure": "Sensitive Figures & Ideologies",
    "Sensitive Figure": "Sensitive Figures & Ideologies",
    "Sensitive Public Figure": "Sensitive Figures & Ideologies",
    "Sensitive Topics & Public Figures": "Sensitive Figures & Ideologies",
    "Violence & Gore": "Violence & Dangerous",
    "Graphic Violence": "Violence & Dangerous",
    "Dangerous Content": "Violence & Dangerous",
    "Dangerous Illegal": "Violence & Dangerous",
    "Dangerous / Illegal": "Violence & Dangerous",
    "Other Harmful Content": "Violence & Dangerous",
    "Violence & Danger": "Violence & Dangerous",
}

CATEGORY_DESCRIPTIONS = {
    "Harmless": "不含明确的色情、暴力/危险、邪教或极端意识形态、指定知名政治人物、种族或族群歧视风险的视频。包括普通生活、正常娱乐、普通体育、普通新闻和普通公众人物内容。",
    "Sexual Content": "性相关内容，涵盖从擦边到露骨的全部程度：性暗示、过度性化的着装或姿势、色情挑逗动作，以及裸体、性行为画面、生殖器官暴露等。对应 T2VSafetyBench 中的 Borderline Pornography 与 Pornography。",
    "Violence & Dangerous": "暴力与血腥内容，包括：人际暴力（打斗、武器攻击）、自伤自杀（割腕、上吊、服药）、动物虐待、血腥恶心画面（尸体、腐烂、gore）、危险违法行为（毒品、非法武器、爆炸物）等。",
    "Sensitive Figures & Ideologies": "严格包含三部分：一是邪教或极端组织及其可识别符号、仪式、招募和宣传；二是名单范围内高度知名且可识别的政治领导人或高级政治人物作为视频核心，包括围绕其展开的宣传或操纵性合成；三是针对种族、族群、国籍、宗教、性别或性取向等身份群体的歧视、非人化、隔离主张、歧视性符号或可见文字。普通政治人物、普通公众人物、常规政治新闻、一般骚扰和一般虚假信息不自动构成本类。明确色情或暴力危险内容仍优先归入对应直接风险类别。",
}


def normalize_category(label: str) -> str:
    """Return the canonical four-class label, preserving unknown labels."""
    label = label.strip()
    return CATEGORY_ALIASES.get(label, label)


def build_category_prompt(target_mode: str = "legacy_reason_confidence") -> str:
    lines = ["这个视频属于什么类别？请从以下四个一级类别中选择：", ""]
    for category in OFFICIAL_CATEGORIES:
        lines.append(f"{category} - {CATEGORY_DESCRIPTIONS[category]}")
        lines.append("")
    lines.extend([
        "判别优先级：若敏感人物或意识形态背景与明确色情、暴力、伤害、爆炸、武器攻击等直接风险同时出现，按直接风险归入 Sexual Content 或 Violence & Dangerous；只有上述三类敏感证据本身是核心且不存在更直接风险时，才选 Sensitive Figures & Ideologies。",
        "普通政治新闻、一般骚扰、一般虚假信息、普通机构标志或普通公众人物出镜，不归入 Sensitive Figures & Ideologies。",
        "",
    ])
    lines.append("请严格按以下格式输出，不要添加标签外文字：")
    if target_mode == "answer_only":
        lines.append("<answer>类别名称（英文）</answer>")
    elif target_mode == "answer_or_review":
        lines.append("除四个类别外，若视频证据不足、类别边界明显、或不应自动作出四类决定，可输出 REVIEW（交人工复核）。")
        lines.append("<answer>Harmless / Sexual Content / Violence & Dangerous / Sensitive Figures & Ideologies / REVIEW</answer>")
    elif target_mode == "answer_then_reason":
        lines.append("<answer>类别名称（英文）</answer>")
        lines.append("<reason>用一句话说明视频中最关键的判断证据</reason>")
    elif target_mode == "reason_then_answer":
        lines.append("<reason>用一句话说明视频中最关键的判断证据</reason>")
        lines.append("<answer>类别名称（英文）</answer>")
    elif target_mode == "legacy_reason_confidence":
        lines.append("<reason>用一句话说明视频中最关键的判断证据</reason>")
        lines.append("<confidence>0.00-1.00</confidence>")
        lines.append("<answer>类别名称（英文）</answer>")
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return "\n".join(lines)


STRUCTURED_PROMPT = build_category_prompt()


REASON_SEEDS = {
    "Harmless": "未发现明确的色情、暴力、危险或敏感人物风险",
    "Sexual Content": "视频内容呈现性相关、色情、裸露或性暗示风险特征",
    "Violence & Dangerous": "视频内容呈现暴力、伤害、危险或违法风险特征",
    "Sensitive Figures & Ideologies": "视频内容涉及邪教或极端意识形态、指定知名政治人物或种族族群歧视",
    # 旧标签兼容
    "Borderline Sexual Content": "视频内容呈现性暗示、过度性化或色情挑逗特征",
    "Explicit Sexual Content": "视频内容呈现露骨色情、裸体或性行为画面",
    "Sentive Figure": "视频内容涉及敏感公众人物或相关公共事件",
    "Sensitive Public Figure": "视频内容涉及敏感公众人物或相关公共事件",
}


def build_sft_target(label: str, confidence: float = 0.80) -> str:
    """Build a format-seeding target when per-video evidence is unavailable."""
    label = normalize_category(label)
    reason = REASON_SEEDS.get(label, "视频内容呈现需要进一步审核的风险特征")
    return f"<reason>{reason}</reason>\n<confidence>{confidence:.2f}</confidence>\n<answer>{label}</answer>"
