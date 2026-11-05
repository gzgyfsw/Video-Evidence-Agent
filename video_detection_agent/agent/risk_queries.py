from __future__ import annotations

import re


RISK_QUERY_BANK: dict[str, tuple[str, ...]] = {
    "Sexual Content": (
        "explicit nudity or exposed intimate body parts",
        "people engaged in a sexual act or intimate sexual contact",
        "a person making a provocative sexual pose or gesture",
        "a close-up emphasizing breasts buttocks crotch or thighs for sexual appeal",
        "a sexualized display of underwear stockings or revealing clothing",
    ),
    "Violence & Dangerous": (
        "a person punching kicking or physically attacking another person",
        "people fighting wrestling or forcefully grappling",
        "a person grabbing another person's head or neck",
        "a forceful tackle body check or aggressive physical collision",
        "a person threatening someone with a knife gun or other weapon",
        "an injured person visible blood or a serious wound",
        "a dangerous illegal act explosion self-harm or reckless driving",
    ),
    "Sensitive Figures & Ideologies": (
        "a highly prominent political leader central to the image",
        "cult or extremist organization symbols rituals or recruitment",
        "extremist political propaganda banners posters or leader portraits",
        "visible identity-based slurs or discriminatory propaganda",
    ),
}


RISK_QUERY_CUES: dict[str, tuple[str, ...]] = {
    "Sexual Content": (
        "nude", "nudity", "naked", "intimate body", "sexual", "provocative",
        "fetish", "breast", "buttock", "crotch", "thigh", "underwear",
        "stocking", "revealing clothing",
    ),
    "Violence & Dangerous": (
        "punch", "kick", "strike", "push", "fight", "attack", "aggressive",
        "forceful", "tackle", "body check", "collision", "grappl", "grab",
        "restrain", "weapon", "knife", "gun", "blood", "wound", "injur",
        "explosion", "self-harm", "reckless", "dangerous",
    ),
    "Sensitive Figures & Ideologies": (
        "political leader", "political figure", "government official", "president",
        "prime minister", "dictator", "extremist",
        "cult", "propaganda", "political symbol", "leader portrait", "slur",
        "discriminat", "dehumaniz", "hate symbol", "recruitment",
    ),
}

NEGATIVE_OR_BENIGN_PATTERNS = (
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsence\b", r"\bneutral\b",
    r"\bbenign\b", r"\bharmless\b", r"\bnon[- ]", r"\bnormal\b",
    r"\binterview\b", r"\bconversation\b", r"\bcheck (?:for|whether)\b",
)

def is_positive_risk_query(risk_type: str, query: str) -> bool:
    """Accept only positive, visibly grounded descriptions of the requested risk."""
    normalized = " ".join(query.lower().split())
    if not normalized or any(re.search(pattern, normalized) for pattern in NEGATIVE_OR_BENIGN_PATTERNS):
        return False
    return any(cue in normalized for cue in RISK_QUERY_CUES.get(risk_type, ()))


def risk_queries(risk_type: str, dynamic_query: str | list[str] | None = None) -> list[str]:
    """Combine recall-oriented project queries with one scene-specific Agent query."""
    queries = list(RISK_QUERY_BANK.get(risk_type, ()))
    dynamic = [dynamic_query] if isinstance(dynamic_query, str) else (dynamic_query or [])
    queries.extend(item.strip() for item in dynamic
                   if item.strip() and is_positive_risk_query(risk_type, item))
    # Preserve order while avoiding repeated SigLIP text encodes.
    return list(dict.fromkeys(query.rstrip(". ") for query in queries if query.strip()))
