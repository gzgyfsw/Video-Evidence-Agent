import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outcome_cost_reward_v1 import reward

VIOLENCE = "Violence & Dangerous"
SEXUAL = "Sexual Content"


def score(ref="L2", category=VIOLENCE, level="L2", frames=4, **kw):
    return reward(reference_category=VIOLENCE, reference_severity=ref,
                  predicted_category=category, predicted_severity=level,
                  visual_inputs=frames, **kw)


def test_cost_cannot_trade_away_correct_l2_or_favor_blanket_review():
    assert score(frames=10**9)["total"] > score(level="L1")["total"]
    assert score(level="L1")["total"] > score(category=SEXUAL)["total"]
    assert score()["total"] > score(frames=12)["total"]
    assert score(category="Harmless", level="L0")["total"] == 0.


def test_review_category_is_secondary_and_cost_cannot_reverse_it():
    assert score(ref="L1", level="L1", frames=10**9)["total"] > score(
        ref="L1", level="L1", category=SEXUAL)["total"]
    assert score(ref="L1", level="L1", category=SEXUAL)["total"] > score(ref="L1")["total"]


def test_boundary_retains_review_preference_and_accepts_block():
    assert score(level="L1", boundary=True, frames=10**9)["total"] > score(boundary=True)["total"]
    assert score(boundary=True)["total"] > 0.


def test_unknown_and_infrastructure_failures_are_not_student_negatives():
    assert score(infrastructure_error=True)["total"] is None
    assert score(frames=None)["total"] is None
    assert reward(reference_category=None, reference_severity="L2",
                  predicted_category=VIOLENCE, predicted_severity="L2", visual_inputs=4)["total"] is None


def test_invalid_output_is_worse_than_valid_abstention():
    assert score(valid=False)["total"] < score(level="L1")["total"]
    assert score(category=None)["total"] < 0.
    assert score(category="Harmless", level="L1")["total"] < 0.
