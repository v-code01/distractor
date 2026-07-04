from dis_ import (
    NUMBER_DISTRACTORS,
    TEXT_DISTRACTORS,
    correct,
    distractor_for,
    incorporates_number,
    inject,
    parse_gold,
    parse_pred,
    retention_rate,
)


def test_parse_gold_and_pred():
    assert parse_gold("work\n#### 72") == "72"
    assert parse_gold("x\n#### 1,000") == "1000"
    assert parse_pred("So the answer is 18.") == "18"
    assert parse_pred("no digits") is None


def test_correct():
    assert correct("18", "18") is True
    assert correct(None, "18") is False


def test_distractor_pools_are_typed():
    # number distractors contain a digit; text distractors do not
    assert all(any(c.isdigit() for c in d) for d in NUMBER_DISTRACTORS)
    assert all(not any(c.isdigit() for c in d) for d in TEXT_DISTRACTORS)


def test_distractor_for_deterministic():
    a = distractor_for("number", 3, seed=1)
    assert a == distractor_for("number", 3, seed=1)
    assert a in NUMBER_DISTRACTORS
    assert distractor_for("text", 3, seed=1) in TEXT_DISTRACTORS


def test_inject_after_first_sentence():
    q = "Tom has 5 apples. He eats 2. How many are left?"
    out = inject(q, "The sky was blue.")
    assert out == "Tom has 5 apples. The sky was blue. He eats 2. How many are left?"
    # single-sentence question: distractor goes in front
    assert inject("What is 2 plus 2?", "It rained.") == "It rained. What is 2 plus 2?"


def test_retention_rate():
    # (gold, final_pred): retained iff final still equals gold
    pairs = [("5", "5"), ("7", "3"), ("9", None), ("2", "2")]
    assert retention_rate(pairs) == 0.5
    assert retention_rate([]) == 0.0


def test_incorporates_number():
    # the model's wrong answer literally contains the distractor's number
    assert incorporates_number("42", 7) is False
    assert incorporates_number("47", 7) is True     # 7 appears in the answer
    assert incorporates_number(None, 7) is False
