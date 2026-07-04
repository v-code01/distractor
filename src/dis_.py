"""dis_ core: exact GSM8K scoring plus the distractor-robustness reductions the study needs -
pools of irrelevant sentences (one set with a stray number, one control set without), a
deterministic picker, an injector that drops a sentence after the first, the retention rate on
initially-correct items, and a check for whether a wrong answer incorporates the distractor's
number. Deterministic and label-based; no judge.
"""
from __future__ import annotations

import random
import re
from typing import Optional, Sequence

_INT = re.compile(r"-?\d[\d,]*")

# Irrelevant sentences that carry a stray number (should not change the answer).
NUMBER_DISTRACTORS = [
    "The store is 7 miles from the house.",
    "There were 12 clouds in the sky that day.",
    "The building has 5 floors.",
    "The temperature was 3 degrees that morning.",
    "The bus arrives every 9 minutes.",
    "A poster on the wall was 4 feet wide.",
]

# Irrelevant sentences with no number (the control).
TEXT_DISTRACTORS = [
    "The weather was pleasant that morning.",
    "A gentle breeze blew through the trees.",
    "The room was painted a soft shade of blue.",
    "Everyone seemed to be in a good mood.",
    "The music played softly in the background.",
    "It was a quiet and ordinary day.",
]


def parse_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def parse_pred(text: str) -> Optional[str]:
    m = _INT.findall(text.replace(",", ""))
    return m[-1] if m else None


def correct(pred: Optional[str], gold: str) -> bool:
    return pred == gold


def distractor_for(kind: str, idx: int, seed: int) -> str:
    """A deterministic distractor sentence of the requested kind for item idx."""
    pool = NUMBER_DISTRACTORS if kind == "number" else TEXT_DISTRACTORS
    return random.Random(f"{seed}:{kind}:{idx}").choice(pool)


def inject(question: str, sentence: str) -> str:
    """Insert `sentence` right after the first sentence of `question`; if the question is a
    single sentence, place it in front. Preserves the rest verbatim.
    """
    parts = question.split(". ")
    if len(parts) > 1:
        return parts[0] + ". " + sentence + " " + ". ".join(parts[1:])
    return sentence + " " + question


def retention_rate(pairs: Sequence[tuple[str, Optional[str]]]) -> float:
    """Fraction of (gold, final_pred) pairs that still give the gold answer."""
    if not pairs:
        return 0.0
    return sum(1 for gold, final in pairs if final == gold) / len(pairs)


def incorporates_number(pred: Optional[str], distractor_number: int) -> bool:
    """Whether the model's (wrong) answer contains the distractor's number as a substring - the
    mark of the stray number having been pulled into the computation.
    """
    return pred is not None and str(distractor_number) in pred
