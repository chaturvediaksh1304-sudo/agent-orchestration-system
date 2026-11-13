"""GAIA quasi-exact-match scoring.

Follows the logic of the official GAIA leaderboard scorer: a numeric ground
truth is compared as a number, a ground truth containing "," or ";" is compared
element by element, and anything else is compared as a normalised string.
"""

import re
import string
from typing import List, Optional, Tuple

FINAL_ANSWER_MARKER = "FINAL ANSWER:"


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_number(text: str) -> float:
    """Strip "$", "%" and "," then parse. Anything unparseable becomes inf.

    Only those three characters are removed, exactly as upstream does. A word
    unit ("12 kg") therefore does not match "12".
    """
    for char in ("$", "%", ","):
        text = text.replace(char, "")
    try:
        return float(text)
    except ValueError:
        return float("inf")


def _split_list(text: str) -> List[str]:
    return re.split(r"[,;]", text)


def _normalize_text(text: str, remove_punct: bool = True) -> str:
    squashed = re.sub(r"\s", "", text).lower()
    if remove_punct:
        return squashed.translate(str.maketrans("", "", string.punctuation))
    return squashed


def question_scorer(model_answer: Optional[str], ground_truth: str) -> bool:
    if model_answer is None:
        model_answer = "None"

    if _is_float(ground_truth):
        return _normalize_number(model_answer) == float(ground_truth)

    if any(char in ground_truth for char in (",", ";")):
        truth_items = _split_list(ground_truth)
        answer_items = _split_list(model_answer)
        if len(truth_items) != len(answer_items):
            return False
        return all(
            _normalize_number(answer) == float(truth)
            if _is_float(truth)
            # Punctuation is kept inside list elements, as upstream does.
            else _normalize_text(answer, remove_punct=False)
            == _normalize_text(truth, remove_punct=False)
            for answer, truth in zip(answer_items, truth_items)
        )

    return _normalize_text(model_answer) == _normalize_text(ground_truth)


def extract_final_answer(text: str) -> Tuple[str, bool]:
    """Pull the answer out of the agent's last message.

    Returns (answer, had_marker). Without the marker the whole message is the
    answer, which is what the leaderboard would be handed too.
    """
    index = text.upper().rfind(FINAL_ANSWER_MARKER)
    if index == -1:
        return text.strip(), False
    return text[index + len(FINAL_ANSWER_MARKER):].strip(), True
