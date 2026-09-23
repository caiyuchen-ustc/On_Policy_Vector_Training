"""Custom reward / verification function for training on SciKnowEval (MCQ) data
inside G-OPD (verl).

SciKnowEval (hicai-zju/SciKnowEval) is a multiple-choice science benchmark
(physics / chemistry / biology / materials). Every example is an MCQ whose gold
answer is a single option letter A/B/C/D (a few 2-choice items use only A/B).
The prompt (see data/sciknoweval/convert_sciknoweval.py) instructs the model to
reply with `<reasoning>..</reasoning><answer>X</answer>`, mirroring SDPO's
`reward_score/feedback/mcq.py`.

verl calls this through ``custom_reward_function.path`` / ``.name`` with the
signature ``compute_score(data_source, solution_str, ground_truth, extra_info)``
and the naive reward manager accepts a float return.

Scoring is intentionally simple and tamper-proof: we robustly EXTRACT the chosen
option letter from the model output, then award 1.0 iff it equals the gold
letter, else 0.0. "Leniency" only affects *which wrapper* we read the letter
from (``<answer>``, ``\\boxed{}``, "the answer is X", or a trailing standalone
letter) — it never changes the requirement that the letter must equal the gold,
so a wrong choice can never score.

Environment variables:
  SCIKNOWEVAL_REQUIRE_FORMAT : "1" to multiply accuracy by a strict format
                               reward (response must end with `<answer>X</answer>`).
                               Default "0" (accuracy-only), friendlier for
                               cold-start RL where the model may not yet emit the
                               exact tags.
  SCIKNOWEVAL_VERBOSE        : "1" to print per-sample extraction debug. Default "0".
"""

import os
import re

THOUGHT_DELIMITER_END = "</think>"

# Valid option letters. SciKnowEval items have at most 4 choices (A-D).
_VALID = "ABCD"


def _verbose() -> bool:
    return os.environ.get("SCIKNOWEVAL_VERBOSE", "0") == "1"


def _gold_letter(ground_truth) -> str:
    s = str(ground_truth).strip().upper()
    m = re.search(r"[A-D]", s)
    return m.group(0) if m else ""


def _first_option_letter(text: str):
    """Return the first standalone A-D option letter in ``text``, or None.

    "Standalone" means not glued to surrounding letters (so we don't pick the
    'A' inside a word). We accept a letter optionally followed by ) . : or end.
    Case-insensitive: used only on explicit wrappers (<answer>, \\boxed{}) where
    the model deliberately placed the letter, so a lowercase 'c' is fine.
    """
    if not text:
        return None
    m = re.search(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
    return m.group(1).upper() if m else None


def _strip_think(text: str) -> str:
    if THOUGHT_DELIMITER_END in text:
        return text.split(THOUGHT_DELIMITER_END)[-1]
    return text


def _extract_pred(solution_str: str):
    """Extract the model's chosen option letter, trying the most reliable
    wrappers first. Returns an uppercase letter A-D, or None if none found."""
    text = "" if solution_str is None else str(solution_str)

    # 1) Last <answer>...</answer> block (the prompt-mandated format).
    ans_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if ans_blocks:
        letter = _first_option_letter(ans_blocks[-1])
        if letter:
            return letter

    # 2) \boxed{X}.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        letter = _first_option_letter(boxed[-1])
        if letter:
            return letter

    # 3) Natural-language cue: "answer is X" / "答案是 X" / "正确答案 X" / "选 X".
    cue = re.search(
        r"(?:answer\s*(?:is|:)?|答案\s*(?:是|为|:|：)?|正确答案\s*(?:是|为|:|：)?|选)\s*[\(\[\{]?\s*([A-D])\b",
        text,
        re.IGNORECASE,
    )
    if cue:
        return cue.group(1).upper()

    # 4) Fallback: after dropping any <think> block, take the LAST standalone
    #    A-D letter in the response (models often end with the bare choice).
    tail = _strip_think(text)
    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", tail)
    if matches:
        return matches[-1].upper()
    return None


def _format_ok(solution_str: str) -> bool:
    """Strict format: response ends with <answer>X</answer> (X in A-D)."""
    text = "" if solution_str is None else str(solution_str)
    return re.search(r"<answer>\s*[A-D]\s*</answer>\s*$", text.strip(),
                     re.IGNORECASE) is not None


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  timeout_score: float = 0, timeout: float = 30.0):
    del data_source, extra_info, timeout_score, timeout

    gold = _gold_letter(ground_truth)
    pred = _extract_pred(solution_str)

    accuracy = 1.0 if (pred is not None and gold and pred == gold) else 0.0

    require_format = os.environ.get("SCIKNOWEVAL_REQUIRE_FORMAT", "0") == "1"
    format_res = (1.0 if _format_ok(solution_str) else 0.0) if require_format else 1.0

    if _verbose():
        print(f"[sciknoweval_verify] gold={gold!r} pred={pred!r} "
              f"acc={accuracy} fmt={format_res}")

    return float(accuracy * format_res)
