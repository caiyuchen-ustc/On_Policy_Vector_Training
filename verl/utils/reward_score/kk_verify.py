"""Custom reward / verification function for Knights-and-Knaves (K&K) logic RL
inside G-OPD (verl).

K&K (K-and-K/knights-and-knaves, the dataset behind Logic-RL, arXiv:2502.14768)
is a propositional-logic puzzle: an island has only *knights* (always tell the
truth) and *knaves* (always lie); given each inhabitant's statement, decide who
is a knight and who is a knave. Every puzzle has a unique gold assignment.

Verification is purely **rule-based and tamper-proof**: we know the gold role of
every named inhabitant, so we extract the model's claimed role for each name and
award 1.0 iff EVERY name's role matches the gold (an all-or-nothing exact match,
as in Logic-RL). A partially-correct or reordered answer scores 0.0, so the model
cannot earn reward without solving the whole puzzle. "Leniency" only affects which
wrapper we read the roles from (``<answer>``, ``\\boxed{}``, the
``(1) X is a knight`` conclusion list, or the free-text tail) — never the
requirement that each extracted role equals the gold.

verl calls this through ``custom_reward_function.path`` / ``.name`` with the
signature ``compute_score(data_source, solution_str, ground_truth, extra_info)``
(mirroring ``sciknoweval_verify.py`` / ``synlogic_verify.py``); the naive reward
manager accepts the float return.

``ground_truth`` is a JSON string ``{"Michael": "knight", "Zoey": "knave", ...}``
produced by ``data/logic_kk/convert_kk.py`` from the dataset's ``names`` +
``solution`` (bool list, True == knight).

Environment variables:
  KK_REQUIRE_FORMAT : "1" to multiply accuracy by a strict format reward
                      (response must contain one <think>..</think> block and an
                      <answer>..</answer> block). Default "0" (accuracy-only),
                      friendlier for cold-start RL.
  KK_VERBOSE        : "1" to print per-sample extraction debug. Default "0".
"""

import json
import os
import re

THOUGHT_DELIMITER_END = "</think>"

_KNIGHT = "knight"
_KNAVE = "knave"


def _verbose() -> bool:
    return os.environ.get("KK_VERBOSE", "0") == "1"


def _parse_gold(ground_truth):
    """Return a {name: 'knight'|'knave'} dict from the ground_truth.

    Accepts the JSON dict produced by convert_kk.py, or a dict already.
    """
    gt = ground_truth
    if isinstance(gt, (bytes, bytearray)):
        gt = gt.decode("utf-8", errors="ignore")
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            return {}
    if not isinstance(gt, dict):
        return {}
    out = {}
    for name, role in gt.items():
        r = str(role).strip().lower()
        if r.startswith("knight"):
            out[str(name)] = _KNIGHT
        elif r.startswith("knave"):
            out[str(name)] = _KNAVE
    return out


def _strip_think(text: str) -> str:
    if THOUGHT_DELIMITER_END in text:
        return text.split(THOUGHT_DELIMITER_END)[-1]
    return text


def _candidate_regions(solution_str: str):
    """Yield text regions to read role assignments from, most reliable first.

    1) the last <answer>...</answer> block (prompt-mandated),
    2) the last \\boxed{...} block,
    3) the whole post-<think> tail (catches "(1) X is a knight" conclusion lists
       and free-text conclusions).
    """
    text = "" if solution_str is None else str(solution_str)
    regions = []

    ans = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if ans:
        regions.append(ans[-1])

    boxed = re.findall(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if boxed:
        regions.append(boxed[-1])

    regions.append(_strip_think(text))
    return regions


def _role_for_name(region: str, name: str):
    """Find name's claimed role in ``region``: returns 'knight'/'knave'/None.

    We look for "<name> ... is ... (a) knight/knave" allowing a short gap (e.g.
    "is a knight", "is not a knave" -> handled by taking the nearest role word
    AFTER the name). To stay robust to "X is a knave" we scan for the FIRST
    knight/knave token following the name within a small window.
    """
    if not region or not name:
        return None
    # All occurrences of the name; for each, read the first role word after it
    # within a window of ~40 chars (covers "is a knight", "is, in fact, a knave").
    role = None
    for m in re.finditer(re.escape(name), region, re.IGNORECASE):
        window = region[m.end(): m.end() + 60].lower()
        rm = re.search(r"\b(knight|knave)\b", window)
        if rm:
            # last occurrence wins (a later, more explicit statement overrides)
            role = _KNIGHT if rm.group(1) == _KNIGHT else _KNAVE
    return role


def _format_ok(solution_str: str) -> bool:
    text = "" if solution_str is None else str(solution_str)
    has_think = text.count("<think>") >= 1 and text.count(THOUGHT_DELIMITER_END) >= 1
    has_answer = re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE) is not None
    return has_think and has_answer


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  timeout_score: float = 0, timeout: float = 30.0):
    del data_source, extra_info, timeout_score, timeout

    gold = _parse_gold(ground_truth)
    if not gold:
        if _verbose():
            print("[kk_verify] empty/invalid gold -> 0.0")
        return 0.0

    regions = _candidate_regions(solution_str)

    # Try each region in priority order; accept the first region in which EVERY
    # name resolves to a role, then require all of them to match the gold.
    best = 0.0
    for region in regions:
        preds = {name: _role_for_name(region, name) for name in gold}
        if any(v is None for v in preds.values()):
            continue  # this region doesn't pin down every name; try the next
        correct = all(preds[name] == gold[name] for name in gold)
        acc = 1.0 if correct else 0.0
        if acc > best:
            best = acc
        if best >= 1.0:
            break

    require_format = os.environ.get("KK_REQUIRE_FORMAT", "0") == "1"
    format_res = (1.0 if _format_ok(solution_str) else 0.0) if require_format else 1.0

    if _verbose():
        print(f"[kk_verify] gold={gold} acc={best} fmt={format_res}")

    return float(best * format_res)
