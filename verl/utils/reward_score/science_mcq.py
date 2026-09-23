import re
from typing import Any, Optional

from .math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

try:
    from . import math_verify as math_verify_reward
except Exception:
    math_verify_reward = None


_OPTION_RE = re.compile(r"^\s*([A-J])\s*[\.:：\)]\s*(.+?)\s*$", re.MULTILINE)
_ANSWER_RE = re.compile(r"(?i)answer\s*[:：]\s*(.+)")
_STANDALONE_CHOICE_RE = re.compile(r"(?<![A-Z])([A-J])(?![A-Z])")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _prompt_text(extra_info: Optional[dict[str, Any]]) -> str:
    if not extra_info:
        return ""

    raw_prompt = extra_info.get("raw_prompt")
    if raw_prompt is not None:
        return _to_text(raw_prompt)

    return _to_text(extra_info.get("prompt_str"))


def _parse_options(prompt_text: str) -> dict[str, str]:
    return {match.group(1).upper(): match.group(2).strip() for match in _OPTION_RE.finditer(prompt_text)}


def _extract_candidate(solution_str: str) -> str:
    boxed = last_boxed_only_string(solution_str)
    if boxed is not None:
        try:
            candidate = remove_boxed(boxed)
        except Exception:
            candidate = None
        if candidate:
            return candidate.strip()

    answer_matches = _ANSWER_RE.findall(solution_str)
    if answer_matches:
        return answer_matches[-1].strip()

    lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return solution_str.strip()


def _strip_tex_wrappers(text: str) -> str:
    text = text.strip().replace("$", "")
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\%", "%")
    text = text.replace("\\,", "")
    text = text.replace("\\!", "")
    return text.strip()


def _normalize_general(text: str) -> str:
    text = _strip_tex_wrappers(str(text))
    try:
        text = normalize_final_answer(text)
    except Exception:
        pass
    text = _strip_tex_wrappers(text)
    return re.sub(r"\s+", "", text)


def _extract_choice_sequence(text: str) -> Optional[str]:
    cleaned = _strip_tex_wrappers(str(text)).upper()
    if re.fullmatch(r"[A-J](?:[\s,;/|&]+[A-J])*", cleaned):
        return re.sub(r"[^A-J]", "", cleaned)

    if len(cleaned) <= 32:
        matches = _STANDALONE_CHOICE_RE.findall(cleaned)
        if matches:
            return "".join(matches)

    return None


def _is_choice_ground_truth(ground_truth: str) -> bool:
    return _extract_choice_sequence(ground_truth) is not None


def _choice_match(candidate: str, ground_truth: str) -> bool:
    pred = _extract_choice_sequence(candidate)
    gt = _extract_choice_sequence(ground_truth)
    if pred is None or gt is None:
        return False

    if pred == gt:
        return True

    return len(pred) == len(gt) and sorted(pred) == sorted(gt)


def _extract_numeric_token(text: str) -> Optional[str]:
    match = _NUMBER_RE.search(text)
    return match.group(0) if match else None


def _option_matches_ground_truth(option_text: str, ground_truth: str) -> bool:
    option_norm = _normalize_general(option_text)
    gt_norm = _normalize_general(ground_truth)

    if option_norm == gt_norm:
        return True

    option_num = _extract_numeric_token(option_norm)
    gt_num = _extract_numeric_token(gt_norm)
    if gt_num is not None and gt_norm == gt_num and option_num == gt_num:
        return True

    if math_verify_reward is not None:
        try:
            if math_verify_reward.compute_score(f"\\boxed{{{option_text}}}", ground_truth) == 1.0:
                return True
        except Exception:
            pass

    return False


def _infer_correct_choice(options: dict[str, str], ground_truth: str) -> Optional[str]:
    for choice, option_text in options.items():
        if _option_matches_ground_truth(option_text, ground_truth):
            return choice
    return None


def _general_match(solution_str: str, candidate: str, ground_truth: str) -> bool:
    pred_norm = _normalize_general(candidate)
    gt_norm = _normalize_general(ground_truth)

    if pred_norm == gt_norm:
        return True

    if math_verify_reward is not None:
        try:
            if math_verify_reward.compute_score(f"\\boxed{{{candidate}}}", ground_truth) == 1.0:
                return True
        except Exception:
            pass

        try:
            if math_verify_reward.compute_score(solution_str, ground_truth) == 1.0:
                return True
        except Exception:
            pass

    return False


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict[str, Any]] = None) -> float:
    candidate = _extract_candidate(solution_str)

    if _is_choice_ground_truth(ground_truth):
        return 1.0 if _choice_match(candidate, ground_truth) else 0.0

    options = _parse_options(_prompt_text(extra_info))
    pred_choice = _extract_choice_sequence(candidate)
    correct_choice = _infer_correct_choice(options, ground_truth) if options else None

    if pred_choice is not None and len(pred_choice) == 1 and correct_choice is not None:
        return 1.0 if pred_choice == correct_choice else 0.0

    mapped_candidate = candidate
    if pred_choice is not None and len(pred_choice) == 1 and pred_choice in options:
        mapped_candidate = options[pred_choice]

    return 1.0 if _general_match(solution_str, mapped_candidate, ground_truth) else 0.0
