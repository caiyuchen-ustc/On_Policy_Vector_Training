try:
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError:
    verify = None
    parse = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


_SOLUTION_CLIP_CHARS = 300


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def _math_verify_score(ground_truth: str, model_output: str) -> float:
    if verify is None or parse is None:
        return 0.0

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    clipped_model_output = model_output[-_SOLUTION_CLIP_CHARS:]
    last_boxed = last_boxed_only_string(clipped_model_output)
    extracted_gold = parse("\\boxed{" + ground_truth + "}", gold_targets)
    extracted_pred = parse(last_boxed or "", pred_targets)

    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    timeout_score: float = 0,
    timeout: float = 30.0,
):
    del data_source, timeout_score, timeout

    model_output = "" if solution_str is None else str(solution_str)
    gt = "" if ground_truth is None else str(ground_truth)

    # First try math-verify for numeric/latex targets.
    try:
        score = _math_verify_score(gt, model_output)
        if score > 0:
            return float(score)
    except Exception as e:
        print(f"Error in scicence_verify math_verify path: {e}")

    # Fallback to science_mcq parser/scorer for A/B/C... style answers.
    try:
        from verl.utils.reward_score import science_mcq

        return float(science_mcq.compute_score(model_output, gt, extra_info=extra_info))
    except Exception as e:
        print(f"Error in scicence_verify science_mcq fallback: {e}")
        return 0.0
