from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    result = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    if isinstance(result, dict):
        if "score" in result:
            return float(result["score"])
        if "acc" in result:
            return float(result["acc"])
        raise TypeError(f"Unsupported reward dict keys: {list(result.keys())}")
    return float(result)
