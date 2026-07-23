"""Custom reward / verification function for instruction-following (IF) RL inside
G-OPD (verl), using the **IFEvalG** scorer (the official Google-Research IFEval
implementation extended with IFBench constraints, ported from M2RL's
``slime/slime/rewards/IFEvalG``).

Unlike the VerIF scheme (``verif_verify.py``), IFEvalG scoring is **strict and
binary** and does NOT ``exec`` any dataset-shipped code:

  * each sample carries an ``instruction_id_list`` (e.g.
    ``"length_constraints:number_words"``) and a parallel ``kwargs`` list
    (the arguments for each constraint, e.g. ``{"relation":"at least",
    "num_words":50}``);
  * every instruction id maps to a constraint class in
    ``ifevalg.instructions_registry``; the class's ``check_following(response)``
    is run, and the response scores **1.0 only if it follows every instruction**,
    else **0.0**.

Two checker backends are dispatched per-sample (matching M2RL, which scores the
training set with ``rm_type=ifevalg`` and the IFBench_test val set with
``rm_type=ifbench``):

  * **ifevalg** (``verl.utils.reward_score.ifevalg``): the training-time scorer,
    covering the constraint ids in the Nemotron-IF training subset;
  * **ifbench** (``verl.utils.reward_score.ifbench``, ported from
    allenai/IFBench): covers the 58 IFBench-only constraints
    (``count:``/``ratio:``/``words:``/``format:``/``sentence:``/``repeat:``/
    ``custom:``) used by the ``allenai/IFBench_test`` validation set.

If every instruction id in a sample is in the IFEvalG registry it is scored with
ifevalg; otherwise the whole sample is scored with ifbench. The ifbench backend
needs extra deps (``emoji``, ``syllapy``); they are imported lazily so the
training set scores without them.

This mirrors M2RL's ``compute_ifevalg_reward`` (slime/rewards/IFEvalG/__init__.py
and rm_hub/ifbench.py), adapted to the G-OPD custom-reward signature
``compute_score(data_source, solution_str, ground_truth, extra_info)`` and loaded
via ``custom_reward_function.path`` / ``.name`` with
``reward_model.reward_manager=naive``.

The per-sample verification metadata lives in the dataset's ``extra_info``
column (built by ``data/ifevalg/convert_ifevalg_*.py``):

    extra_info = {
        "instruction_id_list": [...],     # constraint ids
        "kwargs": [ {...}, {...}, ... ],   # per-constraint arguments
        "prompt_text": "...",              # the raw instruction (optional)
        ...
    }

The naive reward manager also injects the decoded prompt as
``extra_info["prompt_str"]``, used as a fallback prompt when ``prompt_text`` is
absent (some constraints, e.g. ``combination:repeat_prompt``, need the prompt).

Environment variables:
  IFEVALG_VERBOSE : "1" to print per-sample scoring debug. Default "0".
  IFEVALG_LOOSE   : "1" to use the loose IFEval check (upper bound) instead of
                    the strict check. Default "0" (strict, matches M2RL).
"""

import os

from verl.utils.reward_score.ifevalg import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from verl.utils.reward_score.ifevalg import instructions_registry as _ifevalg_registry

THOUGHT_DELIMITER_END = "</think>"


def _ifevalg_known_ids():
    return set(_ifevalg_registry.INSTRUCTION_DICT.keys())


def _verbose() -> bool:
    return os.environ.get("IFEVALG_VERBOSE", "0") == "1"


def _loose() -> bool:
    return os.environ.get("IFEVALG_LOOSE", "0") == "1"


def _normalize_instruction_ids(raw_ids) -> list:
    """Coerce instruction identifiers into a clean list[str]."""
    normalized = []
    for entry in (raw_ids if raw_ids is not None else []):
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(raw_kwargs, num_instructions: int) -> list:
    """Convert stored kwargs into the per-instruction list expected by IFEval.

    Mirrors M2RL's ``_coerce_kwargs_list``: accept a list[dict] / single dict /
    anything; pad or truncate to ``num_instructions``; drop explicit ``None``
    values (which IFEval's ``build_description`` treats as "use the default").
    """
    if isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    elif raw_kwargs is None:
        processed = [{} for _ in range(num_instructions)]
    else:
        # list / tuple / numpy array of (dict | None | other)
        processed = []
        for entry in list(raw_kwargs):
            processed.append(dict(entry) if isinstance(entry, dict) else {})

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    sanitized = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _metadata_from(extra_info, ground_truth) -> dict:
    """Pull the IFEvalG fields out of ``extra_info`` (or ``ground_truth`` dict)."""
    meta = {}
    if isinstance(extra_info, dict):
        meta = dict(extra_info)
    # Allow the verification fields to live under ground_truth too (defensive).
    if isinstance(ground_truth, dict):
        for key in ("instruction_id_list", "kwargs", "prompt_text"):
            meta.setdefault(key, ground_truth.get(key))
    return meta


def _build_input_example(metadata: dict):
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        if _verbose():
            print("[ifevalg_verify] missing instruction_id_list -> None")
        return None

    # Prompt text: prefer the dataset's stored prompt, fall back to the decoded
    # prompt the naive reward manager injects as prompt_str.
    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = metadata.get("prompt_str")
    prompt_text = "" if prompt_text is None else str(prompt_text)

    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))

    record_id = metadata.get("record_id") or metadata.get("index") or 0
    try:
        key = int(record_id)
    except (TypeError, ValueError):
        key = 0

    return InputExample(
        key=key,
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def _parse_solution(solution_str: str) -> str:
    """Strip the model's reasoning and trailing chat markers (mirror M2RL).

    M2RL's async_rm keeps the text after the last ``</think>`` and trims a
    trailing ``<|im_end|>``. We do the same so scoring sees the final answer.
    """
    text = "" if solution_str is None else str(solution_str)
    if THOUGHT_DELIMITER_END in text:
        text = text.split(THOUGHT_DELIMITER_END)[-1]
    text = text.strip()
    if text.endswith("<|im_end|>"):
        text = text[: text.rfind("<|im_end|>")].strip()
    return text


def _score_ifevalg(inp, response: str):
    """Score with the training-time IFEvalG checker (signature: inp, response)."""
    checker = test_instruction_following_loose if _loose() else test_instruction_following_strict
    return checker(inp, response)


def _score_ifbench(inp, response: str):
    """Score with the IFBench checker (covers the 58 IFBench_test constraints).

    IFBench's strict/loose checkers take ``(inp, prompt_to_response_dict)`` keyed
    by the prompt string, so adapt the call here. Imported lazily because it
    pulls extra deps (emoji / syllapy) only needed for the val constraints.
    """
    from verl.utils.reward_score.ifbench import (
        InputExample as _IFBInputExample,
        test_instruction_following_loose as _ifb_loose,
        test_instruction_following_strict as _ifb_strict,
    )

    ifb_inp = _IFBInputExample(
        key=inp.key,
        instruction_id_list=list(inp.instruction_id_list),
        prompt=inp.prompt,
        kwargs=[dict(k) for k in inp.kwargs],
    )
    checker = _ifb_loose if _loose() else _ifb_strict
    return checker(ifb_inp, {ifb_inp.prompt: response})


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  timeout_score: float = 0, timeout: float = 30.0):
    del data_source, timeout_score, timeout

    metadata = _metadata_from(extra_info, ground_truth)
    inp = _build_input_example(metadata)
    if inp is None:
        return 0.0

    response = _parse_solution(solution_str)

    # Dispatch per-sample: the training set (Nemotron-IF) uses IFEvalG-registry
    # constraints; the validation set (allenai/IFBench_test) uses IFBench-only
    # constraints. Route to whichever backend covers this sample's ids; if any
    # id is outside the IFEvalG registry, score the whole sample with IFBench.
    use_ifbench = not set(inp.instruction_id_list).issubset(_ifevalg_known_ids())

    try:
        if use_ifbench:
            output = _score_ifbench(inp, response)
            backend = "ifbench"
        else:
            output = _score_ifevalg(inp, response)
            backend = "ifevalg"
        score = 1.0 if output.follow_all_instructions else 0.0
    except Exception as e:  # a buggy constraint must not crash training
        if _verbose():
            import traceback

            print(f"[ifevalg_verify] scoring error: {e}\n{traceback.format_exc()}")
        return 0.0

    if _verbose():
        print(
            f"[ifevalg_verify] backend={backend} ids={inp.instruction_id_list} "
            f"per_instr={output.follow_instruction_list} -> {score}"
        )

    return float(score)
