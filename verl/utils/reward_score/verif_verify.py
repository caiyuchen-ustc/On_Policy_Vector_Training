"""Custom reward / verification function for instruction-following (IF) RL inside
G-OPD (verl), reproducing THU-KEG's **VerIF** scheme (arXiv:2506.09942).

VerIF computes the reward as

    reward = rule_score * llm_score

where

  * **rule_score** comes from *rule-based code checks*: each constraint ships a
    short Python function defining ``check_following(instruction, response)``
    that returns truthy iff the response satisfies that one hard constraint
    (length / format / keyword / ...). We run every rule function and take the
    mean (1 per passed check, 0 per failed/erroring check).
  * **llm_score** comes from an *LLM reasoning verifier* (QwQ-32B or THU-KEG's
    IF-Verifier-7B) that judges the *soft* constraints (tone, style, ...). The
    constraint's function calls ``llm_score`` / ``llm_judge`` / ``llm_extract``
    from a module named ``llm_call``; we inject a synthetic ``llm_call`` module
    (OpenAI-compatible ``/chat/completions`` client) so those imports resolve.

This file is the G-OPD port of VerIF's
``verl/utils/reward_score/local_server/{constraint_analyzer,code_executer,llm_call}.py``,
collapsed into a single self-contained custom reward function with the G-OPD
signature ``compute_score(data_source, solution_str, ground_truth, extra_info)``
(mirroring ``sciknoweval_verify.py`` / ``synlogic_verify.py``). verl loads it via
``custom_reward_function.path`` / ``.name`` with ``reward_model.reward_manager=naive``.

The dataset (``THU-KEG/VerInstruct``, built by ``data/verif/convert_verif.py``)
stores per-sample verification in ``reward_model.ground_truth`` as a JSON string::

    {"checkers": ["[rule] Number_Paragraphs", "[llm] Desired_Writing_Style", ...],
     "functions": ["def check_following(instruction, response): ...",
                   "from llm_call import llm_score\\ndef check_following(...): ..."]}

The instruction is the decoded prompt, which the naive reward manager places in
``extra_info["prompt_str"]``.

Environment variables:
  VERIF_LLM_ENABLE     : "1" to enable the LLM soft-constraint judge. Default "0"
                         -> rule-only (llm_score forced to 1.0, so the reward is
                         purely the rule pass-rate). Enable only after deploying a
                         verifier endpoint, otherwise every soft constraint would
                         block reward.
  IF_LLM_VERIFIER_URL  : OpenAI-compatible base URL of the verifier (e.g.
                         "http://127.0.0.1:8000/v1"). Required when LLM is enabled.
  IF_LLM_VERIFIER_MODEL: served model name (e.g. "QwQ-32B" / "IF-Verifier-7B").
  IF_LLM_VERIFIER_KEY  : API key (default "EMPTY", as vLLM/SGLang accept).
  VERIF_LLM_TIMEOUT    : per-call timeout seconds (default 120).
  VERIF_REQUIRE_FORMAT : "1" to multiply the reward by a strict
                         <think>..</think> format reward. Default "0".
  VERIF_VERBOSE        : "1" to print per-sample scoring debug. Default "0".

SECURITY NOTE: like VerIF upstream, this ``exec``s the per-sample ``functions``
source that ships with the dataset. VerInstruct is a trusted, curated dataset;
do not point this verifier at untrusted ``functions`` strings.
"""

import json
import os
import re
import sys
import traceback
import types
from concurrent.futures import ThreadPoolExecutor

THOUGHT_DELIMITER_END = "</think>"

# ---------------------------------------------------------------------------
# LLM verifier client (synthetic ``llm_call`` module)
# ---------------------------------------------------------------------------
# VerIF's LLM constraint functions begin with ``from llm_call import llm_score``
# (or llm_judge / llm_extract). To run those functions here we register a module
# named ``llm_call`` in sys.modules that talks to an OpenAI-compatible endpoint.
# This mirrors VerIF/verl/utils/reward_score/local_server/llm_call.py, but reads
# the endpoint from env vars and degrades gracefully when the LLM is disabled.

_LLM_PROMPT_SCORE = """请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。

[指令]
{instruction}

[回复]
{response}

[约束]
{checkers}

请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
请在回答的最开始用[[score]]格式输出你的分数。
如果遵循所有的约束，请输出[[1]]，否则输出[[0]]
"""

_LLM_PROMPT_JUDGE = """请判断以下文本是否满足给定的约束，仅回答是或否，不要输出其他内容。

原始指令：{instruction}

文本：{response}

约束：{constraint}

原始指令描述了基本的任务信息，给定的约束介绍了应该满足的具体的一个约束。
请判断以下文本是否满足给定的这个约束（仅仅判断是否满足给定的约束），仅回答是或否，不要输出其他内容。
"""


def _llm_enabled() -> bool:
    return os.environ.get("VERIF_LLM_ENABLE", "0") == "1" and bool(
        os.environ.get("IF_LLM_VERIFIER_URL")
    )


def _verbose() -> bool:
    return os.environ.get("VERIF_VERBOSE", "0") == "1"


def _llm_generate_chat(messages, max_tokens=4096, temperature=0.0):
    """Call the configured OpenAI-compatible verifier endpoint; return text.

    Returns "" on any failure so a flaky verifier degrades to a 0 soft score
    rather than crashing the reward computation.
    """
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover - openai should be installed
        if _verbose():
            print(f"[verif_verify] openai import failed: {e}")
        return ""

    url = os.environ.get("IF_LLM_VERIFIER_URL")
    model = os.environ.get("IF_LLM_VERIFIER_MODEL", "")
    key = os.environ.get("IF_LLM_VERIFIER_KEY", "EMPTY")
    timeout = float(os.environ.get("VERIF_LLM_TIMEOUT", "120"))
    try:
        client = OpenAI(api_key=key, base_url=url, timeout=timeout)
        resp = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        if _verbose():
            print(f"[verif_verify] LLM call failed: {e}")
        return ""


def _extract_bracket_score(text: str) -> int:
    """Read VerIF's ``[[1]]`` / ``[[0]]`` score marker from the verifier output."""
    m = re.search(r"\[\[(\d+)\]\]", text or "")
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _llm_score(instruction, response, checkers):
    """Soft-constraint judge: 1 if the response follows all soft constraints."""
    if not _llm_enabled():
        # LLM disabled -> do not penalize; reward becomes the pure rule pass-rate.
        return 1
    if isinstance(response, list):
        response = "\n\n".join(response)
    prompt = _LLM_PROMPT_SCORE.format(
        instruction=instruction, response=response, checkers=checkers
    )
    out = _llm_generate_chat([{"role": "user", "content": prompt}], max_tokens=4096)
    return _extract_bracket_score(out)


def _llm_judge(instruction, response, constraint):
    """Single-constraint yes/no judge (VerIF llm_judge)."""
    if not _llm_enabled():
        return True
    if isinstance(response, list):
        response = "\n\n".join(response)
    prompt = _LLM_PROMPT_JUDGE.format(
        instruction=instruction, response=response, constraint=constraint
    )
    out = _llm_generate_chat([{"role": "user", "content": prompt}], max_tokens=128)
    return bool(out) and out[0] == "是"


def _llm_extract(instruction, response, specific_prompt):
    """Extract original text per an extraction prompt (VerIF llm_extract)."""
    if not _llm_enabled():
        return response if isinstance(response, str) else str(response)
    suffix = "\n请直接输出文本中的原文信息，不要改写，不要添加任何额外的信息。"
    prompt = f"文本：{response}\n抽取要求：{specific_prompt}" + suffix
    return _llm_generate_chat([{"role": "user", "content": prompt}], max_tokens=1024)


def _ensure_llm_call_module():
    """Register a synthetic ``llm_call`` module so ``from llm_call import ...``
    inside the dataset's constraint functions resolves to our client."""
    if "llm_call" in sys.modules:
        return
    mod = types.ModuleType("llm_call")
    mod.llm_score = _llm_score
    mod.llm_judge = _llm_judge
    mod.llm_extract = _llm_extract
    sys.modules["llm_call"] = mod


# ---------------------------------------------------------------------------
# Rule-based code checks (port of VerIF code_executer.execute_code)
# ---------------------------------------------------------------------------


def _run_check_following(instruction, response, function_src):
    """Exec one constraint function and call ``check_following(instruction, response)``.

    Returns 1 if it returns truthy, else 0. Any error (bad source, raised
    exception, missing function) -> 0, matching VerIF's behavior of treating a
    failed/erroring check as not-followed.
    """
    global_context = {}
    local_vars = {"response": response}
    try:
        exec(function_src, global_context, local_vars)
        fn = local_vars.get("check_following") or global_context.get("check_following")
        if callable(fn):
            return 1 if fn(instruction, response) else 0
        if _verbose():
            print("[verif_verify] check_following missing/not callable")
        return 0
    except Exception as e:
        if _verbose():
            print(f"[verif_verify] rule exec error: {e}\n{traceback.format_exc()}")
        return 0


def _is_llm_function(checker_name: str, function_src: str) -> bool:
    """Classify a constraint as LLM-based vs rule-based.

    Priority 1: explicit VerIF tag in the checker name ("[llm]" / "[rule]").
    Priority 2 (fallback, since many real VerInstruct checkers are untagged):
    treat it as LLM-based iff its function source references the llm_call API.
    """
    name = checker_name or ""
    if "[llm]" in name:
        return True
    if "[rule]" in name:
        return False
    src = function_src or ""
    return bool(re.search(r"\bllm_(?:call|score|judge|extract)\b", src)) or (
        "llm_call" in src
    )


def _format_reward(response: str) -> float:
    """Strict format check: a single <think>..</think> block is present."""
    r = response.strip()
    if r.count("<think>") == 1 and r.count(THOUGHT_DELIMITER_END) == 1:
        return 1.0
    return 0.0


def _parse_ground_truth(ground_truth):
    """Return (checkers, functions) from the ground_truth JSON (or dict)."""
    item = ground_truth
    if isinstance(item, (bytes, bytearray)):
        item = item.decode("utf-8", errors="ignore")
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except Exception:
            return [], []
    if not isinstance(item, dict):
        return [], []
    checkers = list(item.get("checkers", []) or [])
    functions = list(item.get("functions", []) or [])
    return checkers, functions


def _parse_solution(solution_str: str) -> str:
    """Strip the model's reasoning: keep the text after the last </think>."""
    text = "" if solution_str is None else str(solution_str)
    if THOUGHT_DELIMITER_END in text:
        text = text.split(THOUGHT_DELIMITER_END)[-1]
    return text.strip()


def _instruction_from(extra_info, ground_truth) -> str:
    """The IF instruction is the decoded prompt (naive manager -> prompt_str)."""
    if extra_info:
        for key in ("prompt_str", "instruction", "question"):
            val = extra_info.get(key)
            if val:
                return val if isinstance(val, str) else str(val)
    return ""


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  timeout_score: float = 0, timeout: float = 30.0):
    del data_source, timeout_score, timeout

    checkers, functions = _parse_ground_truth(ground_truth)
    if not functions:
        if _verbose():
            print("[verif_verify] no functions in ground_truth -> 0.0")
        return 0.0

    instruction = _instruction_from(extra_info, ground_truth)
    response = _parse_solution(solution_str)

    # Split constraints into rule-based and LLM-based.
    rule_functions = []
    llm_checkers = []
    n = min(len(checkers), len(functions)) if checkers else 0
    if checkers and len(checkers) == len(functions):
        for checker, func in zip(checkers, functions):
            if _is_llm_function(checker, func):
                llm_checkers.append(checker)
            else:
                rule_functions.append(func)
    else:
        # No (aligned) checker names: classify purely from the function source.
        for func in functions:
            if _is_llm_function("", func):
                llm_checkers.append(func)
            else:
                rule_functions.append(func)
    del n

    # rule_score = mean pass-rate over rule functions (1.0 if there are none).
    if rule_functions:
        if _llm_enabled():
            _ensure_llm_call_module()  # some "rule" funcs may still import llm_call
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(
                ex.map(lambda f: _run_check_following(instruction, response, f), rule_functions)
            )
        rule_score = sum(results) / len(results)
    else:
        rule_score = 1.0

    # llm_score = soft-constraint judge over all LLM checkers (1.0 when disabled).
    if llm_checkers and _llm_enabled():
        _ensure_llm_call_module()
        q_score = float(_llm_score(instruction, response, llm_checkers))
    else:
        q_score = 1.0

    accuracy = float(rule_score) * float(q_score)

    require_format = os.environ.get("VERIF_REQUIRE_FORMAT", "0") == "1"
    format_res = _format_reward(response) if require_format else 1.0

    if _verbose():
        print(
            f"[verif_verify] rule={rule_score:.3f} (n={len(rule_functions)}) "
            f"llm={q_score:.3f} (n={len(llm_checkers)}, enabled={_llm_enabled()}) "
            f"fmt={format_res} -> {accuracy * format_res:.3f}"
        )

    return float(accuracy * format_res)
