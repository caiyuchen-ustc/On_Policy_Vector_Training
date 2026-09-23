"""Custom reward / verification function for training on SynLogic data inside G-OPD (verl).

This implements the verification scheme described in SynLogic's
`docs/training_guidance.md` and `docs/reward_example.py`:

  1. Select a verifier based on the sample's ``data_source`` (the SynLogic task name).
  2. Deserialize the full game data from ``extra_info["game_data_str"]`` into a
     SynLogic ``Data`` object (this carries the constraints / metadata that the
     verifier needs, not just the answer string).
  3. Extract the model's answer from ``<think>...</think><answer>...</answer>``
     and run ``verifier.verify(game_data, answer)``.

verl calls this through ``custom_reward_function.path`` / ``.name`` with the
signature ``compute_score(data_source, solution_str, ground_truth, extra_info)``.

Environment variables:
  SYNLOGIC_ROOT        : path to the SynLogic repo (default: sibling CODE/SynLogic).
  SYNLOGIC_REQUIRE_FORMAT : "1" to multiply accuracy by a strict format reward
                            (the paper's `final = format * accuracy`). Default "0"
                            (accuracy-only), which is friendlier for cold-start
                            distillation where the student may not yet emit the
                            exact tag format.
  SYNLOGIC_VERBOSE     : "1" to let the SynLogic verifiers print their (very
                            chatty) per-sample debug output. Default "0" silences
                            stdout/stderr during verification so training logs stay
                            readable.
  SYNLOGIC_LENIENT_FORMAT : "1" (default) to score the model's answer under
                            several format shells (\\boxed{}, ```python```,
                            [[...]], raw) and keep the best — so a correct answer
                            that omits the exact wrapper the verifier expects
                            still gets reward. Only the wrapper varies, never the
                            answer content, so wrong answers stay 0.0. Set "0" to
                            require the verifier's exact expected format.
"""

import contextlib
import io
import json
import os
import re
import sys

THOUGHT_DELIMITER_END = "</think>"

_DEFAULT_SYNLOGIC_ROOT = os.environ.get(
    "SYNLOGIC_ROOT",
    "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/SynLogic",
)

_verifier_classes = None
_Data = None


@contextlib.contextmanager
def _maybe_silence():
    """Suppress the verifiers' noisy stdout/stderr unless SYNLOGIC_VERBOSE=1.

    The SynLogic verifiers print a lot of Chinese debug text per sample
    ("验证成功", "检查区域 1...", "无法提取答案"). At RL scale that floods the
    training logs, so we redirect both streams to a throwaway buffer by default.
    """
    if os.environ.get("SYNLOGIC_VERBOSE", "0") == "1":
        yield
        return
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _ensure_synlogic_imported():
    """Lazily import SynLogic's verifier registry and Data class."""
    global _verifier_classes, _Data
    if _verifier_classes is not None and _Data is not None:
        return
    root = _DEFAULT_SYNLOGIC_ROOT
    if root not in sys.path:
        sys.path.insert(0, root)
    from base.data import Data  # noqa: E402
    from task2verifier import verifier_classes  # noqa: E402

    _verifier_classes = verifier_classes
    _Data = Data


def _normalize_data_source(data_source: str) -> str:
    if data_source is None:
        return data_source
    for prefix in ("val/", "train/", "test/"):
        if data_source.startswith(prefix):
            return data_source[len(prefix):]
    return data_source


def _extract_answer(text: str):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_solution_with_thought(solution_str: str) -> str:
    model_output = solution_str
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    predict_answer = _extract_answer(model_output)
    return predict_answer if predict_answer is not None else model_output


def _lenient_answer_candidates(answer: str):
    """Wrap the model's answer in the various format shells SynLogic verifiers expect.

    Different SynLogic ``extract_answer`` implementations require a specific
    wrapper before they will read the answer at all: ``boolean_expressions``
    and most math tasks only read ``\\boxed{...}``; the grid puzzles
    (``star_placement``, etc.) only read a fenced ``​```python``` block; a few
    read ``[[...]]``. A model that solved the puzzle but emitted the bare answer
    (e.g. ``A, B, E`` instead of ``\\boxed{A, B, E}``) would otherwise score 0.

    We try the raw answer plus each shell and let the caller take the MAX score.
    Crucially this only varies the *wrapper*, never the content, so a wrong
    answer stays wrong under every shell -> still 0.0. No reward hacking risk.
    """
    a = (answer or "").strip()
    if not a:
        return [a]
    cands = [a]
    # Only add shells the answer doesn't already carry, to avoid e.g. nesting
    # \boxed{} inside \boxed{} which would break bracket-matching extractors.
    if "\\boxed{" not in a:
        cands.append(f"\\boxed{{{a}}}")
    if not (a.startswith("[[") and a.endswith("]]")):
        cands.append(f"[[{a}]]")
    if "```" not in a:
        cands.append(f"```python\n{a}\n```")
        cands.append(f"```json\n{a}\n```")
    return cands


def _grid_dims(game_data):
    """Pull (n, m) grid dimensions from a SynLogic Data object's metadata, if any."""
    try:
        md = getattr(game_data, "metadata", None) or {}
        n = md.get("n")
        m = md.get("m")
        if isinstance(n, int) and isinstance(m, int) and n > 0 and m > 0:
            return n, m
        # Fall back to inferring from the initial grid if n/m absent.
        grid = md.get("grid")
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            return len(grid), len(grid[0])
    except Exception:
        pass
    return None, None


def _ascii_grid_candidate(answer: str, game_data):
    """Recover a grid puzzle's answer from a plain-text / ASCII-table response.

    Grid tasks (campsite, ...) require a Python 2D list ``[['T','C',...],...]``,
    but a model that solved the puzzle sometimes "draws" the grid as an ASCII
    table instead. We scan the text line by line, keep the cell tokens
    (single letters like T/C/X, standalone), and accept a line only if it has
    exactly ``m`` tokens; if at least ``n`` such lines exist we take the last
    ``n`` (skipping any echoed input grid above the answer) and serialize them
    as a Python list string for the verifier.

    This is SAFE for reward: grid verifiers re-check every puzzle constraint
    against whatever grid we hand them, so a mis-parsed / wrong grid simply
    fails the rules and scores 0. We never compare against a stored solution.
    Returns a ``[[...]]`` string, or None if no clean n×m grid can be recovered.
    """
    n, m = _grid_dims(game_data)
    if not n or not m:
        return None
    rows = []
    for line in (answer or "").splitlines():
        # Standalone single-letter cell tokens (not part of a longer word).
        toks = re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", line)
        if len(toks) == m:
            rows.append([t.upper() for t in toks])
    if len(rows) < n:
        return None
    grid = rows[-n:]  # last n aligned rows = the answer, not the echoed input
    return json.dumps(grid)


def _format_reward(response: str) -> float:
    """Strict SynLogic format check: exactly one think block then one answer block."""
    if (
        response.strip().startswith("<think>")
        and response.strip().endswith("</answer>")
        and response.count("<think>") == 1
        and response.count("</think>") == 1
        and response.count("<answer>") == 1
        and response.count("</answer>") == 1
    ):
        return 1.0
    return 0.0


def _get_game_data_str(extra_info, ground_truth) -> str:
    if extra_info:
        gds = extra_info.get("game_data_str")
        if gds:
            return gds
    # Fallback: synthesize a minimal Data payload from the ground_truth answer.
    return json.dumps({"question": "", "answer": ground_truth, "difficulty": 1, "metadata": {}})


# Tasks whose SynLogic verifier cannot, by itself, confirm the answer is correct
# and therefore need an extra check in this wrapper. Currently only mathador.
_MATHADOR_TASKS = {"mathador", "game_of_24"}


def _mathador_result_ok(answer: str, game_data) -> bool:
    """Confirm a mathador / game-of-24 expression actually equals the target.

    SynLogic's GameOf24Verifier has an upstream bug: it overwrites the target
    ``result`` with ``eval(test_answer)`` and then checks ``abs(result-result)``,
    which is always 0 -> ANY syntactically valid expression using the allowed
    numbers scores 1.0 even if it does not equal the target. That is a
    reward-hacking hole. We re-evaluate the model's expression here and require
    it to equal ``metadata["result"]`` within a small tolerance. The verifier's
    own number/operator-legality checks still run first; this only adds the
    missing "does it equal the target" check.
    """
    try:
        md = getattr(game_data, "metadata", None) or {}
        target = md.get("result")
        if target is None:
            return True  # nothing to check against; don't override verifier
        expr = (answer or "").strip()
        # Strip a fenced code block if the model wrapped the expression.
        m = re.search(r"```(?:python)?\s*(.*?)\s*```", expr, re.DOTALL)
        if m:
            expr = m.group(1).strip()
        # Only allow an arithmetic expression: digits, operators, parens, dots.
        if not expr or not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return False
        value = eval(expr, {"__builtins__": {}}, {})  # safe: charset-restricted above
        return abs(float(value) - float(target)) < 1e-6
    except Exception:
        return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  timeout_score: float = 0, timeout: float = 30.0):
    del timeout_score, timeout

    response = "" if solution_str is None else str(solution_str)

    try:
        _ensure_synlogic_imported()
    except Exception as e:  # pragma: no cover - environment misconfiguration
        print(f"[synlogic_verify] failed to import SynLogic from {_DEFAULT_SYNLOGIC_ROOT}: {e}")
        return 0.0

    task = _normalize_data_source(data_source)
    verifier_cls = _verifier_classes.get(task)
    if verifier_cls is None:
        print(f"[synlogic_verify] no verifier for data_source={data_source!r} (task={task!r})")
        return 0.0

    game_data_str = _get_game_data_str(extra_info, ground_truth)

    require_format = os.environ.get("SYNLOGIC_REQUIRE_FORMAT", "0") == "1"
    format_res = _format_reward(response) if require_format else 1.0

    lenient = os.environ.get("SYNLOGIC_LENIENT_FORMAT", "1") == "1"

    accuracy_res = 0.0
    if format_res > 0:
        try:
            game_data = _Data.from_json_str(game_data_str)
            answer = _extract_solution_with_thought(response)
            # In lenient mode, retry the answer under each format shell the
            # verifiers might require (\boxed{}, ```python```, [[...]], ...) and
            # keep the best score. Only the wrapper changes, never the content,
            # so a wrong answer can never be turned into a correct one this way.
            candidates = _lenient_answer_candidates(answer) if lenient else [answer]
            # Grid puzzles want a Python 2D list; if the model drew an ASCII
            # table instead, recover it as a last-resort candidate. Safe because
            # the verifier re-checks all constraints (a wrong grid still -> 0).
            if lenient:
                ascii_grid = _ascii_grid_candidate(answer, game_data)
                if ascii_grid is not None and ascii_grid not in candidates:
                    candidates.append(ascii_grid)
            for cand in candidates:
                verifier = verifier_cls()
                with _maybe_silence():
                    verdict = verifier.verify(game_data, cand)
                score = float(verdict) if verdict else 0.0
                # Patch the upstream mathador bug: the verifier passes any
                # number/operator-legal expression, so additionally require the
                # expression to actually equal the target before awarding reward.
                if score >= 1.0 and task in _MATHADOR_TASKS:
                    if not _mathador_result_ok(cand, game_data):
                        score = 0.0
                if score > accuracy_res:
                    accuracy_res = score
                if accuracy_res >= 1.0:
                    break
        except Exception as e:
            print(f"[synlogic_verify] verify error for task={task!r}: {e}")
            accuracy_res = 0.0

    return float(accuracy_res * format_res)
