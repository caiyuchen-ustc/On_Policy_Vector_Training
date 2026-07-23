"""IFEvalG instruction-following scorer (ported from M2RL / Google Research IFEval).

This package is a self-contained copy of M2RL's
``slime/slime/rewards/IFEvalG`` (the official IFEval implementation extended
with IFBench constraints), with its internal imports rewritten to package-local
relative imports so it works inside G-OPD (verl).

The public entry points used by ``verl.utils.reward_score.ifevalg_verify`` are
``InputExample`` and ``test_instruction_following_strict`` from
``evaluation_main``. Scoring is strict and binary: a response scores 1 only if
it follows *every* instruction in ``instruction_id_list``, else 0.
"""

from .evaluation_main import (
    InputExample,
    OutputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

__all__ = [
    "InputExample",
    "OutputExample",
    "test_instruction_following_strict",
    "test_instruction_following_loose",
]
