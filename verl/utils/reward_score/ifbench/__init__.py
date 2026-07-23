"""IFBench instruction-following scorer (ported from allenai/IFBench).

Self-contained copy of the AllenAI IFBench ``evaluation_lib`` + instruction
checkers (https://github.com/allenai/IFBench), with the internal module imports
rewritten to package-local relative imports so it works inside G-OPD (verl).

This package covers the 58 IFBench-specific constraint families
(``count:``/``ratio:``/``words:``/``format:``/``sentence:``/``repeat:``/
``custom:``) used by the ``allenai/IFBench_test`` validation set, which the
smaller training-time ``ifevalg`` scorer does not implement.

Public entry points used by ``verl.utils.reward_score.ifevalg_verify``:
``InputExample`` and ``test_instruction_following_strict`` /
``test_instruction_following_loose``. Note IFBench's strict/loose checkers take
``(inp, prompt_to_response_dict)`` rather than ``(inp, response)``.

Extra dependencies beyond the ``ifevalg`` package: ``emoji``, ``syllapy``
(``pip install emoji "syllapy" setuptools<81``). NLTK punkt/punkt_tab/stopwords/
averaged_perceptron_tagger are auto-downloaded on import via certifi.
"""

from .evaluation_lib import (
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
