"""verl multi-turn tool for the M2RL workbench (workplace-assistant) agent task.

Each trajectory gets its own fresh tool environment (email / calendar / analytics /
project-management / CRM toolkits).  The model calls tools one at a time; after all
turns complete we compare the final system state against the ground-truth action
sequence using M2RL's ``is_correct()`` and return 1.0 or 0.0.

Registration (workbench_tool_config.yaml):
    tools:
      - class_name: "verl.tools.workbench_tool.WorkbenchTool"
        config:
          type: native
          m2rl_gym_root: "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/M2RL/M2RL/Gym"
        tool_schema: ...  (27 workbench tools, see workbench_tool_config.yaml)

Data format (extra_info.tools_kwargs):
    "workbench": {
        "create_kwargs": {
            "ground_truth_tool_calls": [{"type":"function_call","name":"...","arguments":{...}}, ...]
        }
    }
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default path to M2RL's Gym directory (overridable via tool config).
_DEFAULT_M2RL_GYM = "/apdcephfs_zwfy3/share_302867165/ewencai/CODE/M2RL/M2RL/Gym"


def _ensure_m2rl_imported(gym_root: str):
    """Add M2RL Gym to sys.path so ``resources_servers.*`` can be imported."""
    if gym_root not in sys.path:
        sys.path.insert(0, gym_root)


_ALL_TOOLKITS = [
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]


def _make_fresh_tool_env(gym_root: str) -> dict:
    """Create a fresh set of workplace-assistant tool instances."""
    _ensure_m2rl_imported(gym_root)
    from resources_servers.workplace_assistant.utils import get_tools  # noqa: E402
    return get_tools(_ALL_TOOLKITS)


def _execute_actions(actions: list[dict], tool_env: dict) -> None:
    """Execute a list of ``{type, name, arguments}`` action dicts against tool_env in-place."""
    functions = tool_env["functions"]
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("type") != "function_call":
            continue
        name = action.get("name", "")
        args = action.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        fn = functions.get(name)
        if fn is None:
            logger.debug("[WorkbenchTool] unknown tool %r — skipping", name)
            continue
        try:
            fn(**args)
        except Exception as exc:
            logger.debug("[WorkbenchTool] tool %r error: %s", name, exc)


def _compare_states(gym_root: str, predicted: list[dict], ground_truth: list[dict]) -> bool:
    """Run both action lists on independent fresh envs and compare final states.

    Uses M2RL's ``is_correct()`` which does DataFrame.equals() on all five
    toolkits (case-insensitive for most fields).
    """
    _ensure_m2rl_imported(gym_root)
    from resources_servers.workplace_assistant.utils import is_correct  # noqa: E402
    return bool(is_correct(predicted, ground_truth, error=None))


class WorkbenchTool(BaseTool):
    """Multi-turn workplace-assistant tool for verl's sglang multi-turn rollout.

    Life-cycle per trajectory:
        create()   → allocate a fresh tool environment, store ground truth
        execute()  → dispatch one tool call; update action history; return text response
        calc_reward() → compare final state with ground truth via is_correct()
        release()  → free the instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._gym_root: str = config.get("m2rl_gym_root", _DEFAULT_M2RL_GYM)
        # The function name of this specific tool instance (e.g. "email_send_email").
        # Derived from the tool schema — each of the 27 workbench tools is a separate
        # WorkbenchTool instance registered in the tool_map with its own name.
        self._tool_function_name: str = (
            tool_schema.function.name if tool_schema and tool_schema.function else ""
        )
        # instance_id → {"tool_env", "ground_truth", "action_history"}
        self._instance_dict: dict[str, dict] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth_tool_calls: Optional[list] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        # Also accept nested create_kwargs (verl passes create_kwargs as a top-level kwarg)
        if ground_truth_tool_calls is None:
            create_kwargs = kwargs.get("create_kwargs", {})
            ground_truth_tool_calls = create_kwargs.get("ground_truth_tool_calls", [])

        if isinstance(ground_truth_tool_calls, str):
            try:
                ground_truth_tool_calls = json.loads(ground_truth_tool_calls)
            except Exception:
                ground_truth_tool_calls = []

        loop = asyncio.get_event_loop()
        tool_env = await loop.run_in_executor(None, _make_fresh_tool_env, self._gym_root)

        self._instance_dict[instance_id] = {
            "tool_env": tool_env,
            "ground_truth": list(ground_truth_tool_calls or []),
            "action_history": [],
        }
        return instance_id, ToolResponse(text="Workplace assistant is ready.")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute one tool call; return (response_text, step_reward=0, metrics).

        Each WorkbenchTool instance corresponds to exactly one of the 27 workbench
        functions (e.g. "email_send_email").  The rollout calls
        ``tool_map[tool_call.function.name].execute(instance_id, arguments)``,
        so we already know which function to call from self._tool_function_name.
        """
        state = self._instance_dict[instance_id]
        tool_env = state["tool_env"]
        functions = tool_env["functions"]

        tool_name = self._tool_function_name

        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except Exception:
                parameters = {}

        # Record the action in the trajectory history for final reward calc.
        action = {
            "type": "function_call",
            "name": tool_name,
            "arguments": dict(parameters),
        }
        state["action_history"].append(action)

        # Execute synchronously in an executor so we don't block the event loop.
        fn = functions.get(tool_name)
        response_text = ""
        if fn is None:
            response_text = f"Unknown tool: {tool_name!r}"
        else:
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, lambda: fn(**parameters))
                response_text = str(result) if result is not None else "Done."
            except Exception as exc:
                response_text = f"Error executing {tool_name}: {exc}"

        # No step-level reward (only final reward matters for workbench).
        return ToolResponse(text=response_text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict[instance_id]
        loop = asyncio.get_event_loop()
        try:
            correct = await loop.run_in_executor(
                None,
                _compare_states,
                self._gym_root,
                state["action_history"],
                state["ground_truth"],
            )
            return 1.0 if correct else 0.0
        except Exception as exc:
            logger.warning("[WorkbenchTool] calc_reward error: %s", exc)
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
