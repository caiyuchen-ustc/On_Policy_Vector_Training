# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline teacher rollout for distillation.

Instead of generating rollouts on-the-fly with vLLM, this loads pre-generated teacher
responses from a parquet (produced by gen_teacher_rollouts.sh + merge_rollouts_to_sft.py:
columns `prompt` [chat list] + `response` [teacher text]) and returns them as the rollout
batch. It patches `actor_rollout_wg.generate_sequences` on the driver (same mechanism as
verl.utils.rollout_skip.RolloutSkip), building the exact rollout DataProto layout that the
vLLM rollout produces (prompts / responses / input_ids / attention_mask / position_ids), so
the downstream PPO loop (student old_log_prob -> teacher ref_log_prob -> reverse-KL advantage)
is unchanged.
"""

import json
from collections import defaultdict

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


def _to_plain(obj):
    """Recursively convert numpy arrays / scalars in a chat message to plain python types
    so that json serialization is stable and matches across dataset and parquet."""
    if isinstance(obj, np.ndarray):
        return [_to_plain(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def canonical_prompt_key(messages) -> str:
    """Stable string key for a chat prompt (list of {role, content} dicts)."""
    plain = _to_plain(messages)
    return json.dumps(plain, sort_keys=True, ensure_ascii=False)


class OfflineTeacherRollout:
    print_mark = "[OfflineTeacherRollout()]"

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        rollout_cfg = config.actor_rollout_ref.rollout

        self.data_path = rollout_cfg.get("offline_teacher_data_path", None)
        if not self.data_path:
            raise ValueError(
                "offline_teacher_rollout=True requires rollout.offline_teacher_data_path to be set"
            )
        self.prompt_key = config.data.get("prompt_key", "prompt")
        self.response_key = rollout_cfg.get("offline_teacher_response_key", "response")
        self.response_length = int(rollout_cfg.response_length)
        self.truncation = config.data.get("truncation", "right")

        import pandas as pd

        local_path = copy_to_local(self.data_path)
        df = pd.read_parquet(local_path)
        if self.prompt_key not in df.columns or self.response_key not in df.columns:
            raise KeyError(
                f"{self.data_path} must contain columns '{self.prompt_key}' and "
                f"'{self.response_key}', found {list(df.columns)}"
            )

        # Build {canonical_prompt_key: [teacher_response_str, ...]}.
        self.prompt_to_responses = defaultdict(list)
        for _, row in df.iterrows():
            key = canonical_prompt_key(row[self.prompt_key])
            resp = row[self.response_key]
            self.prompt_to_responses[key].append("" if resp is None else str(resp))

        # Round-robin cursor per prompt so repeated epochs / multiple samples cycle responses.
        self._cursor = defaultdict(int)
        # Original generate_sequences, set by wrap_generate_sequences; used for validation batches.
        self._orig_generate_sequences = None
        total_pairs = sum(len(v) for v in self.prompt_to_responses.values())
        print(
            f"{self.print_mark} Loaded {total_pairs} teacher responses for "
            f"{len(self.prompt_to_responses)} unique prompts from {self.data_path}",
            flush=True,
        )

    def _lookup_response(self, messages) -> str:
        key = canonical_prompt_key(messages)
        responses = self.prompt_to_responses.get(key, None)
        if not responses:
            preview = canonical_prompt_key(messages)[:300]
            raise KeyError(
                f"{self.print_mark} No offline teacher response found for prompt (preview): {preview}"
            )
        idx = self._cursor[key] % len(responses)
        self._cursor[key] += 1
        return responses[idx]

    def _encode_response(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Ensure the response ends with EOS so get_response_mask terminates the sequence.
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            if len(ids) >= self.response_length:
                if self.truncation == "left":
                    ids = ids[-(self.response_length - 1):]
                else:  # right / middle / error -> right-truncate to leave room for eos
                    ids = ids[: self.response_length - 1]
            ids = ids + [eos_id]
        else:
            ids = ids[: self.response_length]
        return ids

    def generate_sequences(self, gen_batch: DataProto, **kwargs) -> DataProto:
        """Build the rollout DataProto from offline teacher responses.

        Mirrors the field layout of vLLMRollout.generate_sequences
        (verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py).

        Validation batches (meta_info["validate"]=True) are delegated back to the real
        vLLM generation: validation prompts were never sampled by the teacher, and we want
        the student to actually generate so IFEvalG reward reflects the current policy.
        """
        if gen_batch.meta_info.get("validate", False):
            if self._orig_generate_sequences is None:
                raise RuntimeError(
                    f"{self.print_mark} validation batch received but original generate_sequences "
                    "was not saved; call wrap_generate_sequences() first."
                )
            return self._orig_generate_sequences(gen_batch, **kwargs)

        idx = gen_batch.batch["input_ids"]  # (bs, prompt_length), left-padded
        attention_mask = gen_batch.batch["attention_mask"]
        position_ids = gen_batch.batch["position_ids"]
        batch_size = idx.size(0)

        raw_prompts = gen_batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            raise KeyError(
                f"{self.print_mark} gen_batch has no 'raw_prompt'; set data.return_raw_chat=True"
            )

        eos_token_id = gen_batch.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)

        response_ids_list = [self._encode_response(self._lookup_response(raw_prompts[i])) for i in range(batch_size)]

        response = pad_2d_list_to_length(
            response_ids_list, self.tokenizer.pad_token_id, max_length=self.response_length
        ).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (bs, 4, seq_len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        # The trainer reads gen_batch_output.meta_info["timing"] right after generate_sequences.
        meta_info = dict(gen_batch.meta_info)
        meta_info.setdefault("timing", {})
        return DataProto(batch=batch, non_tensor_batch=gen_batch.non_tensor_batch, meta_info=meta_info)

    def wrap_generate_sequences(self, rollout_wg):
        """Patch the worker group's generate_sequences to serve offline teacher rollouts.

        The original method is kept so validation batches can still generate with vLLM.
        """
        self._orig_generate_sequences = rollout_wg.generate_sequences
        rollout_wg.generate_sequences = self.generate_sequences
        print(
            f"{self.print_mark} Patched actor_rollout_wg.generate_sequences() with offline teacher rollouts "
            f"(validation batches still use real generation)",
            flush=True,
        )
