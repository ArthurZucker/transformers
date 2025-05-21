# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Flashinfer integration file"""

from typing import Optional, Tuple

import torch

from ..cache_utils import StaticCache

try:
    from flashinfer.decode import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )
except Exception:
    BatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None


def flashinfer_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    past_key_values: Optional[StaticCache] = None,
    is_prefilling: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Wrapper around Flashinfer decode kernels.

    Args:
        module: Attention module using the integration.
        query: Query states of shape ``(batch, num_heads, seq_len, head_dim)``.
        key: Key states.
        value: Value states.
        attention_mask: Optional causal mask.
        past_key_values: ``StaticCache`` instance used to store KV states.
        is_prefilling: Whether we are in prefill phase. If ``True`` the prefill
            wrapper is used, otherwise the decode wrapper is called.
    """
    if BatchDecodeWithPagedKVCacheWrapper is None or BatchPrefillWithPagedKVCacheWrapper is None:
        raise ImportError("flashinfer is not available")

    # Fallback to an empty static cache if none is provided
    if past_key_values is None:
        past_key_values = StaticCache(module.config, query.size(0), query.size(2), device=query.device, dtype=query.dtype)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if is_prefilling:
        wrapper = BatchPrefillWithPagedKVCacheWrapper()
    else:
        wrapper = BatchDecodeWithPagedKVCacheWrapper()

    attn_output = wrapper(
        query,
        key,
        value,
        past_key_values,
        attention_mask=attention_mask,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
