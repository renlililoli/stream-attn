"""Pinned SageAttention3 NVFP4 partial attention with valid, comparable FP32 LSE."""

from __future__ import annotations

import importlib
import math
from functools import lru_cache

import torch

from ..kernels import initialize_split_attention_state, merge_split_attention_state

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


@lru_cache(maxsize=1)
def load_sage3():
    api = importlib.import_module("sageattn3.api")
    extension = importlib.import_module("fp4attn_cuda")
    if getattr(extension, "seqattn_lse_abi", lambda: 0)() != 1:
        raise ImportError("Sage3 requires the SeqAttn LSE patch shipped in Dockerfile.cu13")
    return api, extension


def sage3_is_available():
    if triton is None:
        return False
    try:
        load_sage3()
    except (ImportError, OSError):
        return False
    return True


if triton is not None:

    @triton.jit
    def _restore_key_mean_lse(
        q,
        mean,
        lse,
        q_stride,
        h_stride,
        TOKENS: tl.constexpr,
        HEADS: tl.constexpr,
        PADDED: tl.constexpr,
        DIM: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tokens, heads = rows // HEADS, rows % HEADS
        dims = tl.arange(0, DIM)
        query = tl.load(
            q + tokens[:, None] * q_stride + heads[:, None] * h_stride + dims[None, :],
            tokens[:, None] < TOKENS,
            0,
        ).to(tl.float32)
        mu = tl.load(mean + heads[:, None] * DIM + dims[None, :]).to(tl.float32)
        bias = tl.sum(query * mu, 1) * SCALE
        ptr = lse + heads * PADDED + tokens
        old = tl.load(ptr, tokens < TOKENS, 0)
        tl.store(ptr, old + bias, tokens < TOKENS)


def workspace_bound_bytes(q_tokens, kv_tokens, heads, head_dim, element_bytes):
    """Conservative simultaneous bound on vendor packing, padding and output tensors.

    This reservation is separate from persistent core tensors. It deliberately
    covers overlapping Python locals and allocator alignment rather than treating
    external quantization/output storage as free.
    """
    q = (q_tokens + 127) // 128 * 128
    k = (kv_tokens + 127) // 128 * 128
    qe, ke = q * heads * head_dim, k * heads * head_dim
    half = element_bytes * (5 * qe + 6 * ke + qe // 128 + heads * head_dim)
    packed = (9 * (qe + 2 * ke) + 15) // 16
    correction = 4 * (heads * (q // 128) * k + 2 * heads * q)
    return half + packed + correction + 2 * 2**20


class Sage3State:
    """Single-flight query preparation; K/V partitions never mutate caller tensors."""

    def __init__(self):
        self.api, self.extension = load_sage3()
        self.query = self.q_fp4 = self.q_scale = self.q_mean = None

    def reset(self):
        self.query = self.q_fp4 = self.q_scale = self.q_mean = None

    @staticmethod
    def _pad(x):
        pad = (-x.shape[2]) % 128
        return torch.nn.functional.pad(x, (0, 0, 0, pad)) if pad else x

    def prepare_query(self, q):
        self.query = q
        dense = self._pad(q.unsqueeze(0).transpose(1, 2).contiguous())
        centered, self.q_mean = self.api.triton_group_mean(dense)
        self.q_fp4, self.q_scale = self.api.scale_and_quant_fp4(centered)

    def partial(self, k, v, softmax_scale):
        q = self.query
        if not math.isfinite(softmax_scale) or softmax_scale < 0:
            raise ValueError("Sage3 softmax scale must be finite and nonnegative")
        if q is None:
            raise RuntimeError("Sage3 query was not prepared")
        # clone, not just contiguous(): single-token/head views can otherwise alias K.
        dense_k = k.unsqueeze(0).transpose(1, 2).clone(memory_format=torch.contiguous_format)
        key_mean = dense_k.mean(dim=2, keepdim=True)
        dense_k.sub_(key_mean)
        dense_k = self._pad(dense_k)
        dense_v = self._pad(v.unsqueeze(0).transpose(1, 2).contiguous())
        if q.dtype == torch.float16:
            correction = torch.matmul(
                self.q_mean.float(), dense_k.transpose(-2, -1).float()
            ).contiguous()
        else:
            correction = torch.matmul(self.q_mean, dense_k.transpose(-2, -1)).float().contiguous()
        k_fp4, k_scale = self.api.scale_and_quant_fp4_permute(dense_k)
        v_fp4, v_scale = self.api.scale_and_quant_fp4_transpose(dense_v)
        output, lse = self.extension.fwd(
            self.q_fp4,
            k_fp4,
            v_fp4,
            self.q_scale,
            k_scale,
            v_scale,
            correction,
            k.shape[0],
            None,
            softmax_scale,
            False,
            True,
            q.dtype == torch.bfloat16,
        )
        # Sage3 centers K. Its LSE must be restored to the original score origin
        # before combining partitions with different K means.
        _restore_key_mean_lse[(triton.cdiv(q.shape[0] * q.shape[1], 32),)](
            q,
            key_mean,
            lse,
            q.stride(0),
            q.stride(1),
            TOKENS=q.shape[0],
            HEADS=q.shape[1],
            PADDED=output.shape[2],
            DIM=q.shape[2],
            SCALE=softmax_scale,
            BLOCK=32,
        )
        return output[:, :, : q.shape[0]].transpose(1, 2).contiguous(), lse[
            :, :, : q.shape[0]
        ].contiguous()

    def update(self, k, v, accumulator, state_lse, *, softmax_scale, initialize):
        output, lse = self.partial(k, v, softmax_scale)
        combine = initialize_split_attention_state if initialize else merge_split_attention_state
        combine(output, lse, accumulator.unsqueeze(0), state_lse.unsqueeze(0))
