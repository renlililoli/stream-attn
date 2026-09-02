from __future__ import annotations

import math
from itertools import pairwise

import torch

from ..kernels.sol_preprocess import SOL_BLOCK_TOKENS, SOL_HEAD_DIM
from ..validation import validate_cu_seqlens
from .plan import _validate_exact_prefix_tokens


def sol_streaming_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    exact_prefix_tokens: tuple[int, ...],
    tau: float = 1.0,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Small-shape semantic oracle for the streamed Sol block approximation."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise ValueError("q, k, and v must share [tokens, heads, head_dim]")
    if q.shape[-1] != SOL_HEAD_DIM:
        raise ValueError(f"Sol reference requires head_dim={SOL_HEAD_DIM}")
    bounds = validate_cu_seqlens(cu_seqlens, q.shape[0], "cu_seqlens")
    _validate_exact_prefix_tokens(exact_prefix_tokens, bounds)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    route_scale = scale * math.log2(math.e)
    output = torch.empty_like(q)
    for segment_id, (start, stop) in enumerate(pairwise(bounds)):
        if start == stop:
            continue
        qs, ks, vs = q[start:stop], k[start:stop], v[start:stop]
        blocks = math.ceil((stop - start) / SOL_BLOCK_TOKENS)
        k_centroids = []
        value_sums = []
        lengths = []
        for block in range(blocks):
            lo = block * SOL_BLOCK_TOKENS
            hi = min(lo + SOL_BLOCK_TOKENS, stop - start)
            k_centroids.append(ks[lo:hi].float().mean(dim=0).to(q.dtype))
            value_sums.append(vs[lo:hi].float().sum(dim=0).to(q.dtype))
            lengths.append(hi - lo)
        kc = torch.stack(k_centroids).float()
        vc = torch.stack(value_sums).float()
        kc_mean = kc.mean(dim=0)
        kc_variance = kc.var(dim=0, correction=0)
        prefix_blocks = math.ceil(exact_prefix_tokens[segment_id] / SOL_BLOCK_TOKENS)
        for q_block in range(blocks):
            q_lo = q_block * SOL_BLOCK_TOKENS
            q_hi = min(q_lo + SOL_BLOCK_TOKENS, stop - start)
            q_tile = qs[q_lo:q_hi].float()
            q_centroid = q_tile.mean(dim=0)
            threshold = route_scale * (q_centroid * kc_mean).sum(dim=-1) + tau * torch.sqrt(
                route_scale * route_scale * (q_centroid.square() * kc_variance).sum(dim=-1) + 1.0e-6
            )
            rows = []
            for row in q_tile:
                head_rows = []
                for head in range(q.shape[1]):
                    head_logits = []
                    head_values = []
                    for kv_block in range(blocks):
                        route_score = route_scale * (q_centroid[head] * kc[kv_block, head]).sum()
                        exact = (
                            bool((route_score > threshold[head]).item())
                            or abs(q_block - kv_block) <= 1
                            or kv_block < prefix_blocks
                            or q_block < prefix_blocks
                        )
                        kv_lo = kv_block * SOL_BLOCK_TOKENS
                        kv_hi = min(kv_lo + SOL_BLOCK_TOKENS, stop - start)
                        if exact:
                            scores = scale * (ks[kv_lo:kv_hi, head].float() @ row[head])
                            head_logits.extend(scores.unbind())
                            head_values.extend(vs[kv_lo:kv_hi, head].float().unbind())
                        else:
                            score = scale * (kc[kv_block, head] * row[head]).sum()
                            head_logits.extend([score] * lengths[kv_block])
                            average_v = vc[kv_block, head] / lengths[kv_block]
                            head_values.extend([average_v] * lengths[kv_block])
                    probabilities = torch.softmax(torch.stack(head_logits), dim=0)
                    head_rows.append((probabilities[:, None] * torch.stack(head_values)).sum(dim=0))
                rows.append(torch.stack(head_rows))
            output[start + q_lo : start + q_hi].copy_(torch.stack(rows).to(output.dtype))
    return output


__all__ = ["sol_streaming_reference"]
