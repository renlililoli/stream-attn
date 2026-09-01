from __future__ import annotations

from contextlib import suppress

import torch


class RecomputeWorkspace:
    """Persistent hidden staging used by large-tile Q/KV recomputation."""

    def __init__(
        self,
        *,
        hidden_features: int,
        staging_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.hidden_features = hidden_features
        self.staging_tokens = staging_tokens
        self.hidden = torch.empty(
            (staging_tokens, hidden_features),
            dtype=dtype,
            device=device,
        )
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.hidden_ready = torch.cuda.Event()
        self.hidden_free = torch.cuda.Event()
        self.hidden_has_pending_compute = False

    def recover(self) -> None:
        with suppress(Exception):
            self.h2d_stream.synchronize()
        self.hidden_has_pending_compute = False


class CrossRecomputeWorkspace:
    """Independent query and context staging for projected cross-attention."""

    def __init__(
        self,
        *,
        query_hidden_features: int,
        context_hidden_features: int,
        q_staging_tokens: int,
        kv_staging_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.query = RecomputeWorkspace(
            hidden_features=query_hidden_features,
            staging_tokens=q_staging_tokens,
            dtype=dtype,
            device=device,
        )
        self.context = RecomputeWorkspace(
            hidden_features=context_hidden_features,
            staging_tokens=kv_staging_tokens,
            dtype=dtype,
            device=device,
        )


__all__ = ["CrossRecomputeWorkspace", "RecomputeWorkspace"]
