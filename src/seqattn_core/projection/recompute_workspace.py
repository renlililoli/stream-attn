from __future__ import annotations

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


__all__ = ["RecomputeWorkspace"]
