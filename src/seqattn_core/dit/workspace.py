from __future__ import annotations

import torch


class H3BlockWorkspace:
    def __init__(
        self,
        *,
        hidden_features: int,
        mlp_chunk_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        num_final_output_buffers: int = 2,
    ) -> None:
        self.hidden_features = hidden_features
        self.mlp_chunk_tokens = mlp_chunk_tokens
        self.dtype = dtype
        self.device = device
        self.carry = torch.empty(
            (mlp_chunk_tokens, hidden_features),
            dtype=dtype,
            device=device,
        )
        self.final_output = [
            torch.empty_like(self.carry) for _ in range(num_final_output_buffers)
        ]
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.output_ready = [torch.cuda.Event() for _ in self.final_output]
        self.output_free = [torch.cuda.Event() for _ in self.final_output]
        self.output_pending = [False for _ in self.final_output]


__all__ = ["H3BlockWorkspace"]
