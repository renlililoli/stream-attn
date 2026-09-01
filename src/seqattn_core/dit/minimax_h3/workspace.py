from __future__ import annotations

import torch


class H3BlockWorkspace:
    def __init__(
        self,
        *,
        hidden_features: int,
        ffn_tile_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        num_final_output_buffers: int = 2,
        final_output_chunk_tokens: int | None = None,
    ) -> None:
        final_output_chunk_tokens = (
            ffn_tile_tokens if final_output_chunk_tokens is None else final_output_chunk_tokens
        )
        if final_output_chunk_tokens <= 0:
            raise ValueError("final_output_chunk_tokens must be positive")
        self.hidden_features = hidden_features
        self.ffn_tile_tokens = ffn_tile_tokens
        self.final_output_chunk_tokens = final_output_chunk_tokens
        self.dtype = dtype
        self.device = device
        self.carry = torch.empty(
            (ffn_tile_tokens, hidden_features),
            dtype=dtype,
            device=device,
        )
        self.final_output = [
            torch.empty(
                (final_output_chunk_tokens, hidden_features),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_final_output_buffers)
        ]
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.output_ready = [torch.cuda.Event() for _ in self.final_output]
        self.output_free = [torch.cuda.Event() for _ in self.final_output]
        self.output_pending = [False for _ in self.final_output]
        self.task_d2h_start = torch.cuda.Event(enable_timing=True)
        self.task_done = torch.cuda.Event(enable_timing=True)


__all__ = ["H3BlockWorkspace"]
