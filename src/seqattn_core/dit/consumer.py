from __future__ import annotations

import torch

from ..stats import H3DiTStats
from .types import H3BlockOps
from .workspace import H3BlockWorkspace


class H3DeviceOutputConsumer:
    def __init__(self, workspace: H3BlockWorkspace) -> None:
        self.workspace = workspace
        self.hidden_host: torch.Tensor | None = None
        self.ops: H3BlockOps | None = None
        self.stats: H3DiTStats | None = None
        self.total_tokens = 0
        self.next_token = 0
        self.carry_start = 0
        self.carry_tokens = 0
        self.output_index = 0

    def reset(
        self,
        *,
        hidden_host: torch.Tensor,
        ops: H3BlockOps,
        stats: H3DiTStats,
    ) -> None:
        self.hidden_host = hidden_host
        self.ops = ops
        self.stats = stats
        self.total_tokens = hidden_host.shape[0]
        self.next_token = 0
        self.carry_start = 0
        self.carry_tokens = 0
        self.output_index = 0

    def _validate_device_tile(
        self,
        tensor: torch.Tensor,
        *,
        tokens: int,
        name: str,
    ) -> None:
        workspace = self.workspace
        expected = (tokens, workspace.hidden_features)
        if tensor.shape != expected:
            raise ValueError(f"{name} returned shape {tuple(tensor.shape)}, expected {expected}")
        if tensor.device != workspace.device:
            raise ValueError(f"{name} must return a tensor on {workspace.device}")
        if tensor.dtype != workspace.dtype:
            raise ValueError(f"{name} output dtype must be {workspace.dtype}")

    def _emit(self, tile: torch.Tensor, start: int, stop: int) -> None:
        assert self.hidden_host is not None
        assert self.ops is not None
        assert self.stats is not None
        workspace = self.workspace
        tokens = stop - start
        compute_stream = torch.cuda.current_stream(workspace.device)
        slot_index = self.output_index % len(workspace.final_output)
        if workspace.output_pending[slot_index]:
            compute_stream.wait_event(workspace.output_free[slot_index])

        result = self.ops.mlp(tile, start, stop)
        self._validate_device_tile(result, tokens=tokens, name="mlp")
        final_slot = workspace.final_output[slot_index][:tokens]
        final_slot.copy_(result)
        workspace.output_ready[slot_index].record(compute_stream)

        with torch.cuda.stream(workspace.d2h_stream):
            workspace.d2h_stream.wait_event(workspace.output_ready[slot_index])
            self.hidden_host[start:stop].copy_(
                final_slot,
                non_blocking=self.hidden_host.is_pinned(),
            )
            workspace.output_free[slot_index].record(workspace.d2h_stream)
        workspace.output_pending[slot_index] = True
        self.output_index += 1
        self.stats.mlp_chunks += 1
        self.stats.final_hidden_d2h_bytes += tokens * workspace.hidden_features * tile.element_size()

    def __call__(self, attention: torch.Tensor, start: int, stop: int) -> None:
        assert self.ops is not None
        assert self.stats is not None
        if start != self.next_token or stop <= start:
            raise ValueError(
                f"attention output ranges must be contiguous, got [{start}, {stop}) "
                f"after token {self.next_token}"
            )
        tokens = stop - start
        post_attention = self.ops.attention_epilogue(attention, start, stop)
        self._validate_device_tile(
            post_attention,
            tokens=tokens,
            name="attention_epilogue",
        )

        chunk = self.workspace.mlp_chunk_tokens
        cursor = 0
        if self.carry_tokens:
            self.stats.mlp_cross_q_boundaries += 1
            take = min(chunk - self.carry_tokens, tokens)
            self.workspace.carry[self.carry_tokens : self.carry_tokens + take].copy_(
                post_attention[:take]
            )
            self.carry_tokens += take
            cursor += take
            if self.carry_tokens == chunk:
                self._emit(
                    self.workspace.carry,
                    self.carry_start,
                    self.carry_start + chunk,
                )
                self.carry_tokens = 0

        while tokens - cursor >= chunk:
            tile_start = start + cursor
            self._emit(
                post_attention[cursor : cursor + chunk],
                tile_start,
                tile_start + chunk,
            )
            cursor += chunk

        if cursor < tokens:
            remaining = tokens - cursor
            self.carry_start = start + cursor
            self.workspace.carry[:remaining].copy_(post_attention[cursor:])
            self.carry_tokens = remaining
        self.next_token = stop

    def finish(self) -> None:
        if self.next_token != self.total_tokens:
            raise ValueError(
                f"attention consumer received {self.next_token} of {self.total_tokens} tokens"
            )
        if self.carry_tokens:
            self._emit(
                self.workspace.carry[: self.carry_tokens],
                self.carry_start,
                self.carry_start + self.carry_tokens,
            )
            self.carry_tokens = 0

    def synchronize(self) -> None:
        self.workspace.d2h_stream.synchronize()


__all__ = ["H3DeviceOutputConsumer"]
