"""Mirror H3DeviceOutputConsumer's carry/emit/finish state machine."""

from __future__ import annotations


class H3Consumer:
    def __init__(self, graph, callbacks):
        self.g, self.callbacks = graph, callbacks
        self.carry_tokens = 0
        self.carry_start = 0
        self.output_index = 0
        self.output_free = [None] * graph.config.num_output_buffers
        self.ffn_ranges = []
        self.ffn_sources = []
        self.cross_q_boundaries = 0

    def emit(self, source, start, stop):
        g, s = self.g, self.g.shape
        tokens = stop - start
        slot = self.output_index % len(self.output_free)
        prefix = f"ffn{self.output_index}"
        # The real consumer waits for output-slot D2H before calling ops.ffn.
        g.milestone(f"{prefix}.slot_free", (self.output_free[slot],), stream="compute")
        self.callbacks.ffn(prefix, source, tokens, start, stop)
        destination = f"output.{slot}"
        packed = g.copy(
            f"{prefix}.output_copy",
            "d2d",
            tokens,
            tokens * s.hidden_features * s.element_bytes,
            source,
            destination,
        )
        self.output_free[slot] = g.copy(
            f"{prefix}.output_d2h",
            "d2h",
            tokens,
            tokens * s.hidden_features * s.element_bytes,
            destination,
            "hidden.source" if g.config.execution_mode == "materialized" else "hidden.output",
            dependencies=(packed,),
            stream="output.d2h",
            component="output transfer",
        )
        self.ffn_ranges.append((start, stop))
        self.ffn_sources.append({"source": source, "slot": slot, "start": start, "stop": stop})
        self.output_index += 1
        return packed

    def consume(self, prefix, start, stop):
        g, s = self.g, self.g.shape
        post, done = self.callbacks.epilogue(prefix, stop - start, start, stop)
        cursor = 0
        chunk = g.config.ffn_tile_tokens
        if self.carry_tokens:
            self.cross_q_boundaries += 1
            take = min(chunk - self.carry_tokens, stop - start)
            done = g.copy(
                f"{prefix}.carry_append",
                "d2d",
                take,
                take * s.hidden_features * s.element_bytes,
                post,
                "ffn.carry",
                details=f"carry_offset={self.carry_tokens}; source_offset=0",
            )
            self.carry_tokens += take
            cursor += take
            if self.carry_tokens == chunk:
                done = self.emit("ffn.carry", self.carry_start, self.carry_start + chunk)
                self.carry_tokens = 0
        while stop - start - cursor >= chunk:
            tile_start = start + cursor
            done = self.emit(post, tile_start, tile_start + chunk)
            cursor += chunk
        if cursor < stop - start:
            remaining = stop - start - cursor
            self.carry_start = start + cursor
            done = g.copy(
                f"{prefix}.carry_tail",
                "d2d",
                remaining,
                remaining * s.hidden_features * s.element_bytes,
                post,
                "ffn.carry",
                details=f"carry_offset=0; source_offset={cursor}",
            )
            self.carry_tokens = remaining
        # Views passed to FFN do not allocate. The post-attention parent survives
        # the complete consumer call, even when only its final tail was copied.
        returned = g.milestone(f"{prefix}.consumer_return", (done,), stream="compute")
        g.retain(post, returned)
        return returned

    def finish(self):
        if self.carry_tokens:
            self.emit("ffn.carry", self.carry_start, self.carry_start + self.carry_tokens)
            self.carry_tokens = 0
        return self.g.milestone(
            "consumer.synchronize", (*self.output_free, self.g.streams.get("compute"))
        )
