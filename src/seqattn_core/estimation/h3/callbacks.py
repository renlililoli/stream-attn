"""No-framework H3 callback graph: normalization, QKV, RoPE, epilogue and SwiGLU.

Implementation-private workspaces (including quantization/ConvRot/LoRA) belong
in operator profiles. The default topology corresponds to no-LoRA callbacks.
"""

from __future__ import annotations

from .linear_memory import eager_int8_workspace


class H3Callbacks:
    def __init__(self, graph):
        self.g = graph
        self.s, self.c = graph.shape, graph.callbacks
        self.modulation = None
        self.positions = None

    def prepare(self):
        g, s = self.g, self.s
        if self.c.variant != "modulated":
            return
        self.positions = g.allocate(
            "position_ids",
            s.tokens * 3 * 8,
            "position metadata",
            host=True,
            owner="caller",
            persistent=True,
        )
        size = 6 * s.modulation_rows * s.hidden_features * s.modulation_element_bytes
        if self.c.compute_modulation:
            embedding_rows = s.modulation_rows // s.modulation_modalities
            time_input = g.allocate(
                "timestep.embedding",
                embedding_rows * s.timestep_features * 4,
                "conditioning",
                owner="caller",
                persistent=True,
            )
            self.modulation = g.allocate("adaln.modulation", size, "modulation")
            g.op(
                "adaln.projection",
                "adaln",
                embedding_rows,
                2 * s.modulation_rows * s.timestep_features * 6 * s.hidden_features,
                (time_input, self.modulation),
            )
        else:
            self.modulation = g.allocate(
                "adaln.modulation", size, "modulation", owner="caller", persistent=True
            )

    def normalized(self, prefix, source, tokens, which):
        g, s = self.g, self.s
        result = g.allocate(
            f"{prefix}.norm{which}",
            tokens * s.hidden_features * s.element_bytes,
            "normalized hidden",
        )
        g.op(
            f"{prefix}.norm{which}",
            f"norm{which}",
            tokens,
            tokens * s.hidden_features,
            (source, result),
        )
        if self.c.variant == "modulated":
            g.op(
                f"{prefix}.modulate{which}",
                f"modulate{which}",
                tokens,
                tokens * s.hidden_features,
                (result, self.modulation),
            )
        return result

    def linear(self, prefix, key, source, tokens, in_features, out_features, *, destination=None):
        g = self.g
        result = destination or g.allocate(
            f"{prefix}.result", tokens * out_features * self.s.element_bytes, f"{key} result"
        )
        workspace = 0
        if self.c.linear_memory == "int8_eager":
            workspace = eager_int8_workspace(
                tokens,
                in_features,
                out_features,
                self.s.element_bytes,
                convrot_group=self.c.convrot_group,
                per_channel_scale=self.c.per_channel_weight_scale,
            )
        done = g.op(
            prefix,
            key,
            tokens,
            2 * tokens * in_features * out_features,
            (source, result),
            modeled_workspace_bytes=workspace,
        )
        return result, done

    def rope(self, prefix, key, source, tokens, *, inplace):
        g, s = self.g, self.s
        position_gpu = g.allocate(f"{prefix}.positions", tokens * 3 * 4, "position tile")
        position_host = g.allocate(
            f"{prefix}.position_cast", tokens * 3 * 4, "position FP32 cast", host=True
        )
        cast = g.milestone(f"{prefix}.position_cast_ready", stream="compute")
        g.use((self.positions, position_host), cast)
        g.copy(
            f"{prefix}.position_h2d",
            "h2d",
            tokens,
            tokens * 3 * 4,
            position_host,
            position_gpu,
            stream="compute",
            component="position transfer",
        )
        angles = g.allocate(f"{prefix}.angles", tokens * s.rope_dim * 4, "RoPE angles")
        g.op(
            f"{prefix}.angles",
            "rope_angles",
            tokens,
            tokens * max(s.rope_dim, 1),
            (position_gpu, angles),
            modeled_workspace_bytes=tokens * s.rope_dim * 4,
        )
        table = g.allocate(
            f"{prefix}.table", tokens * 2 * s.rope_dim * s.element_bytes, "RoPE table"
        )
        g.op(
            f"{prefix}.table",
            "rope_table",
            tokens,
            tokens * max(s.rope_dim, 1),
            (angles, table),
            modeled_workspace_bytes=tokens * s.rope_dim * 12,
        )
        if inplace:
            done = g.op(prefix, key, tokens, 2 * tokens * s.attention_features, (source, table))
            return source, done
        normalized = g.allocate(
            f"{prefix}.qk_norm", tokens * s.attention_features * s.element_bytes, "QK normalized"
        )
        g.op(
            f"{prefix}.norm", "qk_norm", tokens, tokens * s.attention_features, (source, normalized)
        )
        rotated_bytes = tokens * s.heads * s.rope_dim * s.element_bytes
        rotated = g.allocate(f"{prefix}.rotated", rotated_bytes, "QK rotated")
        g.op(
            f"{prefix}.rotate",
            "rope_rotate",
            tokens,
            tokens * s.attention_features,
            (normalized, table, rotated),
            modeled_workspace_bytes=2 * rotated_bytes,
        )
        result = g.allocate(
            f"{prefix}.normalized_rotated",
            tokens * s.attention_features * s.element_bytes,
            "QK norm/RoPE result",
        )
        done = g.op(
            f"{prefix}.concat",
            "rope_concat",
            tokens,
            tokens * s.attention_features,
            (normalized, rotated, result),
        )
        return result, done

    def project(self, prefix, hidden, tokens, mode, *, destinations=()):
        g, s = self.g, self.s
        normalized = (
            hidden if self.c.variant == "block25" else self.normalized(prefix, hidden, tokens, 1)
        )
        widths = {
            "qkv": 3 * s.attention_features,
            "q": s.attention_features,
            "kv": 2 * s.attention_features,
        }
        direct = self.c.recompute_direct_write and mode != "qkv"
        if direct:
            # A declared fused direct-write projector has no returned Q/KV tensor.
            done = g.op(
                f"{prefix}.{mode}",
                mode,
                tokens,
                2 * tokens * s.hidden_features * widths[mode],
                (normalized, *destinations),
            )
            result = None
        else:
            result, done = self.linear(
                f"{prefix}.{mode}", mode, normalized, tokens, s.hidden_features, widths[mode]
            )
            if self.c.variant == "modulated":
                if mode == "qkv":
                    if not self.c.materialized_qk_inplace:
                        raise ValueError(
                            "materialized out-of-place Q/K callbacks need an explicit custom allocation profile"
                        )
                    _, done = self.rope(
                        f"{prefix}.qk_rope", "qk_rope", result, tokens, inplace=True
                    )
                else:
                    rotated, done = self.rope(
                        f"{prefix}.{mode}_rope",
                        "q_rope" if mode == "q" else "k_rope",
                        result,
                        tokens,
                        inplace=False,
                    )
                    done = g.copy(
                        f"{prefix}.write_{mode[0]}",
                        "d2d",
                        tokens,
                        tokens * s.attention_features * s.element_bytes,
                        rotated,
                        destinations[0],
                    )
                    if mode == "kv":
                        done = g.copy(
                            f"{prefix}.write_v",
                            "d2d",
                            tokens,
                            tokens * s.attention_features * s.element_bytes,
                            result,
                            destinations[1],
                        )
                    g.retain(rotated, done)
            if mode != "qkv" and self.c.variant == "block25":
                for i, destination in enumerate(destinations):
                    done = g.copy(
                        f"{prefix}.write_{i}",
                        "d2d",
                        tokens,
                        tokens * s.attention_features * s.element_bytes,
                        result,
                        destination,
                    )
            if mode != "qkv":
                g.retain(result, done)
        # Local normalized hidden survives until the projection callback returns.
        if normalized != hidden:
            g.retain(normalized, done)
        return result, done

    def epilogue(self, prefix, tokens, start, stop):
        g, s = self.g, self.s
        update, _ = self.linear(
            f"{prefix}.out", "out", "Q", tokens, s.attention_features, s.hidden_features
        )
        post = g.allocate(
            f"{prefix}.post_attention",
            tokens * s.hidden_features * s.element_bytes,
            "post attention",
        )
        g.copy(
            f"{prefix}.residual_h2d",
            "h2d",
            tokens,
            tokens * s.hidden_features * s.element_bytes,
            "hidden.source",
            post,
            stream="compute",
            component="residual transfer",
        )
        done = g.op(
            f"{prefix}.attention_residual",
            "attention_residual",
            tokens,
            tokens * s.hidden_features,
            (post, update, self.modulation),
            details=f"range=[{start},{stop}); in-place residual",
        )
        g.retain(update, done)
        return post, done

    def ffn(self, prefix, source, tokens, start, stop):
        g, s = self.g, self.s
        normalized = self.normalized(prefix, source, tokens, 2)
        fc1, _ = self.linear(
            f"{prefix}.fc1", "fc1", normalized, tokens, s.hidden_features, 2 * s.ffn_features
        )
        if self.c.fused_swiglu_fc2:
            if "swiglu_fc2" not in g.profile.operators:
                raise ValueError("fused SwiGLU/FC2 requires its own timing and workspace profile")
            result = g.allocate(
                f"{prefix}.fc2.result", tokens * s.hidden_features * s.element_bytes, "fc2 result"
            )
            g.op(
                f"{prefix}.swiglu_fc2",
                "swiglu_fc2",
                tokens,
                2 * tokens * s.ffn_features * s.hidden_features,
                (fc1, result),
            )
            activated = None
        else:
            activated = g.allocate(
                f"{prefix}.swiglu.result",
                tokens * s.ffn_features * s.element_bytes,
                "SwiGLU activation",
            )
            g.op(
                f"{prefix}.swiglu",
                "swiglu",
                tokens,
                tokens * s.ffn_features,
                (fc1, activated),
                modeled_workspace_bytes=tokens * s.ffn_features * s.element_bytes,
            )
            result, _ = self.linear(
                f"{prefix}.fc2", "fc2", activated, tokens, s.ffn_features, s.hidden_features
            )
        # fc1(x) is the live argument to linear_input_act until FC2 completes.
        g.retain(fc1, g.streams["compute"])
        if activated:
            g.retain(activated, g.streams["compute"])
        done = g.op(
            f"{prefix}.ffn_residual",
            "ffn_residual",
            tokens,
            tokens * s.hidden_features,
            (source, result, self.modulation),
            details=f"range=[{start},{stop}); return aliases input",
        )
        g.retain(normalized, done)
        g.retain(result, done)
        return done
