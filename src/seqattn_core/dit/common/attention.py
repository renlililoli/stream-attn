from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ...projection import (
    CrossProjection,
    CrossRecomputeProjection,
    ProjectedAttentionRunner,
    ProjectedCrossAttentionRunner,
    RecomputedAttentionRunner,
    RecomputedCrossAttentionRunner,
    SelfProjection,
    SelfRecomputeProjection,
)
from ...stats import (
    ProjectedAttentionStats,
    ProjectedCrossAttentionStats,
    RecomputedAttentionStats,
    RecomputedCrossAttentionStats,
)
from .consumer import AttentionOutputConsumer
from .contracts import AttentionEpilogue, LeaseFactory

MaterializedRunner = ProjectedAttentionRunner | ProjectedCrossAttentionRunner
MaterializedStats = ProjectedAttentionStats | ProjectedCrossAttentionStats


@dataclass(frozen=True)
class MaterializedAttentionBatch:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    projection_wall_seconds: float


def _lease_context(lease: LeaseFactory | None):
    return nullcontext() if lease is None else lease()


class MaterializedAttentionExecutor:
    """Shared materialize/consume mechanics with model order left to the caller."""

    def __init__(self, consumer: AttentionOutputConsumer) -> None:
        self.consumer = consumer

    def materialize_self(
        self,
        runner: ProjectedAttentionRunner,
        hidden_host: torch.Tensor,
        projection: SelfProjection,
        stats: ProjectedAttentionStats,
    ) -> MaterializedAttentionBatch:
        started = time.perf_counter()
        with projection.context():
            q, k, v = runner.project_qkv_to_host(
                hidden_host,
                projection.project_qkv,
                stats=stats,
            )
        stats.raw_attention_roundtrip_bytes_avoided += 2 * q.numel() * q.element_size()
        return MaterializedAttentionBatch(q, k, v, time.perf_counter() - started)

    def materialize_cross(
        self,
        runner: ProjectedCrossAttentionRunner,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        projection: CrossProjection,
        stats: ProjectedCrossAttentionStats,
    ) -> MaterializedAttentionBatch:
        started = time.perf_counter()
        with projection.context():
            q, k, v = runner.project_to_host(
                query_hidden_host,
                context_hidden_host,
                project_q=projection.project_q,
                project_kv=projection.project_kv,
                stats=stats,
            )
        stats.raw_attention_roundtrip_bytes_avoided += 2 * q.numel() * q.element_size()
        return MaterializedAttentionBatch(q, k, v, time.perf_counter() - started)

    def consume(
        self,
        runner: MaterializedRunner,
        batch: MaterializedAttentionBatch,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        *,
        destination_hidden_host: torch.Tensor,
        residual_hidden_host: torch.Tensor,
        epilogue: AttentionEpilogue,
        consumer_lease: LeaseFactory | None,
        softmax_scale: float | None,
        causal: bool,
        stats: MaterializedStats,
    ) -> None:
        self.consumer.reset(
            destination_hidden_host=destination_hidden_host,
            residual_hidden_host=residual_hidden_host,
            epilogue=epilogue,
        )
        started = time.perf_counter()
        with _lease_context(consumer_lease):
            runner.attention.run_with_device_consumer(
                batch.q,
                batch.k,
                batch.v,
                cu_seqlens_q,
                cu_seqlens_kv,
                output_consumer=self.consumer,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats.attention,
            )
        attention_seconds = time.perf_counter() - started
        stats.attention_output_seconds += attention_seconds
        stats.wall_seconds += batch.projection_wall_seconds + attention_seconds

    def run_self(
        self,
        runner: ProjectedAttentionRunner,
        hidden_host: torch.Tensor,
        cu_seqlens: torch.Tensor,
        projection: SelfProjection,
        *,
        epilogue: AttentionEpilogue,
        consumer_lease: LeaseFactory | None,
        softmax_scale: float | None,
        causal: bool,
        stats: ProjectedAttentionStats,
    ) -> None:
        batch = self.materialize_self(runner, hidden_host, projection, stats)
        self.consume(
            runner,
            batch,
            cu_seqlens,
            cu_seqlens,
            destination_hidden_host=hidden_host,
            residual_hidden_host=hidden_host,
            epilogue=epilogue,
            consumer_lease=consumer_lease,
            softmax_scale=softmax_scale,
            causal=causal,
            stats=stats,
        )

    def run_cross(
        self,
        runner: ProjectedCrossAttentionRunner,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        projection: CrossProjection,
        *,
        epilogue: AttentionEpilogue,
        consumer_lease: LeaseFactory | None,
        softmax_scale: float | None,
        stats: ProjectedCrossAttentionStats,
    ) -> None:
        batch = self.materialize_cross(
            runner,
            query_hidden_host,
            context_hidden_host,
            projection,
            stats,
        )
        self.consume(
            runner,
            batch,
            cu_seqlens_q,
            cu_seqlens_kv,
            destination_hidden_host=query_hidden_host,
            residual_hidden_host=query_hidden_host,
            epilogue=epilogue,
            consumer_lease=consumer_lease,
            softmax_scale=softmax_scale,
            causal=False,
            stats=stats,
        )


class RecomputedAttentionExecutor:
    """Shared direct-write attention mechanics with explicit caller-owned routing."""

    def __init__(self, consumer: AttentionOutputConsumer) -> None:
        self.consumer = consumer

    def run_self(
        self,
        runner: RecomputedAttentionRunner,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        cu_seqlens: torch.Tensor,
        projection: SelfRecomputeProjection,
        *,
        epilogue: AttentionEpilogue,
        consumer_lease: LeaseFactory | None,
        softmax_scale: float | None,
        causal: bool,
        stats: RecomputedAttentionStats,
    ) -> None:
        self.consumer.reset(
            destination_hidden_host=destination_hidden_host,
            residual_hidden_host=source_hidden_host,
            epilogue=epilogue,
        )
        with projection.context(), _lease_context(consumer_lease):
            runner.run_with_device_consumer(
                source_hidden_host,
                cu_seqlens,
                project_q=projection.project_q,
                project_kv=projection.project_kv,
                output_consumer=self.consumer,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats,
            )

    def run_cross(
        self,
        runner: RecomputedCrossAttentionRunner,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        projection: CrossRecomputeProjection,
        *,
        epilogue: AttentionEpilogue,
        consumer_lease: LeaseFactory | None,
        softmax_scale: float | None,
        stats: RecomputedCrossAttentionStats,
    ) -> None:
        self.consumer.reset(
            destination_hidden_host=destination_hidden_host,
            residual_hidden_host=query_hidden_host,
            epilogue=epilogue,
        )
        with projection.context(), _lease_context(consumer_lease):
            runner.run_with_device_consumer(
                query_hidden_host,
                context_hidden_host,
                cu_seqlens_q,
                cu_seqlens_kv,
                project_q=projection.project_q,
                project_kv=projection.project_kv,
                output_consumer=self.consumer,
                softmax_scale=softmax_scale,
                stats=stats,
            )


__all__ = [
    "MaterializedAttentionBatch",
    "MaterializedAttentionExecutor",
    "RecomputedAttentionExecutor",
]
