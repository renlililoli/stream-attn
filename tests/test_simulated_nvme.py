from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from seqattn_core import (
    MemoryPageSink,
    MemoryPageSource,
    PagedAttentionConfig,
    PagedAttentionRunner,
    PagedAttentionStats,
    SimulatedNvmeConfig,
    SimulatedNvmeDevice,
    SimulatedPageSink,
    SimulatedPageSource,
    StreamingAttentionConfig,
)
from seqattn_core.reference import streaming_attention_reference
from seqattn_core.paged.simulation import _BandwidthTimeline


def test_aggregate_bandwidth_timeline_is_deterministic():
    timeline = _BandwidthTimeline(
        bandwidth_bytes_per_second=1_000.0,
        latency_seconds=0.01,
        jitter_fraction=0.0,
        random_seed=0,
    )
    first = timeline.reserve(100, arrival_seconds=0.0)
    second = timeline.reserve(100, arrival_seconds=0.0)

    assert first.transfer_start_seconds == pytest.approx(0.01)
    assert first.finish_seconds == pytest.approx(0.11)
    assert first.queue_seconds == pytest.approx(0.0)
    assert second.transfer_start_seconds == pytest.approx(0.11)
    assert second.finish_seconds == pytest.approx(0.21)
    assert second.queue_seconds == pytest.approx(0.10)


def test_concurrent_reads_share_one_bandwidth_limit():
    device = SimulatedNvmeDevice(
        SimulatedNvmeConfig(
            read_bandwidth_bytes_per_second=2 * 2**20,
            write_bandwidth_bytes_per_second=2 * 2**20,
            read_latency_seconds=0.0,
            write_latency_seconds=0.0,
            max_concurrent_reads=2,
            max_concurrent_writes=2,
        )
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(device.throttle_read, 64 * 2**10) for _ in range(2)]
        delays = [future.result() for future in futures]
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.055
    assert sum(delay.physical_bytes for delay in delays) == 128 * 2**10
    assert sum(delay.queue_seconds for delay in delays) >= 0.025


def test_simulated_source_and_sink_preserve_attention_results():
    torch.manual_seed(307)
    q = torch.randn(37, 4, 16)
    k = torch.randn(41, 2, 16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 17, 37], dtype=torch.int32)
    cu_k = torch.tensor([0, 19, 41], dtype=torch.int32)
    memory_source = MemoryPageSource(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        page_target_bytes=1024,
        block_n=16,
        kv_storage_dtype="fp32",
    )
    device = SimulatedNvmeDevice(
        SimulatedNvmeConfig(
            read_bandwidth_bytes_per_second=1e12,
            write_bandwidth_bytes_per_second=1e12,
            read_latency_seconds=0.0,
            write_latency_seconds=0.0,
        )
    )
    source = SimulatedPageSource(memory_source, device=device)
    out = torch.empty_like(q)
    sink = SimulatedPageSink(MemoryPageSink(out), device=device)
    config = PagedAttentionConfig(
        attention=StreamingAttentionConfig(
            backend="reference",
            q_chunk_tokens=7,
            kv_chunk_tokens=16,
            block_m=16,
            block_n=16,
        ),
        host_memory_budget_bytes=7 * 2**20,
        pinned_staging_budget_bytes=1 * 2**20,
        direct_io_bounce_budget_bytes=1 * 2**20,
        metadata_margin_bytes=1 * 2**20,
        page_target_bytes=1024,
        io_workers=4,
        io_queue_depth=2,
        num_output_buffers=2,
        direct_io=False,
        kv_storage_dtype="fp32",
    )
    stats = PagedAttentionStats()
    actual = PagedAttentionRunner(config, device="cpu").run(
        source,
        source,
        cu_q,
        cu_k,
        sink,
        stats=stats,
    )
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=7,
        kv_chunk_tokens=16,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert stats.nvme_logical_read_bytes == 0
    assert stats.nvme_logical_write_bytes == 0
    assert stats.simulated_logical_read_bytes > 0
    assert stats.simulated_physical_read_bytes == stats.simulated_logical_read_bytes
    assert stats.simulated_logical_write_bytes == q.numel() * q.element_size()
    assert stats.simulated_read_service_seconds > 0
    assert stats.simulated_write_service_seconds > 0


@pytest.mark.parametrize(
    "changes",
    [
        {"read_bandwidth_bytes_per_second": 0.0},
        {"write_latency_seconds": -1.0},
        {"max_concurrent_reads": 0},
        {"jitter_fraction": 1.0},
    ],
)
def test_simulated_nvme_config_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        SimulatedNvmeConfig(**changes)
