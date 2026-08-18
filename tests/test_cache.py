import pytest
import torch

from seqattn import HostMemoryPlan, KVLayout
from seqattn.cache import KVPageCache
from seqattn.paging import build_page_descriptors


def test_two_region_cache_has_deterministic_hot_set_and_rolling_eviction():
    layout = KVLayout(80, 1, 16, "fp32", "fp32")
    pages = build_page_descriptors(
        [0, 80],
        bytes_per_token=layout.storage_bytes_per_token,
        page_target_bytes=1024,
        token_alignment=16,
    )
    plan = HostMemoryPlan(
        total_budget_bytes=2 * 2**20,
        pinned_limit_bytes=256 * 2**10,
        bounce_limit_bytes=256 * 2**10,
        metadata_margin_bytes=64 * 2**10,
    )
    probe = KVPageCache(
        pages,
        layout,
        capacity_bytes=3 * 2048,
        hot_fraction=2 / 3,
        memory_plan=plan,
    )
    assert probe.slot_count == 3
    assert probe.hot_page_ids == {0, 1}
    k = torch.empty((16, 1, 16))
    v = torch.empty_like(k)
    trace = [0, 1, 2, 3, 0, 1, 4]
    for page_id in trace:
        page = pages[page_id]
        lookup = probe.get(page, k, v)
        if not lookup.hit:
            k.fill_(page_id)
            v.fill_(-page_id)
            probe.put(page, k, v)
    assert probe.hits == 2
    assert probe.misses == 5
    assert probe.peak_bytes <= probe.registered_bytes
    probe.close()
    assert plan.snapshot().dram_cache_allocated_bytes == 0


def test_host_memory_plan_fails_immediately_on_category_overrun():
    plan = HostMemoryPlan(
        total_budget_bytes=8 * 2**30,
        pinned_limit_bytes=1 * 2**30,
        bounce_limit_bytes=512 * 2**20,
        metadata_margin_bytes=128 * 2**20,
    )
    plan.register("pinned", 1 * 2**30)
    with pytest.raises(MemoryError, match="pinned allocation"):
        plan.register("pinned", 1)
    snapshot = plan.snapshot()
    assert snapshot.operator_host_peak_bytes <= 8 * 2**30
    assert plan.cache_limit_bytes == 8 * 2**30 - 1 * 2**30 - 512 * 2**20 - 128 * 2**20


def test_logical_kv_store_larger_than_8gib_maps_only_budgeted_cache():
    # 3,145,728 tokens at 8x128 BF16 are 6GiB per tensor, 12GiB for K+V.
    total_tokens = 3_145_728
    layout = KVLayout(total_tokens, 8, 128, "bf16", "bf16")
    pages = build_page_descriptors(
        [0, total_tokens],
        bytes_per_token=layout.storage_bytes_per_token,
        page_target_bytes=16 * 2**20,
        token_alignment=64,
    )
    plan = HostMemoryPlan(
        total_budget_bytes=8 * 2**30,
        pinned_limit_bytes=1 * 2**30,
        bounce_limit_bytes=512 * 2**20,
        metadata_margin_bytes=128 * 2**20,
    )
    plan.register("pinned", 1 * 2**30)
    plan.register("bounce", 512 * 2**20)
    cache = KVPageCache(
        pages,
        layout,
        capacity_bytes=plan.cache_limit_bytes,
        hot_fraction=0.8,
        memory_plan=plan,
    )
    snapshot = plan.snapshot()
    assert len(pages) == 384
    assert cache.registered_bytes == plan.cache_limit_bytes
    assert snapshot.operator_host_peak_bytes == 8 * 2**30
    cache.close()
    plan.release("pinned", 1 * 2**30)
    plan.release("bounce", 512 * 2**20)
