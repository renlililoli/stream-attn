import errno
import json
import os

import pytest
import torch

from seqattn_core import (
    HostMemoryPlan,
    MemoryPageSink,
    NvmeOutputSink,
    NvmeQKVStore,
    NvmeQKVWriter,
    PagedAttentionConfig,
    PagedAttentionRunner,
    StreamingAttentionConfig,
    load_nvme_output,
)
from seqattn_core.reference import streaming_attention_reference
from seqattn_core.storage import DIRECT_IO_ALIGNMENT
from seqattn_core.storage.direct_io import _open_file


def make_data(dtype=torch.float32):
    torch.manual_seed(307)
    q = torch.randn(37, 4, 16, dtype=dtype)
    k = torch.randn(43, 2, 16, dtype=dtype)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 13, 13, 37], dtype=torch.int32)
    cu_k = torch.tensor([0, 17, 17, 43], dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def config(storage="fp32"):
    return PagedAttentionConfig(
        attention=StreamingAttentionConfig(
            backend="reference",
            q_chunk_tokens=7,
            kv_chunk_tokens=16,
            block_m=16,
            block_n=16,
        ),
        host_memory_budget_bytes=16 * 2**20,
        pinned_staging_budget_bytes=2 * 2**20,
        direct_io_bounce_budget_bytes=4 * 2**20,
        metadata_margin_bytes=1 * 2**20,
        page_target_bytes=2048,
        io_workers=2,
        io_queue_depth=2,
        direct_io=False,
        kv_storage_dtype=storage,
    )


def test_buffered_nvme_source_and_output_match_reference(tmp_path):
    q, k, v, cu_q, cu_k = make_data()
    store = NvmeQKVWriter.from_tensors(
        tmp_path / "qkv",
        q,
        k,
        v,
        cu_q,
        cu_k,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="fp32",
        direct_io=False,
    )
    stats = None
    output_path = PagedAttentionRunner(config(), device="cpu").run(
        store,
        store,
        cu_q,
        cu_k,
        NvmeOutputSink(tmp_path / "output", direct_io=False),
        causal=True,
        stats=stats,
    )
    actual = load_nvme_output(output_path)
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=7,
        kv_chunk_tokens=16,
        causal=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    manifest = json.loads((tmp_path / "qkv" / "manifest.json").read_text())
    for page in [*manifest["q_pages"], *manifest["kv_pages"]]:
        assert page["file_offset"] % DIRECT_IO_ALIGNMENT == 0
        assert page["storage_bytes"] % DIRECT_IO_ALIGNMENT == 0
        assert page["payload_bytes"] + page["padding_bytes"] == page["storage_bytes"]


def test_nvme_int8_matches_memory_quantized_path(tmp_path):
    q, k, v, cu_q, cu_k = make_data()
    store = NvmeQKVWriter.from_tensors(
        tmp_path / "qkv-int8",
        q,
        k,
        v,
        cu_q,
        cu_k,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="int8",
        direct_io=False,
    )
    out = torch.empty_like(q)
    stats = None
    PagedAttentionRunner(config("int8"), device="cpu").run(
        store, store, cu_q, cu_k, MemoryPageSink(out), stats=stats
    )
    assert torch.isfinite(out).all()
    assert store.quantization_seconds > 0


def test_manifest_corruption_is_rejected(tmp_path):
    q, k, v, cu_q, cu_k = make_data()
    NvmeQKVWriter.from_tensors(
        tmp_path / "bad",
        q,
        k,
        v,
        cu_q,
        cu_k,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="fp32",
        direct_io=False,
    )
    manifest_path = tmp_path / "bad" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["kv_pages"][0]["token_start"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="crosses or skips"):
        NvmeQKVStore(tmp_path / "bad", direct_io=False)


def test_short_read_is_reported(tmp_path):
    q, k, v, cu_q, cu_k = make_data()
    store = NvmeQKVWriter.from_tensors(
        tmp_path / "short",
        q,
        k,
        v,
        cu_q,
        cu_k,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="fp32",
        direct_io=False,
    )
    plan = HostMemoryPlan(
        total_budget_bytes=8 * 2**20,
        pinned_limit_bytes=1 * 2**20,
        bounce_limit_bytes=2 * 2**20,
        metadata_margin_bytes=1 * 2**20,
    )
    os.truncate(store.kv_path, store.kv_path.stat().st_size - DIRECT_IO_ALIGNMENT)
    reader = store.open_reader(plan, 1)
    page = store.kv_pages[-1]
    k_out = torch.empty((page.padded_tokens, store.kv_layout.heads, store.kv_layout.head_dim))
    v_out = torch.empty_like(k_out)
    try:
        with pytest.raises(OSError, match="short read"):
            reader.read_kv(page, k_out, v_out)
    finally:
        reader.close()


def test_direct_io_unsupported_never_falls_back(monkeypatch, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"x" * DIRECT_IO_ALIGNMENT)
    real_open = os.open

    def reject_direct(target, flags, mode=0o777):
        if flags & getattr(os, "O_DIRECT", 0):
            raise OSError(errno.EINVAL, "not supported")
        return real_open(target, flags, mode)

    monkeypatch.setattr(os, "open", reject_direct)
    with pytest.raises(RuntimeError, match="does not support required O_DIRECT"):
        _open_file(path, os.O_RDONLY, 0, True)


def test_direct_io_round_trip_when_supported(tmp_path):
    q, k, v, cu_q, cu_k = make_data(torch.bfloat16)
    try:
        store = NvmeQKVWriter.from_tensors(
            tmp_path / "direct",
            q,
            k,
            v,
            cu_q,
            cu_k,
            page_target_bytes=4096,
            block_n=16,
            kv_storage_dtype="bf16",
            direct_io=True,
        )
    except RuntimeError as error:
        pytest.skip(f"test filesystem has no O_DIRECT support: {error}")
    plan = HostMemoryPlan(
        total_budget_bytes=8 * 2**20,
        pinned_limit_bytes=1 * 2**20,
        bounce_limit_bytes=2 * 2**20,
        metadata_margin_bytes=1 * 2**20,
    )
    reader = store.open_reader(plan, 1)
    page = store.q_pages[0]
    out = torch.empty(
        (page.padded_tokens, store.q_layout.heads, store.q_layout.head_dim),
        dtype=q.dtype,
    )
    try:
        reader.read_q(page, out)
    finally:
        reader.close()
    torch.testing.assert_close(out[: page.valid_tokens], q[: page.valid_tokens])


def test_disk_full_error_cleans_unpublished_store(monkeypatch, tmp_path):
    q, k, v, cu_q, cu_k = make_data()
    real_pwritev = os.pwritev
    calls = 0

    def fail_after_first(fd, buffers, offset):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError(errno.ENOSPC, "no space left")
        return real_pwritev(fd, buffers, offset)

    monkeypatch.setattr(os, "pwritev", fail_after_first)
    with pytest.raises(OSError, match="no space left"):
        NvmeQKVWriter.from_tensors(
            tmp_path / "full",
            q,
            k,
            v,
            cu_q,
            cu_k,
            page_target_bytes=2048,
            block_n=16,
            kv_storage_dtype="fp32",
            direct_io=False,
        )
    assert not (tmp_path / "full" / "manifest.json").exists()
    assert not (tmp_path / "full" / "q.bin").exists()
    assert not (tmp_path / "full" / "kv.bin").exists()


def test_interrupted_writer_removes_unpublished_files(tmp_path):
    q, k, _v, cu_q, cu_k = make_data()
    from seqattn_core import KVLayout, TensorLayout

    writer = NvmeQKVWriter(
        tmp_path / "interrupted",
        q_layout=TensorLayout(q.shape[0], q.shape[1], q.shape[2], "fp32"),
        kv_layout=KVLayout(k.shape[0], k.shape[1], k.shape[2], "fp32", "fp32"),
        cu_seqlens_q=cu_q.tolist(),
        cu_seqlens_k=cu_k.tolist(),
        page_target_bytes=2048,
        block_n=16,
        direct_io=False,
    )

    def broken_q_pages():
        first = writer.q_pages[0]
        yield q[first.token_start : first.token_stop]
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        writer.write_pages(broken_q_pages(), iter(()))
    assert not (tmp_path / "interrupted" / "manifest.json").exists()
    assert not (tmp_path / "interrupted" / "q.bin").exists()
    assert not (tmp_path / "interrupted" / "kv.bin").exists()
    assert not list((tmp_path / "interrupted").glob("*.tmp"))
