import pytest
import torch

from seqattn_core.dit.common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    TiledHostStageRunner,
    TiledStageStats,
    cu_seqlens_from_padding_mask,
    reject_additive_attention_mask,
)
from seqattn_core.streaming.tasks import QueryTask


def test_padding_mask_converts_prefix_lengths_to_packed_boundaries():
    mask = torch.tensor(
        [
            [True, True, True, False],
            [False, False, False, False],
            [True, True, False, False],
        ]
    )

    actual = cu_seqlens_from_padding_mask(mask)

    torch.testing.assert_close(actual, torch.tensor([0, 3, 3, 5], dtype=torch.int32))


def test_padding_mask_rejects_non_prefix_rows_and_additive_qk_masks():
    with pytest.raises(ValueError, match="valid prefix"):
        cu_seqlens_from_padding_mask(torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="CPU boolean"):
        cu_seqlens_from_padding_mask(torch.ones((2, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="additive QxK masks"):
        reject_additive_attention_mask(torch.zeros((2, 3, 3)), name="attention_mask")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_output_consumer_applies_epilogue_and_writes_host_tiles():
    torch.manual_seed(311)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    tokens = 9
    hidden_features = 8
    residual = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    destination = torch.empty_like(residual, pin_memory=True)
    attention = torch.randn(tokens, 2, 4, dtype=dtype, device=device)
    workspace = AttentionOutputWorkspace(
        hidden_features=hidden_features,
        output_chunk_tokens=5,
        dtype=dtype,
        device=device,
    )
    consumer = AttentionOutputConsumer(workspace)

    def epilogue(tile, residual_host, start, stop):
        return tile.reshape(stop - start, hidden_features) + residual_host[start:stop].to(device)

    consumer.reset(
        destination_hidden_host=destination,
        residual_hidden_host=residual,
        epilogue=epilogue,
    )
    consumer(attention[:5], 0, 5)
    consumer(attention[5:], 5, tokens)
    consumer.finish()
    consumer.synchronize()

    expected = attention.reshape(tokens, hidden_features).cpu() + residual
    torch.testing.assert_close(destination, expected, atol=0, rtol=0)
    assert consumer.d2h_bytes == destination.numel() * destination.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_output_consumer_supports_dynamic_query_tasks():
    device = torch.device("cuda:0")
    dtype = torch.float16
    hidden = torch.zeros((8, 4), dtype=dtype, pin_memory=True)
    workspace = AttentionOutputWorkspace(
        hidden_features=4,
        output_chunk_tokens=4,
        dtype=dtype,
        device=device,
        num_output_buffers=1,
    )
    consumer = AttentionOutputConsumer(workspace)
    consumer.reset(
        destination_hidden_host=hidden,
        residual_hidden_host=hidden,
        epilogue=lambda tile, residual, start, stop: tile.reshape(stop - start, 4),
        range_start=2,
        range_stop=6,
    )
    task = QueryTask(2, 6, 0, 8, 2, 0)
    consumer.begin_task(task)
    consumer(torch.ones((4, 2, 2), dtype=dtype, device=device), 2, 6)
    done = consumer.finish_task()
    done.synchronize()

    torch.testing.assert_close(hidden[2:6], torch.ones((4, 4), dtype=dtype))
    assert consumer.task_d2h_bytes() == 4 * 4 * hidden.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tiled_host_stage_supports_in_place_host_updates_and_stats():
    torch.manual_seed(313)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    hidden = torch.randn(11, 8, dtype=dtype, pin_memory=True)
    original = hidden.clone()
    stats = TiledStageStats()
    runner = TiledHostStageRunner(
        hidden_features=8,
        chunk_tokens=4,
        dtype=dtype,
        device=device,
    )

    runner.run(hidden, hidden, lambda tile, start, stop: tile + start + 1, stats=stats)

    expected = original.clone()
    for start in range(0, hidden.shape[0], 4):
        stop = min(start + 4, hidden.shape[0])
        expected[start:stop].copy_((original[start:stop].to(device) + start + 1).cpu())
    torch.testing.assert_close(hidden, expected, atol=0, rtol=0)
    assert stats.chunks == 3
    assert stats.tokens == hidden.shape[0]
    assert stats.h2d_bytes == hidden.numel() * hidden.element_size()
    assert stats.d2h_bytes == hidden.numel() * hidden.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tiled_host_stage_recovers_after_operation_failure():
    device = torch.device("cuda:0")
    dtype = torch.float16
    source = torch.randn(17, 8, dtype=dtype, pin_memory=True)
    destination = torch.empty_like(source, pin_memory=True)
    runner = TiledHostStageRunner(
        hidden_features=8,
        chunk_tokens=4,
        dtype=dtype,
        device=device,
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        runner.run(
            source,
            destination,
            lambda tile, start, stop: (_ for _ in ()).throw(RuntimeError("stage failed")),
        )

    runner.run(source, destination, lambda tile, start, stop: tile + 1)
    torch.testing.assert_close(destination, source + 1, atol=0, rtol=0)
