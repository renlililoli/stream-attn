import pytest
import torch

from seqattn_core.dit.minimax_h3 import (
    H3Config,
    H3DenoisingStep,
    H3DiTStats,
    H3SequenceMeta,
)
from seqattn_core.dit.minimax_h3.config import use_sol_streaming


def _meta(prefixes=(64,)):
    return H3SequenceMeta(
        torch.tensor([0, 256], dtype=torch.int32),
        exact_prefix_tokens=prefixes,
    )


def test_h3_sol_policy_uses_explicit_step_and_layer_indices():
    config = H3Config(attention_mode="sol_streaming")
    assert not use_sol_streaming(
        config,
        sequence_meta=_meta(),
        denoising_step=H3DenoisingStep(3, 20),
        block_index=2,
    )
    assert not use_sol_streaming(
        config,
        sequence_meta=_meta(),
        denoising_step=H3DenoisingStep(4, 20),
        block_index=1,
    )
    assert use_sol_streaming(
        config,
        sequence_meta=_meta(),
        denoising_step=H3DenoisingStep(4, 20),
        block_index=2,
    )


def test_h3_sol_policy_never_silently_infers_missing_metadata():
    config = H3Config(attention_mode="sol_streaming")
    with pytest.raises(ValueError, match="denoising step"):
        use_sol_streaming(
            config,
            sequence_meta=_meta(),
            denoising_step=None,
            block_index=2,
        )
    with pytest.raises(ValueError, match="block_index"):
        use_sol_streaming(
            config,
            sequence_meta=_meta(),
            denoising_step=H3DenoisingStep(4, 20),
            block_index=None,
        )
    with pytest.raises(ValueError, match="exact_prefix_tokens"):
        use_sol_streaming(
            config,
            sequence_meta=H3SequenceMeta(torch.tensor([0, 256], dtype=torch.int32)),
            denoising_step=H3DenoisingStep(4, 20),
            block_index=2,
        )


def test_h3_dense_policy_does_not_require_sparse_metadata():
    assert not use_sol_streaming(
        H3Config(),
        sequence_meta=H3SequenceMeta(torch.tensor([0, 3], dtype=torch.int32)),
        denoising_step=None,
        block_index=None,
    )


def test_h3_sequence_meta_validates_per_segment_prefixes():
    H3SequenceMeta(
        torch.tensor([0, 64, 64, 129], dtype=torch.int32),
        exact_prefix_tokens=(64, 0, 1),
    ).validate(129)
    with pytest.raises(ValueError, match="one value"):
        H3SequenceMeta(
            torch.tensor([0, 64, 129], dtype=torch.int32),
            exact_prefix_tokens=(64,),
        ).validate(129)
    with pytest.raises(ValueError, match="exceeds"):
        H3SequenceMeta(
            torch.tensor([0, 64], dtype=torch.int32),
            exact_prefix_tokens=(65,),
        ).validate(64)


def test_h3_sparse_config_round_trips_toml(tmp_path):
    from seqattn_core.dit.minimax_h3 import load_h3_config

    path = tmp_path / "seqattn.toml"
    path.write_text(
        "[minimax_h3]\n"
        "execution_mode = 'recompute'\n"
        "attention_mode = 'sol_streaming'\n"
        "projection_tile_tokens = 2048\n"
        "ffn_tile_tokens = 1024\n"
        "sol_tau = 1.25\n"
        "sol_first_dense_step_fraction = 0.1\n"
        "sol_first_dense_layers = 3\n",
        encoding="utf-8",
    )
    assert load_h3_config(path) == H3Config(
        execution_mode="recompute",
        attention_mode="sol_streaming",
        projection_tile_tokens=2048,
        ffn_tile_tokens=1024,
        sol_tau=1.25,
        sol_first_dense_step_fraction=0.1,
        sol_first_dense_layers=3,
    )


def _build_sparse_h3_case(*, execution_mode, first_dense_step_fraction):
    from seqattn_core import StreamingAttentionConfig, build_attention_plan
    from seqattn_core.dit.minimax_h3 import H3Config, build_h3_runner

    device = torch.device("cuda")
    tokens = 193
    hidden_features = 32
    heads = 1
    head_dim = 128
    hidden = torch.randn(tokens, hidden_features, dtype=torch.bfloat16, pin_memory=True)
    qkv = torch.nn.Linear(hidden_features, 3 * heads * head_dim, bias=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    output = torch.nn.Linear(heads * head_dim, hidden_features, bias=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=128,
            kv_chunk_tokens=128,
            output_mode="device_consumer",
        ),
    )
    config = H3Config(
        execution_mode=execution_mode,
        attention_mode="sol_streaming",
        projection_tile_tokens=tokens,
        ffn_tile_tokens=64,
        sol_tau=1000.0,
        sol_first_dense_step_fraction=first_dense_step_fraction,
        sol_first_dense_layers=0,
    )
    return (
        hidden,
        qkv,
        output,
        build_h3_runner(
            plan,
            hidden_features=hidden_features,
            config=config,
        ),
    )


def _project_reference(hidden, qkv, *, chunk_tokens):
    heads = 1
    head_dim = 128
    q_parts = []
    k_parts = []
    v_parts = []
    for start in range(0, hidden.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, hidden.shape[0])
        projected = qkv(hidden[start:stop].to(qkv.weight.device)).view(
            -1,
            3,
            heads,
            head_dim,
        )
        q_parts.append(projected[:, 0].cpu())
        k_parts.append(projected[:, 1].cpu())
        v_parts.append(projected[:, 2].cpu())
    return tuple(torch.cat(parts) for parts in (q_parts, k_parts, v_parts))


def _dense_reference(q, k, v):
    return (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale=128**-0.5,
        )
        .squeeze(0)
        .transpose(0, 1)
    )


def _apply_attention_epilogue(attention, residual, output, *, q_chunk_tokens=128):
    result = torch.empty_like(residual)
    device = output.weight.device
    for start in range(0, residual.shape[0], q_chunk_tokens):
        stop = min(start + q_chunk_tokens, residual.shape[0])
        tile = output(attention[start:stop].reshape(stop - start, 128).to(device))
        tile.add_(residual[start:stop].to(device))
        result[start:stop].copy_(tile.cpu())
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_h3_materialized_switches_dense_and_sol_without_summary_fallback():
    from seqattn_core.dit.minimax_h3 import (
        H3BlockOps,
        H3DenoisingStep,
        H3DiTStats,
        H3MaterializedProjection,
        H3SequenceMeta,
    )
    from seqattn_core.sparse import sol_streaming_reference

    torch.manual_seed(1811)
    hidden, qkv, output, runner = _build_sparse_h3_case(
        execution_mode="materialized",
        first_dense_step_fraction=0.5,
    )
    original = hidden.clone()
    cu = torch.tensor([0, hidden.shape[0]], dtype=torch.int32)
    meta = H3SequenceMeta(cu, exact_prefix_tokens=(0,))
    projection_calls = 0

    def project_qkv(tile, start, stop):
        nonlocal projection_calls
        del start, stop
        projection_calls += 1
        projected = qkv(tile).view(-1, 3, 1, 128)
        return tuple(projected[:, index].contiguous() for index in range(3))

    def attention_epilogue(attention, residual_host, start, stop):
        return output(attention).add_(residual_host[start:stop].to(output.weight.device))

    ops = H3BlockOps(attention_epilogue, lambda tile, start, stop: tile)
    projection = H3MaterializedProjection(project_qkv)

    dense_stats = H3DiTStats()
    dense_hidden = original.clone().pin_memory()
    runner.run_block_(
        dense_hidden,
        meta,
        projection,
        ops,
        block_index=0,
        denoising_step=H3DenoisingStep(0, 2),
        stats=dense_stats,
    )
    dense_q, dense_k, dense_v = _project_reference(original, qkv, chunk_tokens=193)
    dense_expected = _apply_attention_epilogue(
        _dense_reference(dense_q, dense_k, dense_v),
        original,
        output,
    )
    torch.testing.assert_close(dense_hidden, dense_expected, atol=8e-2, rtol=2e-2)
    assert dense_stats.dense_attention_blocks == 1
    assert dense_stats.sol_streaming_blocks == 0
    assert dense_stats.sol_attention.summary_kv_tokens == 0

    sparse_stats = H3DiTStats()
    sparse_hidden = original.clone().pin_memory()
    runner.run_block_(
        sparse_hidden,
        meta,
        projection,
        ops,
        block_index=0,
        denoising_step=H3DenoisingStep(1, 2),
        stats=sparse_stats,
    )
    sparse_q, sparse_k, sparse_v = _project_reference(original, qkv, chunk_tokens=193)
    sparse_attention = sol_streaming_reference(
        sparse_q,
        sparse_k,
        sparse_v,
        cu,
        exact_prefix_tokens=(0,),
        tau=1000.0,
    )
    sparse_expected = _apply_attention_epilogue(
        sparse_attention,
        original,
        output,
    )
    torch.testing.assert_close(sparse_hidden, sparse_expected, atol=8e-2, rtol=2e-2)
    assert sparse_stats.dense_attention_blocks == 0
    assert sparse_stats.sol_streaming_blocks == 1
    assert sparse_stats.sol_attention.summary_kv_tokens == hidden.shape[0]
    assert sparse_stats.sol_attention.approximate_route_blocks > 0

    calls_before_failure = projection_calls
    with pytest.raises(ValueError, match="does not support causal"):
        runner.run_block_(
            original.clone().pin_memory(),
            meta,
            projection,
            ops,
            block_index=0,
            denoising_step=H3DenoisingStep(1, 2),
            causal=True,
        )
    assert projection_calls == calls_before_failure


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_h3_recompute_sol_uses_one_extra_kv_summary_pass():
    from seqattn_core.dit.minimax_h3 import (
        H3BlockOps,
        H3DenoisingStep,
        H3DiTStats,
        H3RecomputeProjection,
        H3SequenceMeta,
    )
    from seqattn_core.sparse import sol_streaming_reference

    torch.manual_seed(1817)
    source, qkv, output, runner = _build_sparse_h3_case(
        execution_mode="recompute",
        first_dense_step_fraction=0.0,
    )
    original = source.clone()
    destination = torch.empty_like(source, pin_memory=True)
    cu = torch.tensor([0, source.shape[0]], dtype=torch.int32)
    meta = H3SequenceMeta(cu, exact_prefix_tokens=(0,))
    q_ranges = []
    kv_ranges = []

    def project_q(tile, destination_q, start, stop):
        q_ranges.append((start, stop))
        projected = qkv(tile)[:, :128].view(-1, 1, 128)
        destination_q.copy_(projected)

    def project_kv(tile, destination_k, destination_v, start, stop):
        kv_ranges.append((start, stop))
        projected = qkv(tile)
        destination_k.copy_(projected[:, 128:256].view(-1, 1, 128))
        destination_v.copy_(projected[:, 256:].view(-1, 1, 128))

    def attention_epilogue(attention, residual_host, start, stop):
        return output(attention).add_(residual_host[start:stop].to(output.weight.device))

    stats = H3DiTStats()
    runner.run_block(
        source,
        destination,
        meta,
        H3RecomputeProjection(project_q, project_kv),
        H3BlockOps(attention_epilogue, lambda tile, start, stop: tile),
        block_index=0,
        denoising_step=H3DenoisingStep(0, 1),
        stats=stats,
    )

    q, k, v = _project_reference(original, qkv, chunk_tokens=128)
    attention = sol_streaming_reference(
        q,
        k,
        v,
        cu,
        exact_prefix_tokens=(0,),
        tau=1000.0,
    )
    expected = _apply_attention_epilogue(attention, original, output)
    torch.testing.assert_close(destination, expected, atol=8e-2, rtol=2e-2)
    assert stats.sol_streaming_blocks == 1
    assert stats.sol_attention.summary_kv_tokens == source.shape[0]
    assert stats.sol_attention.summary_kv_tiles == 2
    assert stats.recompute.q_projection_chunks == 2
    assert stats.recompute.kv_projection_chunks == 6
    assert q_ranges == [(0, 128), (128, 193)]
    assert kv_ranges == [(0, 128), (128, 193)] * 3


def test_h3_stats_preserves_sol_effective_density():
    stats = H3DiTStats()
    stats.sol_attention.exact_route_blocks = 3
    stats.sol_attention.approximate_route_blocks = 1
    assert stats.as_dict()["sol_attention"]["effective_density"] == 0.75
