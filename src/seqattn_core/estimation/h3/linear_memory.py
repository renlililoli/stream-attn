"""Live tensor accounting for eager INT8 ConvRot + in-place row quantization.

The output and caller input are accounted separately. These equations mirror
rotation, BF16/FP16 row quantization, INT32 matmul, chunked scaling and cat.
Different/fused implementations must supply their own operator workspace sample.
"""


def eager_int8_workspace(
    tokens,
    inputs,
    outputs,
    element_bytes,
    *,
    convrot_group=256,
    per_channel_scale=False,
    scaling_chunk_bytes=256 * 2**20,
):
    if inputs % convrot_group:
        raise ValueError("ConvRot group must divide the linear input width")
    hadamard = convrot_group**2 * element_bytes
    rotated = tokens * inputs * element_bytes
    quantized = tokens * inputs
    row_scales = tokens * 4
    output = tokens * outputs * element_bytes
    # CUDA eager pads M to 32 rows and K to 8. H3 dimensions already satisfy
    # the N alignment; model tail-row padding explicitly rather than truncating it.
    padded_m = max(32, (tokens + 31) // 32 * 32)
    padded_k = (inputs + 7) // 8 * 8
    padded_input = padded_m * padded_k if (padded_m, padded_k) != (tokens, inputs) else 0
    accumulator = padded_m * outputs * 4
    base = hadamard + rotated + quantized + row_scales
    # abs().amax() releases the full abs tensor before division. Division stays
    # in input dtype; round_ and clamp_ are in-place, followed by conversion to I8.
    quantization = hadamard + 2 * rotated + quantized + tokens * (4 + 5 * element_bytes + 1)
    row_padding = (padded_m - tokens) * inputs
    matmul = base + padded_input + row_padding + accumulator + inputs * outputs
    peak = max(quantization, matmul)
    chunk_rows = max(1, min(tokens, scaling_chunk_bytes // (outputs * 4)))
    kept_parts = 0
    previous_chunk = previous_scales = 0
    last_chunk = last_scales = 0
    for start in range(0, tokens, chunk_rows):
        rows = min(chunk_rows, tokens - start)
        chunk = rows * outputs * 4
        scales = rows * (outputs if per_channel_scale else 1) * 4
        part = rows * outputs * element_bytes
        # Python evaluates RHS before rebinding each loop local. Previous chunk
        # and scale tensors overlap their replacements; BF16 parts stay in a list.
        peak = max(
            peak,
            base + accumulator + kept_parts + previous_chunk + chunk + previous_scales,
            base + accumulator + kept_parts + chunk + previous_scales + scales,
            base + accumulator + kept_parts + chunk + scales + chunk + part,
        )
        kept_parts += part
        previous_chunk, previous_scales = chunk, scales
        last_chunk, last_scales = chunk, scales
    # Only the final (possibly tiny) chunk/scales remain alive at concatenation.
    # The old INT32 result is not released until the cat RHS has produced output.
    peak = max(peak, base + accumulator + kept_parts + last_chunk + last_scales + output)
    return max(0, peak - output)
