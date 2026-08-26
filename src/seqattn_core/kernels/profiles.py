from __future__ import annotations

import re
from typing import NamedTuple

import torch


class KernelLaunchProfile(NamedTuple):
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


PORTABLE_KERNEL = KernelLaunchProfile(64, 64, 4, 2)
A30_TRITON37_D128_KERNEL = KernelLaunchProfile(128, 64, 8, 4)
BLACKWELL_D128_KERNEL = KernelLaunchProfile(128, 64, 8, 3)


def triton_major_minor() -> tuple[int, int] | None:
    try:
        import triton
    except ImportError:
        return None
    match = re.match(r"^(\d+)\.(\d+)", triton.__version__)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_builtin_kernel_profile(
    *,
    device: torch.device,
    head_dim: int,
    dtype: torch.dtype,
) -> KernelLaunchProfile:
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or dtype not in {torch.float16, torch.bfloat16}
        or head_dim != 128
    ):
        return PORTABLE_KERNEL

    major, _ = torch.cuda.get_device_capability(device)
    if major >= 12:
        return BLACKWELL_D128_KERNEL
    if (
        major == 8
        and "A30" in torch.cuda.get_device_name(device).upper()
        and triton_major_minor() == (3, 7)
    ):
        return A30_TRITON37_D128_KERNEL
    return PORTABLE_KERNEL


__all__ = [
    "A30_TRITON37_D128_KERNEL",
    "BLACKWELL_D128_KERNEL",
    "PORTABLE_KERNEL",
    "KernelLaunchProfile",
    "resolve_builtin_kernel_profile",
    "triton_major_minor",
]
