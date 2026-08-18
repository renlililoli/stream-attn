"""Compatibility facade for NVMe-backed page stores."""

from seqattn_core.storage import (
    DIRECT_IO_ALIGNMENT,
    NvmeOutputSink,
    NvmeQKVStore,
    NvmeQKVWriter,
    ephemeral_nvme_directory,
    load_nvme_output,
)
from seqattn_core.storage.direct_io import _open_file as _direct_open_file


def _open_file(path, flags, mode, direct_io):
    return _direct_open_file(path, flags, mode, direct_io)


__all__ = [
    "DIRECT_IO_ALIGNMENT",
    "NvmeOutputSink",
    "NvmeQKVStore",
    "NvmeQKVWriter",
    "ephemeral_nvme_directory",
    "load_nvme_output",
]
