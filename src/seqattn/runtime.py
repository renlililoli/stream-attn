"""Compatibility facade for the streaming attention runtime."""

from seqattn_core.streaming import StreamingAttentionRunner, resolve_backend

__all__ = ["StreamingAttentionRunner", "resolve_backend"]
