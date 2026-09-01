from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

ResultT = TypeVar("ResultT")


def init_single_flight(owner: object) -> None:
    owner._single_flight_lock = threading.RLock()  # type: ignore[attr-defined]


def single_flight(method: Callable[..., ResultT]) -> Callable[..., ResultT]:
    """Reject concurrent calls while allowing nested calls on the same thread."""

    @wraps(method)
    def wrapped(self: object, *args: Any, **kwargs: Any) -> ResultT:
        lock = self._single_flight_lock  # type: ignore[attr-defined]
        if not lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} is single-flight")
        try:
            return method(self, *args, **kwargs)
        finally:
            lock.release()

    return cast(Callable[..., ResultT], wrapped)


__all__ = ["init_single_flight", "single_flight"]
