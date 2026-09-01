import pytest

from seqattn_multigpu import (
    MultiGpuH3MaterializedRunner,
    MultiGpuQKVProjectionRunner,
    MultiGpuStreamingAttentionRunner,
)


class _Executor:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self, *, wait):
        assert wait is True
        self.shutdown_calls += 1


class _Closer:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _streaming_runner():
    runner = MultiGpuStreamingAttentionRunner.__new__(MultiGpuStreamingAttentionRunner)
    runner._executor = _Executor()
    runner._closed = False
    return runner, (runner._executor,)


def _projection_runner():
    runner = MultiGpuQKVProjectionRunner.__new__(MultiGpuQKVProjectionRunner)
    runner._executor = _Executor()
    runner._closed = False
    return runner, (runner._executor,)


def _h3_runner():
    runner = MultiGpuH3MaterializedRunner.__new__(MultiGpuH3MaterializedRunner)
    runner.projection = _Closer()
    runner.attention = _Closer()
    runner._closed = False
    return runner, (runner.projection, runner.attention)


@pytest.mark.parametrize(
    "runner_factory",
    [_streaming_runner, _projection_runner, _h3_runner],
)
def test_multigpu_runners_are_idempotent_context_managers(runner_factory):
    runner, resources = runner_factory()

    with runner as entered:
        assert entered is runner

    runner.close()
    for resource in resources:
        calls = getattr(resource, "shutdown_calls", None)
        if calls is None:
            calls = resource.close_calls
        assert calls == 1
