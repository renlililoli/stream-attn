"""Ordered CUDA-style stream operations and physical reference lifetimes."""

from __future__ import annotations

from dataclasses import asdict

from ..specs import BufferSpec, ExecutionSpec, OperationSpec


class H3Graph:
    def __init__(self, shape, config, profile, callbacks, weights):
        self.shape, self.config, self.profile = shape, config, profile
        self.callbacks, self.weights = callbacks, weights
        self.operations, self.buffers, self.uses = [], {}, {}
        self.streams = {}
        self.counts, self.transfers, self.resolutions = {}, {}, {}
        self.core_persistent_bytes = 0

    def allocate(
        self, name, size, component, *, owner="callback", persistent=False, host=False, pool=None
    ):
        if not size:
            return None
        if name in self.buffers:
            raise ValueError(f"duplicate physical allocation: {name}")
        self.buffers[name] = {
            "name": name,
            "pool": pool
            or (self.profile.host_pool.name if host else self.profile.device_pool.name),
            "size_bytes": size,
            "component": component,
            "owner": owner,
            "persistent": persistent,
        }
        self.uses[name] = []
        if owner == "operator" and persistent and not host:
            self.core_persistent_bytes += size
        return name

    def use(self, buffers, operation):
        for name in set(buffers):
            if name is not None:
                if name not in self.buffers:
                    raise ValueError(f"unknown allocation {name}")
                self.uses[name].append(operation)

    def retain(self, buffer, through):
        if buffer is not None and not self.buffers[buffer]["persistent"]:
            self.buffers[buffer]["release_after"] = through

    def milestone(self, name, dependencies=(), *, stream=None):
        deps = list(dependencies)
        if stream and self.streams.get(stream):
            deps.append(self.streams[stream])
        self.operations.append(
            OperationSpec(
                name,
                0.0,
                (),
                tuple(dict.fromkeys(d for d in deps if d)),
                kind="control",
                component="synchronization",
            )
        )
        if stream:
            self.streams[stream] = name
        return name

    def op(
        self,
        name,
        key,
        tokens,
        work,
        buffers=(),
        *,
        dependencies=(),
        stream="compute",
        kv_tokens=0,
        component=None,
        details="",
        modeled_workspace_bytes=0,
    ):
        if key not in self.profile.operators:
            raise ValueError(f"missing H3 operator profile: {key}")
        model = self.profile.operators[key]
        seconds, workspace_bytes, provenance, resolution = model.resolve(tokens, work, kv_tokens)
        if resolution != "sample":
            workspace_bytes += modeled_workspace_bytes
        if key == "h2d":
            kind, resources = "io", self.profile.h2d_resources
        elif key == "d2h":
            kind, resources = "io", self.profile.d2h_resources
        else:
            kind = "compute"
            resources = model.rate.resources if model.rate else self.profile.compute_resources
        deps = [*dependencies, self.streams.get(stream)]
        self.operations.append(
            OperationSpec(
                name,
                seconds,
                tuple(resources),
                tuple(dict.fromkeys(d for d in deps if d)),
                kind,
                component or key,
                provenance,
                f"tokens={tokens}; kv_tokens={kv_tokens}; stream={stream}; {details}",
            )
        )
        self.streams[stream] = name
        self.use(buffers, name)
        if workspace_bytes:
            scratch = self.allocate(f"{name}.workspace", workspace_bytes, f"{key} workspace")
            self.use((scratch,), name)
        for index, extra in enumerate(model.additional_workspaces):
            size = extra.fixed_bytes + tokens * extra.bytes_per_token
            if size:
                buffer = self.allocate(
                    f"{name}.extra_workspace{index}", size, extra.component, pool=extra.pool
                )
                self.use((buffer,), name)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.resolutions[resolution] = self.resolutions.get(resolution, 0) + 1
        if key in {"h2d", "d2h", "d2d"}:
            self.transfers[key] = self.transfers.get(key, 0) + work
        return name

    def copy(self, name, key, tokens, size, source, destination, **kwargs):
        return self.op(name, key, tokens, size, (source, destination), **kwargs)

    def finish(self, name):
        return ExecutionSpec(
            name,
            (self.profile.device_pool, self.profile.host_pool, *self.profile.local_pools),
            tuple(self.operations),
            tuple(
                BufferSpec(**data, uses=tuple(dict.fromkeys(self.uses[key])))
                for key, data in self.buffers.items()
            ),
            metadata={
                "model": "H3 full dense block",
                "shape": asdict(self.shape),
                "execution": asdict(self.config),
                "callbacks": asdict(self.callbacks),
                "device_profile": asdict(self.profile),
                "weights": asdict(self.weights),
                "operator_counts": self.counts,
                "transfer_bytes": self.transfers,
                "profile_resolutions": self.resolutions,
                "core_persistent_cuda_bytes": self.core_persistent_bytes,
                "workspace_margin_bytes": self.config.workspace_margin_bytes,
                "core_workspace_budget_bytes": self.core_persistent_bytes
                + self.config.workspace_margin_bytes,
            },
            assumptions=(
                "H3 single-flight dense materialized/recompute runner order with cross-Q FFN carry.",
                "Packed attention tiles never cross sequence segments. Pointwise projection/FFN follow the runner's global ranges.",
                "Physical allocations include callback outputs, declared operator workspaces and declared weights. Aliased views count once.",
                "Rate-based durations and extra workspaces must match the selected backend, dtype and tile. Missing sample shapes never interpolate silently.",
                "Workspace margin is a separate allowance, not an allocated tensor. Context and allocator cached pages are excluded.",
                "GPU work/reference lifetimes are modeled at operation boundaries; CPU launch/allocator overhead requires calibration.",
            ),
        )
