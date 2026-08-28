# seqattn-multigpu

`seqattn-multigpu` is the optional multi-GPU feature package for
`seqattn-core`. It contains static and dynamic query scheduling, multi-device
streaming attention, and the materialized-QKV MiniMax-H3 multi-GPU runner.

The plugin and core must use the exact same version. For GitHub source
installs, install both from the same immutable commit:

```bash
python -m pip install \
  'seqattn-core[dit] @ git+https://github.com/renlililoli/stream-attn.git@COMMIT'
python -m pip install \
  'seqattn-multigpu @ git+https://github.com/renlililoli/stream-attn.git@COMMIT#subdirectory=packages/seqattn-multigpu'
```

The plugin is discoverable through `seqattn_core.features` and exports its
runtime API from `seqattn_multigpu`. Installing `seqattn-core` alone does not
install or import this package.

The validated RTX 5090 scheduling report and its preserved raw records live in
[`docs/rtx5090_dynamic_multigpu_524k_2026-08-26.md`](docs/rtx5090_dynamic_multigpu_524k_2026-08-26.md).
