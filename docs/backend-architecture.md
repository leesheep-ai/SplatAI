# SplatAI backend architecture

## Goal

SplatAI separates the desktop orchestration layer from research-heavy Gaussian Splatting runtimes. The desktop layer owns input preparation, project state, progress reporting, artifact validation, and publishing. Each training implementation is treated as a process-level backend.

## Why this boundary exists

Brush is attractive as a compatibility backend because it has broad graphics-platform support. gsplat, 3DGUT, FastGS, 3DGRUT/NHT, and Fast-PGSR depend on rapidly changing Python/CUDA stacks. Linking those stacks directly into the desktop process would make packaging, upgrades, crash isolation, and multi-backend support unnecessarily fragile.

The stable boundary is therefore:

```text
Desktop/Tauri
    |
    | normalized TrainingJob
    v
TrainingBackend adapter
    |
    | process invocation
    v
Pinned external runtime
    |
    | output directory
    v
Artifact discovery + validation
    |
    v
Atomic publisher
```

## Phase 1 backends

### Brush

The Brush adapter preserves the current OOOSplat-style CLI contract:

- `--total-steps`
- `--max-resolution`
- `--export-every`
- `--export-path`
- `--export-name final.ply.tmp`

This is the compatibility/fallback path.

### gsplat

The initial gsplat adapter intentionally uses the upstream `examples/simple_trainer.py` interface rather than vendoring a fork of its training loop.

Defaults:

- strategy: MCMC
- viewer: disabled
- training video generation: disabled
- final-step PLY export: enabled
- checkpoints preserved when emitted by upstream trainer

Full 3DGUT mode is requested using both `with_ut` and `with_eval3d`. Upstream gsplat explicitly notes that this combination should use `MCMCStrategy`, so SplatAI's 3DGUT path never maps to `DefaultStrategy`.

## Resolution handling

Brush exposes a maximum-resolution CLI flag. gsplat's COLMAP dataset loader instead primarily exposes dataset downsampling. SplatAI must not silently convert one semantic into the other.

Resolution normalization will move into the common dataset-preparation layer. Backends then receive a dataset already prepared at the intended training resolution. This keeps product quality presets backend-independent.

## Device selection

A future runtime health probe will report a structured capability record including:

- Python runtime availability;
- torch version;
- CUDA runtime version;
- visible CUDA devices;
- GPU compute capability;
- VRAM;
- gsplat import/version;
- 3DGUT support;
- backend-specific optional dependencies.

The UI should choose a recommended product mode from these capabilities but allow an expert override.

## Product-mode mapping

| Product mode | Phase 1 mapping | Planned mapping |
| --- | --- | --- |
| Compatible | Brush | Brush |
| Quality | gsplat MCMC | gsplat MCMC + tuned presets |
| Ultimate | gsplat MCMC + optional 3DGUT | 3DGUT / 3DGRUT + NHT |
| Fast | unavailable | FastGS |
| Geometry | unavailable | Fast-PGSR |
| Large Scene | unavailable | Octree-GS / hierarchical runtime |

Unimplemented modes fail explicitly; they must never silently fall back to a different quality/performance profile.

## Artifact contract

Backends may emit multiple artifacts:

- Gaussian PLY;
- backend checkpoint;
- mesh;
- advanced runtime asset.

The publisher selects supported artifacts, validates them, and only then publishes stable project-level outputs. Backends never directly overwrite a user's final output file.

## Next implementation milestones

1. Add runtime health probing and backend recommendation.
2. Integrate this contract into the Rust/Tauri pipeline.
3. Vendor or pin the gsplat runtime reproducibly.
4. Add progress parsing for gsplat training output.
5. Add FastGS as the first independent high-speed backend.
6. Add geometry and large-scene artifact types to the desktop viewer/exporter.
