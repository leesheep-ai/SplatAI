# SplatAI

SplatAI is an experimental, local-first Gaussian Splatting workstation focused on high-quality reconstruction, fast training, advanced camera models, geometry-aware reconstruction, and scalable rendering.

> SplatAI is an independent project. It is based in part on ideas and workflows from [OOOSplat](https://github.com/ooolabdev/ooosplat), but it is not affiliated with or endorsed by ooolabdev.

## Direction

SplatAI is being built around a pluggable training/runtime architecture instead of coupling the desktop pipeline to one Gaussian Splatting implementation.

Planned backend order:

1. **Brush compatibility backend** — broad hardware compatibility and fallback path.
2. **gsplat + MCMC** — primary high-quality NVIDIA/CUDA path.
3. **3DGUT** — advanced camera models and distortion-aware rendering/training.
4. **FastGS** — accelerated training mode.
5. **3DGRUT + NHT** — ultimate-quality research/experimental mode.
6. **Fast-PGSR** — geometry-oriented reconstruction and mesh workflows.
7. **HiGS / Octree-GS** — high-throughput rendering and large-scene LOD.

## Product modes

The UI will expose product-level modes rather than implementation details:

- **Compatible** — Brush fallback.
- **Fast** — FastGS-oriented training.
- **Quality** — gsplat + MCMC.
- **Ultimate** — 3DGUT / 3DGRUT / NHT where supported.
- **Geometry** — Fast-PGSR and mesh-oriented output.
- **Large Scene** — hierarchical/LOD-oriented execution.

## Architecture principle

```text
Video / Images
      |
Frame preparation + masks
      |
Camera reconstruction / dataset normalization
      |
TrainingBackend
  |-- Brush
  |-- gsplat MCMC
  |-- 3DGUT
  |-- FastGS
  |-- 3DGRUT + NHT
  `-- Fast-PGSR
      |
Artifact Publisher
  |-- PLY
  |-- Checkpoint
  |-- Mesh
  `-- advanced runtime assets
      |
Viewer / HiGS / LOD
```

The contract between the desktop application and training engines is intentionally process-based and artifact-based. Python/CUDA research stacks must not leak dependency complexity into the Tauri/Rust core.

## Current status

**Phase 1 — backend foundation**

The first implementation milestone establishes:

- a stable backend contract;
- Brush and gsplat adapter boundaries;
- MCMC as the default gsplat strategy;
- optional 3DGUT configuration;
- runtime capability/health reporting;
- deterministic artifact publishing;
- CI tests for backend selection and command generation.

## Upstream attribution

OOOSplat is licensed under Apache-2.0 and has a separate trademark policy. SplatAI uses a distinct project/product name and will retain required attribution for any OOOSplat-derived source incorporated into this repository.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
