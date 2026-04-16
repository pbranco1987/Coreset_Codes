# API Reference

**Canonical reference for every public symbol in the `coreset_selection` package and the `scripts/` entry points.**

This reference is organised by subpackage. Each page documents every public function, class, and dataclass — with signatures, parameters, returns, examples, and cross-references. A symbol is "public" if it is re-exported from a subpackage's `__init__.py`, or if it is an entry-point (`scripts/*.py` CLI).

## Conventions

Every entry follows the same format:

```
### <fully-qualified name>

**Kind:** function / class / dataclass / constant
**Source:** <file_path>:<line_number or symbol>

**Summary:** one-sentence description.

**Description:** 2–5 sentences explaining what it does, when to use it, and
how it fits into the pipeline. For algorithms, a manuscript equation or
theorem number is referenced where applicable.

**Signature:**
    def name(param1: Type, param2: Type = default) -> ReturnType

**Parameters:** table.

**Returns:** table.

**Raises:** exceptions with conditions (when relevant).

**Example:** compile-checked minimal snippet.

**See also:** cross-references.
```

## Reading Order

Pick the page that matches your task:

| If you want to… | Read |
|----------------|------|
| Build a replicate cache or load features | [data](./data.md) |
| Set up geographic constraints or quotas | [geo](./geo.md) |
| Enforce population- or municipality-share | [constraints](./constraints.md) |
| Compute MMD, Sinkhorn, SKL, or NystromLogDet | [objectives](./objectives.md) |
| Run NSGA-II or pick a Pareto representative | [optimization](./optimization.md) |
| Evaluate a coreset (Nystrom, kPCA, KRR, KPI, geo, etc.) | [evaluation](./evaluation.md) |
| Run one of the 8 baseline selection methods | [baselines](./baselines.md) |
| Orchestrate a full experiment (runner, saver) | [experiment_models](./experiment_models.md) |
| Train a VAE or fit PCA | [experiment_models](./experiment_models.md) |
| Use helper utilities (math, I/O, seeds, plotting) | [utils](./utils.md) |
| Invoke a script from the CLI | [scripts](./scripts.md) |

## Package Layout at a Glance

```
coreset_selection/
├── data/              Replicate caches, loaders, preprocessing, target-leakage guards
├── geo/               Geographic group info, KL utilities, quota projection
├── constraints/       Population-share & municipality-share soft/hard constraints
├── objectives/        SKL, MMD², Sinkhorn, NystromLogDet; unified computer
├── optimization/      NSGA-II operators, Pareto selection, knee detection
├── evaluation/        RawSpaceEvaluator, diagnostics, KPI stability, QoS, R11
├── baselines/         Uniform, k-means, herding, FF, RLS, k-DPP, KT, KKN
├── experiment/        ExperimentRunner, ResultsSaver, ParetoFrontData
├── models/            TabularVAE, VAETrainer, PCA utilities
├── utils/             median_sq_dist, seeds, plotting palette
└── scripts/           Package-internal helpers (see `scripts/` at project root for entry points)
```

## Finding Things Quickly

- **By manuscript equation** → [docs/METHODOLOGY.md](../METHODOLOGY.md) Implementation Mapping section.
- **By pipeline phase** → [scripts/PIPELINE.md](../../scripts/PIPELINE.md).
- **By symbol name** → `grep "^### " docs/api/*.md` in your shell, or GitHub's code search.

## Version

This reference corresponds to **`coreset_selection` v0.1.0** (source: `coreset_selection/__init__.py:61`). Any symbol listed here is part of the public API contract and is subject to the deprecation policy in [CHANGELOG.md](../CHANGELOG.md).
