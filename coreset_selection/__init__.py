"""
Coreset Selection Package
=========================

Geographically-constrained Pareto coreset selection for multi-objective
optimization of distributional divergences.

This package provides:

- **config**: Configuration dataclasses and run specifications
- **data**: Data loading, preprocessing, and caching
- **geo**: Geographic constraint handling and KL divergence
- **models**: VAE for representation learning
- **objectives**: SKL, MMD², and Sinkhorn divergence computation
- **optimization**: NSGA-II multi-objective optimization
- **baselines**: Comparison methods (uniform, k-means, herding, etc.)
- **evaluation**: Coreset quality metrics
- **experiment**: Experiment orchestration and result saving
- **artifacts**: Manuscript figure and table generation
- **cli**: Command-line interface
- **runners**: Experiment runner orchestration (scenario, parallel, run_all)
- **constraints**: Constraint handling (calibration, proportionality)

Quick Start
-----------

1. Prepare replicate caches:
   
   ```bash
   python -m coreset_selection prep --n-replicates 10
   ```

2. Run an experiment:
   
   ```bash
   python -m coreset_selection run --run-id R1 --rep-id 0
   ```

3. Generate artifacts:
   
   ```bash
   python -m coreset_selection artifacts --runs-dir runs_out
   ```

Programmatic Usage
------------------

```python
from coreset_selection.config import ExperimentConfig, get_run_specs
from coreset_selection.experiment import run_single_experiment

# Load run specification
specs = get_run_specs()
cfg = apply_run_spec(base_config, specs['R1'], rep_id=0)

# Run experiment
result = run_single_experiment(cfg)
```
"""

__version__ = "0.1.0"
__author__ = "Coreset Selection Authors"

# Lazy imports for top-level convenience
def __getattr__(name):
    """Lazy import submodules."""
    if name == "config":
        from . import config
        return config
    elif name == "data":
        from . import data
        return data
    elif name == "geo":
        from . import geo
        return geo
    elif name == "models":
        from . import models
        return models
    elif name == "objectives":
        from . import objectives
        return objectives
    elif name == "optimization":
        from . import optimization
        return optimization
    elif name == "baselines":
        from . import baselines
        return baselines
    elif name == "evaluation":
        from . import evaluation
        return evaluation
    elif name == "experiment":
        from . import experiment
        return experiment
    elif name == "artifacts":
        from . import artifacts
        return artifacts
    elif name == "utils":
        from . import utils
        return utils
    elif name == "cli":
        from . import cli
        return cli
    elif name == "runners":
        from . import runners
        return runners
    elif name == "constraints":
        from . import constraints
        return constraints
    elif name == "scripts":
        from . import scripts
        return scripts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "config",
    "data",
    "geo",
    "models",
    "objectives",
    "optimization",
    "baselines",
    "evaluation",
    "experiment",
    "artifacts",
    "utils",
    "cli",
    "runners",
    "constraints",
    "scripts",
]
