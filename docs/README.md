# Documentation Index

Welcome to the Coreset_Codes documentation. This index provides **suggested reading paths** for four common audiences. Follow the sequence that matches your goal.

## Reading Paths by Audience

### 🎓 You want to understand the paper

You just read the manuscript and want to see how the theory maps to code.

1. [GETTING_STARTED.md](./GETTING_STARTED.md) — 10-minute orientation (what a coreset is, what the repo does).
2. [METHODOLOGY.md](./METHODOLOGY.md) — equations, algorithms, and *Implementation Mapping* pointing each equation to its Python file.
3. [EXPERIMENTS.md](./EXPERIMENTS.md) — full description of the 15 experimental configurations (R0–R14).
4. [api/](./api/index.md) — look up specific functions referenced in the paper.

### 🔁 You want to reproduce the manuscript results

You want to install, run the pipeline, and regenerate the numbers.

1. [GETTING_STARTED.md](./GETTING_STARTED.md) — quick orientation.
2. [INSTALL.md](./INSTALL.md) — environment setup, GPU notes, platform caveats.
3. [../scripts/PIPELINE.md](../scripts/PIPELINE.md) — end-to-end pipeline guide with CLI examples.
4. [EXPERIMENTS.md](./EXPERIMENTS.md) — experimental design and per-run config.
5. [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — when things go wrong.

### 🛠 You want to extend the code

You want to add a new baseline, objective, or constraint mode.

1. [GETTING_STARTED.md](./GETTING_STARTED.md) — minimum viable orientation.
2. [ARCHITECTURE.md](./ARCHITECTURE.md) — module-level design and dependency graph.
3. [api/index.md](./api/index.md) — full public-API reference (flagship document).
4. [../CONTRIBUTING.md](../CONTRIBUTING.md) — development setup, style guide, PR workflow, and step-by-step templates for adding a baseline/objective/constraint.

### 🐛 You are debugging a failed run

Something broke and you need to understand why.

1. [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — FAQ of common errors.
2. [DATA_PIPELINE.md](./DATA_PIPELINE.md) — what the data pipeline does (most errors originate here).
3. [api/data.md](./api/data.md) — look up the specific loader or preprocessor.
4. [CHANGELOG.md](./CHANGELOG.md) — check whether the issue is a known regression.

---

## Full Document Catalogue

### Entry-level
- [GETTING_STARTED.md](./GETTING_STARTED.md) — 10-minute quickstart.
- [INSTALL.md](./INSTALL.md) — detailed setup.

### Research & Theory
- [METHODOLOGY.md](./METHODOLOGY.md) — mathematical formulation, algorithms, implementation mapping.
- [EXPERIMENTS.md](./EXPERIMENTS.md) — experimental configurations (R0–R14).
- [EVALUATION_METRICS.md](./EVALUATION_METRICS.md) — metric definitions.
- [DATA_PIPELINE.md](./DATA_PIPELINE.md) — data loading, preprocessing, splits.

### Engineering & Development
- [ARCHITECTURE.md](./ARCHITECTURE.md) — module-level design.
- [api/](./api/index.md) — full public API reference (10 pages).
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — common errors and fixes.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — contribution guide.
- [CHANGELOG.md](./CHANGELOG.md) — version history.

### Pipeline
- [../scripts/PIPELINE.md](../scripts/PIPELINE.md) — script-level pipeline guide.
- [../examples/README.md](../examples/README.md) — Jupyter notebook tutorials.

---

## Convention

Every document is written with **absolute-minimum prerequisite knowledge**. Cross-references are marked and targets are kept current. If you spot a stale link or outdated section, please open an issue or submit a PR (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
