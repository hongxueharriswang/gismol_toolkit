

# Contributing to GISMOL Toolkit

Thanks for your interest in improving the **GISMOL Toolkit**! This guide explains how to set up your environment, follow project conventions, propose changes, and extend the system safely and consistently.

> **What is GISMOL?**  
> GISMOL (General Intelligent Systems Modelling Language) is a Python toolkit implementing the **Constrained Object Hierarchies (COH)** 9‑tuple framework for hierarchical, constraint‑aware, learnable, and simulable intelligent systems. 

***

## Table of contents

1.  scope--architecture-at-a-glance
2.  ways-to-contribute
3.  development-setup
4.  running-tests--coverage
5.  style-linting--types
6.  git-workflow-branches-commits--prs
7.  documentation--examples
8.  extending-gismol-recipes
9.  backwards-compatibility--deprecations
10. performance-tips
11. security--responsible-disclosure
12. code-of-conduct
13. license

***

## Scope & architecture at a glance

*   **COH 9‑tuple**:  
    Each system is an object $$O$$ with components **(C, A, M, N, E, I, T, G, D)** standing for Components, Attributes, Methods, Neural modules, Embedding, Identity constraints, Trigger constraints, Goal constraints, and Daemons. 

*   **Core modules** (high level):
    *   `gismol.core`: `COH`, `NeuralModule`, `Trigger`, `Daemon`, and `ConstraintViolation`. 
    *   `gismol.constraints`: `Constraint`, `IdentityConstraint`, `GoalConstraint`. 
    *   `gismol.simulation`: `Simulator`, `EventBus`, `Event`. 
    *   `gismol.learning`: `ConstrainedRL` (demo constrained-RL). 
    *   `gismol.category`: categorical composition (`product`; `coproduct`/`exponential` placeholders). 
    *   `gismol.utils`: serialization, DAG checking, default embeddings. 
    *   `gismol.visualization` (optional): hierarchy plots. 

*   **Key design points**:
    *   Hierarchies are DAGs; acyclicity is enforced. 
    *   **Identity constraints** are checked after method execution; violations raise `ConstraintViolation`. 
    *   **Trigger constraints** subscribe to and fire on events via `EventBus`. 
    *   **Goal constraints** are aggregated using `compute_goal()` for evaluation/learning. 
    *   Methods/neural modules/constraints are **not** serialized by `to_dict()`; they must be reattached after `from_dict()`. 

***

## Ways to contribute

*   Report bugs and propose enhancements via **Issues**.
*   Improve docs, tutorials, and examples in `examples/`. The spec references canonical examples such as `basic_agent.py`, `hierarchical_robot.py`, etc. (feel free to add more or refine them). 
*   Add new **constraints**, **daemons**, **composition operators**, **embeddings**, or **learning algorithms** (see #extending-gismol-recipes). 
*   Optimize performance, increase test coverage, or improve the developer experience.

***

## Development setup

> **Python version**: 3.9+ is required. 

1.  **Clone** the repository:
    ```bash
    git clone https://github.com/harriswatau/gismol_toolkit.git
    cd gismol_toolkit
    ```

2.  **Create & activate** a virtual environment (example with `venv`):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install** the package in editable mode with dev extras:
    ```bash
    pip install -e .[dev]
    ```
    > The runtime depends on `numpy`, `networkx`, and optionally `torch` (learning) and `matplotlib` (visualization). 

4.  **Optional GPU**: If you plan to work on `NeuralModule` / `ConstrainedRL`, install a CUDA-enabled PyTorch per your environment.

***

## Running tests & coverage

We use **pytest**.

```bash
pytest
pytest --maxfail=1 -q
pytest --cov=gismol --cov-report=term-missing
```

**Test philosophy**

*   Unit tests live under `tests/` mirroring the package structure.
*   Add focused tests for:
    *   Constraint satisfaction/violations,
    *   EventBus/Trigger flows,
    *   DAG invariants and (de)serialization boundaries,
    *   Category operations (e.g., `product`),
    *   Learning loops (smoke tests for `ConstrainedRL`), and
    *   Visualization (skip if `matplotlib` not installed). 

***

## Style, linting & types

To keep the codebase consistent and friendly:

*   **Formatter**: \[Black] (line length 88)
*   **Import sorting**: \[isort] (compatible with Black)
*   **Linter**: \[Ruff] or Flake8 (team preference)
*   **Typing**: `mypy` (prefer explicit types for public APIs)
*   **Docstrings**: Google- or NumPy-style with concise examples

We recommend **pre-commit hooks**. Add a `.pre-commit-config.yaml` with Black, isort, Ruff/Flake8, and mypy; then:

```bash
pre-commit install
pre-commit run --all-files
```

> Public functions/classes in `gismol.*` should have type hints and docstrings explaining how they interact with the COH 9‑tuple, constraints, and event system. 

***

## Git workflow: branches, commits & PRs

*   **Branches**:
    *   `main`: stable, releasable.
    *   feature branches from `main`: `feat/<short-slug>`
    *   fixes: `fix/<short-slug>`
    *   docs/tests/chore similarly.

*   **Commits**: Conventional style is appreciated:
        feat(core): add batched apply_method
        fix(simulation): prevent double firing on repeated events
        docs(constraints): clarify GoalConstraint weighting

*   **Pull Requests**:
    *   Keep PRs small and scoped; link to an Issue.
    *   Include/adjust tests and docs.
    *   Note any **API changes**, **performance impact**, or **back-compat** concerns.
    *   For features touching constraints/simulation/learning, include a short **design note** referencing the COH component(s) affected. 

***

## Documentation & examples

*   Keep the **Technical Specification** (`gismol-toolkit-specification.md`) as the single source of truth for concepts and module surfaces; update it when APIs evolve. 
*   Put runnable tutorial scripts under `examples/` (see titles referenced in the spec); ensure they run from a clean checkout with minimal setup. 
*   When adding new modules/APIs, include:
    *   A brief section in the spec,
    *   At least one example, and
    *   Minimal tests covering the intended usage.

***

## Extending GISMOL (recipes)

Below are vetted paths for extension. Please include tests and, where applicable, example snippets.

### 1) New **Identity/Goal** constraint

*   Location: `gismol/constraints.py` (or a new module under `gismol/constraints/`).
*   Identity constraints must return `bool`; goal constraints return a scalar and may be **weighted**. 

**Template**

```python
from gismol.core import COH
from gismol.constraints import Constraint, IdentityConstraint, GoalConstraint

def positive_balance_identity(coh: COH) -> bool:
    return coh.attributes.get("balance", 0) >= 0

def minimize_energy_goal(coh: COH) -> float:
    return -float(coh.attributes.get("energy", 0))

IDENTITY = IdentityConstraint(positive_balance_identity, name="positive_balance")
GOAL = GoalConstraint(minimize_energy_goal, weight=1.0)
```

**Tests**: verify `COH.check_identity()` and `compute_goal()` behavior (including recursive aggregation over children). 

***

### 2) New **Daemon**

*   Subclass `Daemon` and override `run(coh, dt)`. Daemons execute within the simulator loop according to their interval. 

```python
from gismol.core import Daemon

class LoggerDaemon(Daemon):
    def run(self, coh, dt):
        print(f"[t=?] {coh.name} -> {coh.attributes}")
```

**Tests**: step the simulator and assert side effects happen when `should_run` signals true. 

***

### 3) New **Trigger** (event‑condition‑action)

*   Build with `Trigger(event, condition, action)`; register via the simulator’s event bus or attach to `COH.trigger_constraints` for auto‑subscription. 

```python
from gismol.core import Trigger

def low_fuel(coh): return coh.attributes.get("fuel", 100) < 10
def refuel(coh): coh.attributes["fuel"] = 100

REFUEL_TRIGGER = Trigger("after_step", low_fuel, refuel)
```

**Tests**: publish an event (e.g., `"after_step"`) and assert the trigger fires only when the condition holds. 

***

### 4) New **Methods** on a `COH`

*   Methods update `attributes` and **must** return `(new_attributes, reward)`.  
    Identity constraints are checked **after** each method; violations must raise `ConstraintViolation`. 

```python
from gismol.core import COH

agent = COH(name="agent", attributes={"x": 0})
def move_right(state):
    x = state["x"] + 1
    return {"x": x}, -1.0  # reward example

agent.methods["move_right"] = move_right
```

**Tests**: ensure state transitions occur and constraints are enforced post‑update. 

***

### 5) **Category operations** (composition)

*   Implement or extend categorical combinators in `gismol/category.py`.  
    The provided `product(*coh_objects)` merges children, prefixes names to avoid collisions, conjuncts identity constraints, sums goals, merges triggers/daemons, and concatenates embeddings. 

**Tests**: verify attribute/method prefixing, constraint aggregation, and embedding concatenation.

***

### 6) **Learning** integrations

*   `NeuralModule` wraps a PyTorch module (optionally with an optimizer) and exposes `forward`, `train/eval`, `step`, `save/load`. 
*   `ConstrainedRL` is a **minimal** demo of constrained policy optimization. Consider adding algorithms (e.g., PPO/Lagrangian) using `COH.embedding` for state featurization. 

**Tests**: fast smoke tests (few steps, CPU‑only) that exercise policy selection and reward collection without relying on heavyweight training.

***

### 7) **Serialization & DAG utilities**

*   Use `to_json/from_json` for structural persistence; remember that methods, neural modules, and constraints are **not** serialized and must be re‑attached post‑load. 
*   Use `is_dag(coh)` to validate acyclicity before/after structural edits. 

***

### 8) **Visualization**

*   `draw_hierarchy(coh, filename=None)` renders the component DAG with NetworkX + Matplotlib; save to file or show interactively. 

***

## Backwards compatibility & deprecations

*   Follow **semantic versioning** for public APIs under `gismol.*`.
*   For breaking changes, add deprecation shims where feasible, update the spec and examples, and note migration steps in the PR.

***

## Performance tips

*   Minimize object churn in tight simulation loops; reuse arrays/tensors when possible.
*   Keep `Trigger` conditions side‑effect‑free and fast; push heavier work into actions/daemons.
*   Batch calls in learning integrations where practical (e.g., batched `embedding` computations).

***

## Security & responsible disclosure

If you discover a security or safety issue (e.g., malicious code path in event handling or deserialization), **do not** open a public Issue. Email the maintainers or use a private security advisory to coordinate a fix before public disclosure.

***

## Code of Conduct

We are committed to a welcoming, harassment‑free community. Be respectful, assume good faith, and focus on technical merit. Harassment or discrimination will not be tolerated. (A full Code of Conduct—e.g., Contributor Covenant—can be added to this repo and linked here.)

***

## License

By contributing, you agree that your contributions will be licensed under the repository’s license (see `LICENSE`).

***

### Final notes

*   The package targets Python **3.9+** and uses `numpy`, `networkx`, optional `torch` (for learning), and optional `matplotlib` (for visualization). 
*   The **Technical Specification** is a living document—please update it when you introduce or modify public concepts/APIs. 

***

Would you like me to open a PR with this as `CONTRIBUTING.md` and (optionally) add a ready‑to‑use `.pre-commit-config.yaml`, `pyproject.toml` lint/type configs, and a minimal GitHub Actions CI for tests?
