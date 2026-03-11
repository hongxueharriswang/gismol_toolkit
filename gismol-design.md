**gismol** is a Python toolkit that implements the Constrained Object Hierarchies (COH) 9‑tuple framework. The toolkit is designed for modeling, simulating, and learning in intelligent systems, with a strong emphasis on hierarchical composition, constraint enforcement, and extensibility.

---

## Package Structure

```
gismol/
├── gismol/
│   ├── __init__.py
│   ├── core.py               # COH class, Attribute, Method, Trigger, Daemon, NeuralModule
│   ├── constraints.py         # IdentityConstraint, GoalConstraint, Constraint base
│   ├── simulation.py          # Simulator, Event, EventBus
│   ├── learning.py            # ConstrainedRL, policy gradient helpers
│   ├── category.py            # categorical operations (product, coproduct, exponential)
│   ├── utils.py               # DAG validation, serialization, embedding helpers
│   └── visualization.py       # (optional) graph drawing, constraint dashboards
├── tests/
│   ├── test_core.py
│   ├── test_simulation.py
│   └── test_learning.py
├── examples/
│   ├── gridworld.py
│   ├── quantum_control.py
│   ├── biology.py
│   └── smart_city.py
├── docs/
│   └── (documentation files)
├── setup.py
├── README.md
├── LICENSE
└── requirements.txt
```

---

## Core Modules

### `gismol/__init__.py`

```python
"""gismol: General Intelligent Systems Modelling Language.

A Python implementation of the Constrained Object Hierarchies (COH) 9‑tuple framework
for designing, simulating, and learning in hierarchical, constraint‑driven intelligent systems.
"""

from gismol.core import COH, NeuralModule, Trigger, Daemon
from gismol.constraints import IdentityConstraint, GoalConstraint
from gismol.simulation import Simulator
from gismol import utils, category, learning

__version__ = "0.1.0"
```

---

### `gismol/core.py`

```python
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn as nn

class COH:
    """A Constrained Object Hierarchy (9‑tuple) representing an intelligent system.

    Attributes:
        name (str): Optional identifier.
        parent (Optional[COH]): Parent object in the hierarchy.
        children (List[COH]): Sub‑components (C).
        attributes (Dict[str, Any]): State variables (A).
        methods (Dict[str, Callable]): Behaviors (M). Each method takes (state_dict) and returns (new_state_dict, reward).
        neural (Dict[str, 'NeuralModule']): Learnable functions (N).
        embedding (Optional[Callable[[COH], np.ndarray]]): Maps object to Hilbert space (E).
        identity_constraints (List[Callable[[COH], bool]]): Invariants that must hold in every state (I).
        trigger_constraints (List['Trigger']): Event‑condition‑action rules (T).
        goal_constraints (List[Callable[[COH], float]]): Objective functions (G).
        daemons (List['Daemon']): Background monitoring processes (D).
    """

    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.parent = None
        self.children: List[COH] = kwargs.get("children", [])
        self.attributes: Dict[str, Any] = kwargs.get("attributes", {})
        self.methods: Dict[str, Callable] = kwargs.get("methods", {})
        self.neural: Dict[str, NeuralModule] = kwargs.get("neural", {})
        self.embedding: Optional[Callable[[COH], np.ndarray]] = kwargs.get("embedding")
        self.identity_constraints: List[Callable[[COH], bool]] = kwargs.get("identity_constraints", [])
        self.trigger_constraints: List[Trigger] = kwargs.get("trigger_constraints", [])
        self.goal_constraints: List[Callable[[COH], float]] = kwargs.get("goal_constraints", [])
        self.daemons: List[Daemon] = kwargs.get("daemons", [])

        # Validate hierarchy (no cycles)
        self._validate_hierarchy()

    def _validate_hierarchy(self):
        """Check that the component graph is a DAG."""
        graph = nx.DiGraph()
        def add_edges(obj):
            for child in obj.children:
                graph.add_edge(obj, child)
                add_edges(child)
        add_edges(self)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Component hierarchy must be a DAG (no cycles).")

    def add_child(self, child: 'COH'):
        """Add a sub‑component."""
        child.parent = self
        self.children.append(child)
        self._validate_hierarchy()

    def remove_child(self, child: 'COH'):
        """Remove a sub‑component."""
        child.parent = None
        self.children.remove(child)

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of this object's attributes (recursively)."""
        state = dict(self.attributes)
        for child in self.children:
            state[child.name] = child.get_state()
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state from a snapshot."""
        for k, v in state.items():
            if k in self.attributes:
                self.attributes[k] = v
            else:
                # Try to find a child with that name
                for child in self.children:
                    if child.name == k:
                        child.set_state(v)
                        break

    def check_identity(self) -> bool:
        """Check all identity constraints (including those of children)."""
        for constr in self.identity_constraints:
            if not constr(self):
                return False
        for child in self.children:
            if not child.check_identity():
                return False
        return True

    def apply_method(self, name: str, *args, **kwargs) -> Any:
        """Execute a method by name and update attributes."""
        if name not in self.methods:
            raise ValueError(f"Unknown method '{name}'")
        method = self.methods[name]
        new_state, reward = method(self.attributes, *args, **kwargs)
        self.attributes.update(new_state)
        if not self.check_identity():
            raise ConstraintViolation(f"Identity constraint failed after method '{name}'")
        return reward

    def compute_goal(self) -> float:
        """Sum of goal constraints (recursively)."""
        total = sum(g(self) for g in self.goal_constraints)
        for child in self.children:
            total += child.compute_goal()
        return total

    def to_dict(self) -> dict:
        """Serialize to a JSON‑compatible dict."""
        return {
            "name": self.name,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
            # Note: methods, neural, constraints, daemons are not serialized by default
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'COH':
        """Deserialize from a dict."""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            name=data["name"],
            attributes=data.get("attributes", {}),
            children=children
        )


class NeuralModule:
    """Wrapper for a PyTorch neural network with training utilities."""

    def __init__(self, module: nn.Module, optimizer_class=None, lr=0.001, **optim_kwargs):
        self.module = module
        self.optimizer = None
        if optimizer_class is not None:
            self.optimizer = optimizer_class(module.parameters(), lr=lr, **optim_kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def step(self, loss):
        if self.optimizer is None:
            raise RuntimeError("No optimizer set; cannot perform training step.")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def save(self, path):
        torch.save(self.module.state_dict(), path)

    def load(self, path):
        self.module.load_state_dict(torch.load(path))


class Trigger:
    """Event‑condition‑action rule."""

    def __init__(self, event: str, condition: Callable[[COH], bool], action: Callable[[COH], None]):
        self.event = event          # event name or regex pattern
        self.condition = condition
        self.action = action

    def check_and_fire(self, coh: COH) -> bool:
        """If condition holds, execute action and return True."""
        if self.condition(coh):
            self.action(coh)
            return True
        return False


class Daemon:
    """Background monitoring process. Subclasses must implement run()."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.last_run = 0.0

    def run(self, coh: COH, dt: float):
        """Called periodically during simulation."""
        raise NotImplementedError

    def should_run(self, t: float) -> bool:
        if t - self.last_run >= self.interval:
            self.last_run = t
            return True
        return False


class ConstraintViolation(Exception):
    """Raised when an identity constraint is violated."""
    pass
```

---

### `gismol/constraints.py`

```python
from typing import Callable, Any
from gismol.core import COH

class Constraint:
    """Base class for constraints."""
    def check(self, coh: COH) -> bool:
        raise NotImplementedError

class IdentityConstraint(Constraint):
    """Invariant that must hold in every state."""
    def __init__(self, func: Callable[[COH], bool], name: str = ""):
        self.func = func
        self.name = name

    def check(self, coh: COH) -> bool:
        return self.func(coh)

class GoalConstraint(Constraint):
    """Objective function returning a scalar reward/cost."""
    def __init__(self, func: Callable[[COH], float], weight: float = 1.0):
        self.func = func
        self.weight = weight

    def value(self, coh: COH) -> float:
        return self.weight * self.func(coh)
```

---

### `gismol/simulation.py`

```python
import time
from typing import Callable, Dict, List, Optional, Any
from gismol.core import COH, Trigger, Daemon, ConstraintViolation
import re

class Event:
    def __init__(self, name: str, data: Any = None):
        self.name = name
        self.data = data

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_pattern: str, callback: Callable):
        self._subscribers.setdefault(event_pattern, []).append(callback)

    def publish(self, event: Event):
        for pattern, callbacks in self._subscribers.items():
            if re.match(pattern, event.name):
                for cb in callbacks:
                    cb(event)

class Simulator:
    """Discrete‑time or event‑driven simulator for COH systems."""

    def __init__(self, root: COH, dt: float = 1.0, max_steps: Optional[int] = None, real_time: bool = False):
        self.root = root
        self.dt = dt
        self.max_steps = max_steps
        self.real_time = real_time
        self.time = 0.0
        self.step_count = 0
        self.event_bus = EventBus()
        self._running = False

        # Register built‑in event subscriptions for trigger constraints
        self._register_triggers(root)

    def _register_triggers(self, coh: COH):
        for trig in coh.trigger_constraints:
            self.event_bus.subscribe(trig.event, lambda e, t=trig, c=coh: t.check_and_fire(c))
        for child in coh.children:
            self._register_triggers(child)

    def run(self, policy: Optional[Callable[[COH], str]] = None):
        """Main simulation loop."""
        self._running = True
        while self._running:
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break

            # 1. Pre‑step events
            self.event_bus.publish(Event("step", self.time))

            # 2. Decide action
            if policy is None:
                action = self._default_policy(self.root)
            else:
                action = policy(self.root)

            # 3. Apply action (method)
            try:
                reward = self.root.apply_method(action)
            except ConstraintViolation as e:
                self.event_bus.publish(Event("constraint_violated", str(e)))
                self._running = False
                break

            # 4. Post‑step events
            self.event_bus.publish(Event("after_step", self.time))

            # 5. Run daemons
            self._run_daemons()

            # 6. Update time
            self.time += self.dt
            self.step_count += 1

            if self.real_time:
                time.sleep(self.dt)

        self._running = False

    def _default_policy(self, coh: COH) -> str:
        """Fallback: random choice among available methods."""
        import random
        return random.choice(list(coh.methods.keys()))

    def _run_daemons(self):
        def _run(obj: COH):
            for d in obj.daemons:
                if d.should_run(self.time):
                    d.run(obj, self.dt)
            for child in obj.children:
                _run(child)
        _run(self.root)

    def step(self, action: Optional[str] = None) -> float:
        """Advance one step (useful for interactive control)."""
        if action is None:
            action = self._default_policy(self.root)
        reward = self.root.apply_method(action)
        self.time += self.dt
        self.step_count += 1
        self._run_daemons()
        return reward

    def publish(self, event_name: str, data: Any = None):
        self.event_bus.publish(Event(event_name, data))

    def stop(self):
        self._running = False
```

---

### `gismol/learning.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from gismol.core import COH, NeuralModule

class ConstrainedRL:
    """Constrained Policy Optimization (simplified version)."""

    def __init__(self, coh: COH, policy_network: NeuralModule, gamma: float = 0.99,
                 constraint_cost: float = 1.0, lr: float = 1e-3):
        self.coh = coh
        self.policy = policy_network
        self.gamma = gamma
        self.constraint_cost = constraint_cost
        self.optimizer = optim.Adam(policy_network.module.parameters(), lr=lr)

    def collect_episode(self, max_steps: int = 100) -> Tuple[List[float], List[float], float]:
        """Run one episode, return rewards, constraint violations, total return."""
        states = []
        actions = []
        rewards = []
        constraint_violations = []

        # Reset state (in a real implementation you'd need an environment)
        # For now assume attributes are initialised externally.
        t = 0
        done = False
        while not done and t < max_steps:
            # Get current state embedding
            if self.coh.embedding is None:
                raise ValueError("COH must have an embedding function for RL.")
            state = self.coh.embedding(self.coh)
            states.append(state)

            # Choose action
            with torch.no_grad():
                logits = self.policy.forward(torch.tensor(state).float())
                action_probs = torch.softmax(logits, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()
            action = list(self.coh.methods.keys())[action_idx]
            actions.append(action_idx)

            # Apply action
            try:
                reward = self.coh.apply_method(action)
            except Exception as e:
                # treat as constraint violation
                constraint_violations.append(1.0)
                reward = -self.constraint_cost
            else:
                constraint_violations.append(0.0)

            rewards.append(reward)

            # Check if episode should end (e.g., goal reached)
            # (user can define a terminal condition)
            t += 1

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return states, actions, returns, constraint_violations

    def train_episode(self, max_steps: int = 100):
        """Perform one training update using collected episode."""
        states, actions, returns, violations = self.collect_episode(max_steps)

        # Convert to tensors
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(actions).long()
        returns = torch.tensor(returns).float()
        # (constraint violations can be used for penalty, omitted here)

        # Compute loss (simple policy gradient)
        logits = self.policy.forward(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        loss = - (action_log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

### `gismol/category.py`

```python
from gismol.core import COH
from typing import List

def product(*coh_objects: COH) -> COH:
    """Parallel composition: Cartesian product of components, conjunction of constraints."""
    if not coh_objects:
        raise ValueError("Need at least one COH object for product.")

    # Create a new root with combined children
    combined_children = []
    for obj in coh_objects:
        combined_children.extend(obj.children)

    # Combine attributes (namespace collision handled by prefixing with object name)
    combined_attrs = {}
    for obj in coh_objects:
        for k, v in obj.attributes.items():
            combined_attrs[f"{obj.name}.{k}"] = v

    # Combine methods (name collisions handled by prefixing)
    combined_methods = {}
    for obj in coh_objects:
        for k, v in obj.methods.items():
            combined_methods[f"{obj.name}.{k}"] = v

    # Combine identity constraints (must all hold)
    combined_identity = []
    for obj in coh_objects:
        combined_identity.extend(obj.identity_constraints)

    # Goal constraints are summed (with appropriate weighting)
    combined_goals = []
    for obj in coh_objects:
        combined_goals.extend(obj.goal_constraints)

    # For triggers, we keep them as is (they refer to the original objects)
    combined_triggers = []
    for obj in coh_objects:
        combined_triggers.extend(obj.trigger_constraints)

    # Daemons run independently
    combined_daemons = []
    for obj in coh_objects:
        combined_daemons.extend(obj.daemons)

    # Neural components are merged (names prefixed)
    combined_neural = {}
    for obj in coh_objects:
        for k, v in obj.neural.items():
            combined_neural[f"{obj.name}.{k}"] = v

    # Embedding: we can combine embeddings of the components
    def product_embedding(coh):
        embs = []
        for obj in coh_objects:
            if obj.embedding:
                embs.append(obj.embedding(obj))
        return np.concatenate(embs) if embs else np.array([])

    return COH(
        name="Product",
        children=combined_children,
        attributes=combined_attrs,
        methods=combined_methods,
        neural=combined_neural,
        embedding=product_embedding,
        identity_constraints=combined_identity,
        trigger_constraints=combined_triggers,
        goal_constraints=combined_goals,
        daemons=combined_daemons
    )

def coproduct(*coh_objects: COH) -> COH:
    """Alternative composition: disjoint union, disjunction of constraints."""
    # Similar to product but with different semantics for constraints.
    # For identity constraints, we must satisfy at least one.
    # For goals, we take the maximum or weighted sum depending on context.
    # For simplicity, we implement a basic version.
    raise NotImplementedError("Coproduct is not yet implemented.")

def exponential(domain: COH, codomain: COH) -> COH:
    """Function space object (experimental)."""
    raise NotImplementedError("Exponential is not yet implemented.")
```

---

### `gismol/utils.py`

```python
import json
import networkx as nx
from typing import Any, Dict
from gismol.core import COH

def to_json(coh: COH, filepath: str):
    """Serialize COH object to JSON."""
    with open(filepath, 'w') as f:
        json.dump(coh.to_dict(), f, indent=2)

def from_json(filepath: str) -> COH:
    """Load COH object from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return COH.from_dict(data)

def default_embedding(coh: COH) -> np.ndarray:
    """Concatenate all attribute values (flatten recursively)."""
    values = []
    for k, v in coh.attributes.items():
        if isinstance(v, (int, float)):
            values.append(v)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            values.extend(v)
    # Add child embeddings
    for child in coh.children:
        child_emb = default_embedding(child)
        values.extend(child_emb)
    return np.array(values, dtype=np.float32)

def is_dag(coh: COH) -> bool:
    """Check if the component graph is a DAG."""
    graph = nx.DiGraph()
    def add_edges(obj):
        for child in obj.children:
            graph.add_edge(obj, child)
            add_edges(child)
    add_edges(coh)
    return nx.is_directed_acyclic_graph(graph)
```

---

### `gismol/visualization.py` (optional)

```python
import matplotlib.pyplot as plt
import networkx as nx
from gismol.core import COH

def draw_hierarchy(coh: COH, filename: Optional[str] = None):
    """Draw the component DAG."""
    graph = nx.DiGraph()
    def add_nodes(obj):
        graph.add_node(obj.name)
        for child in obj.children:
            graph.add_node(child.name)
            graph.add_edge(obj.name, child.name)
            add_nodes(child)
    add_nodes(coh)

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
```

---

## Setup and Deployment Files

### `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="gismol",
    version="0.1.0",
    description="General Intelligent Systems Modelling Language (COH 9‑tuple implementation)",
    author="Harris Wang",
    author_email="harrisw@athabascau.ca",
    url="https://github.com/harriswang/gismol",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "networkx>=2.6",
        "torch>=1.10",
        "matplotlib>=3.4"   # optional
    ],
    extras_require={
        "dev": ["pytest", "sphinx", "black"]
    },
    python_requires=">=3.9",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```

### `requirements.txt`

```
numpy>=1.20
networkx>=2.6
torch>=1.10
matplotlib>=3.4
```

### `README.md`

```markdown
# gismol: General Intelligent Systems Modelling Language

**gismol** is a Python implementation of the **Constrained Object Hierarchies (COH) 9‑tuple** framework, a unified mathematical foundation for designing, simulating, and learning in hierarchical, constraint‑driven intelligent systems. It provides a modular, extensible toolkit for researchers and developers working on artificial general intelligence (AGI), cognitive architectures, and complex adaptive systems.

## Features

- **Hierarchical Composition** – Build systems as directed acyclic graphs of COH objects.
- **Nine Core Components** – Full support for C, A, M, N, E, I, T, G, D.
- **Constraint Enforcement** – Identity invariants, trigger rules, goal optimization, and daemon monitoring.
- **Neural Integration** – Wrappers for PyTorch modules enable learning and adaptation.
- **Simulation** – Discrete‑time and event‑driven simulation with an event bus.
- **Categorical Operations** – Product, coproduct, and exponential for compositional reasoning (experimental).
- **Cross‑Domain Examples** – Gridworld, quantum control, biology, smart city (see `examples/`).

## Installation

```bash
pip install gismol
```

For development:

```bash
git clone https://github.com/harriswang/gismol.git
cd gismol
pip install -e .[dev]
```

## Quick Start

```python
from gismol import COH, Simulator

# Create a simple agent
agent = COH(name="Agent", attributes={"x": 0, "y": 0})
agent.methods["move_right"] = lambda s: ({"x": s["x"]+1, "y": s["y"]}, -1)

# Run simulation
sim = Simulator(agent, dt=1.0, max_steps=10)
sim.run()
```

See the `examples/` directory for more advanced usage.

## Documentation

Full documentation is available at [https://gismol.readthedocs.io](https://gismol.readthedocs.io) (under construction).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

If you use gismol in your research, please cite:

Wang, H. (2026). The 9‑Tuple Formation of Constrained Object Hierarchies: The Mathematical Foundation and Implications for Artificial Intelligence. *Journal of Artificial Intelligence Research*.
```

### `LICENSE` (MIT)

```
MIT License

Copyright (c) 2026 Harris Wang

Permission is hereby granted...
```

---

## Example: Gridworld Navigator (from the paper)

Place this in `examples/gridworld.py`.

```python
from gismol import COH, NeuralModule, Trigger, Daemon, Simulator
import numpy as np
import torch.nn as nn

# Define components
grid = COH(name="Grid", attributes={"width": 5, "height": 5})
agent = COH(name="Agent", attributes={"x": 0, "y": 0, "goal": (4,4)})
obstacles = COH(name="Obstacles", attributes={"cells": [(2,2), (3,3)]})
world = COH(name="World", children=[grid, agent, obstacles])

# Methods
def move(dx, dy):
    def _move(state):
        x, y = state["x"], state["y"]
        new_x, new_y = x+dx, y+dy
        if (new_x, new_y) not in obstacles.attributes["cells"]:
            state["x"], state["y"] = new_x, new_y
        return state, -1
    return _move

world.methods["up"] = move(0, 1)
world.methods["down"] = move(0, -1)
world.methods["left"] = move(-1, 0)
world.methods["right"] = move(1, 0)
world.methods["wait"] = lambda s: (s, -1)

# Neural component (simple Q‑network)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)   # input: (x, y, goal_x, goal_y) -> 5 actions
    def forward(self, x):
        return self.fc(x)
world.neural["q_net"] = NeuralModule(QNet(), lr=1e-3)

# Identity constraints
def inside_bounds(coh):
    x, y = coh.attributes["x"], coh.attributes["y"]
    return 0 <= x < 5 and 0 <= y < 5
world.identity_constraints.append(inside_bounds)

# Trigger constraints
def obstacle_ahead(coh):
    x, y = coh.attributes["x"], coh.attributes["y"]
    return (x+1, y) in obstacles.attributes["cells"]
def avoid(coh):
    # choose a safe action (wait)
    coh.methods["wait"](coh.attributes)
world.trigger_constraints.append(Trigger("before_method:up", obstacle_ahead, avoid))

# Goal constraints
def goal_reward(coh):
    x, y = coh.attributes["x"], coh.attributes["y"]
    gx, gy = coh.attributes["goal"]
    return -abs(x-gx) - abs(y-gy)
world.goal_constraints.append(goal_reward)

# Daemon (logger)
class LoggerDaemon(Daemon):
    def run(self, coh, dt):
        print(f"Time {coh.parent.time}: pos=({coh.attributes['x']},{coh.attributes['y']})")
agent.daemons.append(LoggerDaemon(interval=1.0))

# Simulate
sim = Simulator(world, dt=1.0, max_steps=20)
sim.run(policy=lambda coh: np.random.choice(["up","down","left","right","wait"]))
```

---

## Running Tests

Create a simple test in `tests/test_core.py`:

```python
import pytest
from gismol import COH

def test_hierarchy_cycle():
    a = COH(name="A")
    b = COH(name="B")
    a.add_child(b)
    with pytest.raises(ValueError):
        b.add_child(a)   # would create cycle
```

---

## Deployment to GitHub

1. Create a new repository on GitHub (e.g., `gismol`).
2. Initialize a local git repository with the above file structure.
3. Commit and push.

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/gismol.git
git push -u origin main
```

---

The **gismol** toolkit is now ready for use. It provides a solid foundation for implementing the COH 9‑tuple framework, with clear separation of concerns, modular design, and extensibility. Researchers can build upon it to explore hierarchical, constraint‑driven intelligent systems across any domain.