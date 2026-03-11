
# gismol: General Intelligent Systems Modelling Language

**gismol** is a Python implementation of the **Constrained Object Hierarchies (COH) 9‑tuple** framework, a unified mathematical foundation for designing, simulating, and learning in hierarchical, constraint‑driven intelligent systems.

## Features
- **Hierarchical Composition** – Build systems as directed acyclic graphs of COH objects.
- **Nine Core Components** – Full support for C, A, M, N, E, I, T, G, D.
- **Constraint Enforcement** – Identity invariants, trigger rules, goal optimization, and daemon monitoring.
- **Neural Integration** – Wrappers for PyTorch modules enable learning and adaptation.
- **Simulation** – Discrete‑time and event‑driven simulation with an event bus.
- **Categorical Operations** – Product (implemented), coproduct & exponential (stubs).
- **Cross‑Domain Examples** – Gridworld and more in `examples/`.

## Installation
```bash
pip install -e .
```

## Quick Start
```python
from gismol import COH, Simulator

# Create a simple agent
agent = COH(name='Agent', attributes={'x': 0, 'y': 0})
agent.methods['move_right'] = lambda s: ({'x': s['x'] + 1, 'y': s['y']}, -1)

# Run simulation
sim = Simulator(agent, dt=1.0, max_steps=10)
sim.run()
```

See the `examples/` directory for more advanced usage.

## License
This project is licensed under the MIT License – see the `LICENSE` file for details.
