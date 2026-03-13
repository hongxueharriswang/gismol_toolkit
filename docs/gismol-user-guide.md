# GISMOL User Guide
## Modeling and Implementing Intelligent Systems with COH/GISMOL

**Version:** 0.1.0  
**Date:** March 2026  
**Author:** Harris Wang  
**License:** MIT  
**Repository:** [https://github.com/harriswatau/gismol_toolkit](https://github.com/harriswatau/gismol_toolkit)

Welcome to **GISMOL** (General Intelligent Systems Modelling Language), a Python toolkit for building, simulating, and learning in intelligent systems using the **Constrained Object Hierarchies (COH)** 9‑tuple framework. This guide will walk you through the concepts, provide hands‑on examples, and show you how to leverage GISMOL to create your own hierarchical, constraint‑driven systems.

---

## Table of Contents

1. [Introduction to COH and GISMOL](#introduction-to-coh-and-gismol)
2. [Installation and First Steps](#installation-and-first-steps)
3. [Core Concepts: The COH 9‑tuple in GISMOL](#core-concepts-the-coh-9tuple-in-gismol)
   - [Components (C)](#components-c)
   - [Attributes (A)](#attributes-a)
   - [Methods (M)](#methods-m)
   - [Neural Components (N)](#neural-components-n)
   - [Embedding (E)](#embedding-e)
   - [Identity Constraints (I)](#identity-constraints-i)
   - [Trigger Constraints (T)](#trigger-constraints-t)
   - [Goal Constraints (G)](#goal-constraints-g)
   - [Daemons (D)](#daemons-d)
4. [Building a COH System: Step‑by‑Step](#building-a-coh-system-step-by-step)
   - [Define the Hierarchy (C)](#define-the-hierarchy-c)
   - [Define State (A)](#define-state-a)
   - [Define Behaviors (M)](#define-behaviors-m)
   - [Add Constraints (I, T, G)](#add-constraints-i-t-g)
   - [Add Neural Components and Embedding (N, E)](#add-neural-components-and-embedding-n-e)
   - [Add Daemons (D)](#add-daemons-d)
5. [Simulating the System](#simulating-the-system)
   - [The Simulator Class](#the-simulator-class)
   - [Policies and Action Selection](#policies-and-action-selection)
   - [Event Handling](#event-handling)
   - [Accessing Simulation State](#accessing-simulation-state)
6. [Learning and Adaptation](#learning-and-adaptation)
   - [Using NeuralModule](#using-neuralmodule)
   - [Training with ConstrainedRL](#training-with-constrainedrl)
   - [Custom Learning Algorithms](#custom-learning-algorithms)
7. [Advanced Composition: Categorical Operations](#advanced-composition-categorical-operations)
   - [Product (Parallel Composition)](#product-parallel-composition)
   - [Coproduct and Exponential (Experimental)](#coproduct-and-exponential-experimental)
8. [Serialization and Persistence](#serialization-and-persistence)
   - [Saving to JSON](#saving-to-json)
   - [Loading from JSON](#loading-from-json)
9. [Visualization Tools](#visualization-tools)
   - [Drawing the Hierarchy](#drawing-the-hierarchy)
10. [Best Practices and Design Patterns](#best-practices-and-design-patterns)
11. [Troubleshooting Common Issues](#troubleshooting-common-issues)
12. [API Quick Reference](#api-quick-reference)
13. [Next Steps and Further Resources](#next-steps-and-further-resources)

---

## Introduction to COH and GISMOL

**Constrained Object Hierarchies (COH)** is a mathematical framework for representing intelligent systems as a 9‑tuple:

```
O = (C, A, M, N, E, I, T, G, D)
```

Each component captures a fundamental aspect of intelligence:
- **C (Components)** – Hierarchical decomposition.
- **A (Attributes)** – State variables.
- **M (Methods)** – Behaviors/actions.
- **N (Neural Components)** – Learnable functions.
- **E (Embedding)** – Semantic representation.
- **I (Identity Constraints)** – Invariants.
- **T (Trigger Constraints)** – Reactive rules.
- **G (Goal Constraints)** – Objectives.
- **D (Daemons)** – Continuous monitors.

**GISMOL** brings COH to life in Python. It provides:
- A clean, object‑oriented API for constructing COH objects.
- A simulation engine with discrete‑time and event‑driven modes.
- Integration with PyTorch for learning.
- Categorical composition operators.
- Tools for serialization and visualization.

---

## Installation and First Steps

### Installation

```bash
pip install gismol
```

For the latest development version:

```bash
git clone https://github.com/harriswang/gismol.git
cd gismol
pip install -e .
```

### Your First GISMOL Program

Create a file `hello_agent.py`:

```python
from gismol import COH, Simulator

# Create a simple agent with a position attribute
agent = COH(name="Walker", attributes={"x": 0})

# Define a method to move right
def move_right(state):
    state["x"] += 1
    return state, 0   # reward is zero

agent.methods["move"] = move_right

# Run a simulation for 5 steps
sim = Simulator(agent, dt=1.0, max_steps=5)
sim.run()   # uses random policy (only one action here)

print("Final position:", agent.attributes["x"])
```

Run it:
```bash
python hello_agent.py
```

You should see output like:
```
Final position: 5
```

Congratulations – you’ve just built and simulated your first COH system!

---

## Core Concepts: The COH 9‑tuple in GISMOL

### Components (C)

Components represent the hierarchical decomposition. In GISMOL, a COH object can have **children**, which are themselves COH objects. The parent‑child relationships must form a **directed acyclic graph (DAG)**.

```python
body = COH(name="Body")
sensor = COH(name="Sensor")
robot = COH(name="Robot", children=[body, sensor])
```

Children are accessible via the `children` attribute. The parent of an object is set automatically when added via `add_child()`.

### Attributes (A)

Attributes store the state of an object. They are stored in a dictionary `attributes`. Values can be any JSON‑serializable type (numbers, strings, lists, dicts).

```python
agent.attributes["position"] = (0, 0)
agent.attributes["energy"] = 100.0
```

### Methods (M)

Methods define the behaviors. A method is a callable that receives the current `attributes` dictionary (and optional arguments) and returns a tuple `(new_attributes, reward)`. The reward is a scalar used for learning.

```python
def move_up(state):
    x, y = state["position"]
    state["position"] = (x, y+1)
    return state, -1   # step cost

agent.methods["up"] = move_up
```

Methods can also call methods on children or access neural components.

### Neural Components (N)

Neural components are learnable functions wrapped in `NeuralModule`. They integrate with PyTorch.

```python
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

policy = NeuralModule(PolicyNet(), optimizer_class=torch.optim.Adam, lr=1e-3)
agent.neural["policy"] = policy
```

### Embedding (E)

The embedding function maps a COH object (including its children) to a vector (numpy array) – typically used as input to neural networks.

```python
def my_embedding(coh):
    # Flatten attributes into a vector
    pos = coh.attributes["position"]
    energy = coh.attributes["energy"]
    return np.array([pos[0], pos[1], energy], dtype=np.float32)

agent.embedding = my_embedding
```

If you don’t provide an embedding, the toolkit supplies a simple default via `utils.default_embedding`.

### Identity Constraints (I)

Identity constraints are predicates that must hold in **every** state. They are checked after each method execution.

```python
def within_bounds(coh):
    x, y = coh.attributes["position"]
    return 0 <= x < 10 and 0 <= y < 10

agent.identity_constraints.append(within_bounds)
```

If a constraint fails, a `ConstraintViolation` exception is raised (unless caught by a daemon).

### Trigger Constraints (T)

Trigger constraints are **event‑condition‑action** rules. They are evaluated when a matching event occurs.

```python
from gismol import Trigger

def low_battery_condition(coh):
    return coh.attributes["energy"] < 20

def recharge_action(coh):
    coh.attributes["energy"] = 100

trigger = Trigger("after_step", low_battery_condition, recharge_action)
agent.trigger_constraints.append(trigger)
```

Events are published by the simulator (e.g., `"step"`, `"after_step"`, custom events).

### Goal Constraints (G)

Goal constraints define the objectives. They return a scalar reward (higher is better) for a given state.

```python
def goal_reward(coh):
    # reward = distance to origin (negative)
    x, y = coh.attributes["position"]
    return - (abs(x) + abs(y))

agent.goal_constraints.append(goal_reward)
```

Multiple goals can be combined (summed) with weights.

### Daemons (D)

Daemons are background processes that run periodically. They can monitor state, log data, or intervene. Subclass `Daemon` and implement `run()`.

```python
from gismol import Daemon

class Logger(Daemon):
    def run(self, coh, dt):
        print(f"Time {coh.parent.time}: pos={coh.attributes['position']}")

agent.daemons.append(Logger(interval=2.0))   # run every 2 time units
```

Daemons are executed by the simulator at the end of each time step, if their interval has elapsed.

---

## Building a COH System: Step‑by‑Step

Let’s build a more complex system: a **smart home** with a thermostat and lights.

### Define the Hierarchy (C)

```python
from gismol import COH

# Create components
thermostat = COH(name="Thermostat")
lights = COH(name="Lights")
home = COH(name="Home", children=[thermostat, lights])
```

### Define State (A)

```python
thermostat.attributes["temperature"] = 20.0
thermostat.attributes["target"] = 22.0
thermostat.attributes["heater_on"] = False

lights.attributes["bedroom"] = False
lights.attributes["livingroom"] = False
```

### Define Behaviors (M)

```python
def update_thermostat(state):
    # Simple hysteresis
    if state["temperature"] < state["target"] - 1:
        state["heater_on"] = True
    elif state["temperature"] > state["target"] + 1:
        state["heater_on"] = False

    if state["heater_on"]:
        state["temperature"] += 0.1
    else:
        state["temperature"] -= 0.05
    return state, -1 if state["heater_on"] else 0   # penalty for heating

thermostat.methods["update"] = update_thermostat

def toggle_light(state, room):
    state[room] = not state[room]
    return state, -1   # small cost for switching
lights.methods["toggle"] = toggle_light
```

### Add Constraints (I, T, G)

**Identity Constraint:** temperature must never exceed 30°C (safety).

```python
def safe_temp(coh):
    return coh.attributes["temperature"] <= 30.0

thermostat.identity_constraints.append(safe_temp)
```

**Trigger:** If lights are left on after 10pm (simulated time), turn them off.

```python
def late_hour_condition(coh):
    # Assume parent (home) has a "hour" attribute
    hour = coh.parent.attributes.get("hour", 0)
    return hour >= 22

def turn_off_lights(coh):
    for room in coh.attributes:
        if isinstance(coh.attributes[room], bool):
            coh.attributes[room] = False

lights.trigger_constraints.append(
    Trigger("step", late_hour_condition, turn_off_lights)
)
```

**Goal:** Minimize energy consumption (heater + lights).

```python
def energy_goal(coh):
    cost = 0
    if coh.attributes.get("heater_on", False):
        cost += 1
    # For lights, count how many are on
    for k, v in coh.attributes.items():
        if isinstance(v, bool) and v:
            cost += 1
    return -cost   # negative because we minimize

home.goal_constraints.append(energy_goal)
```

### Add Neural Components and Embedding (N, E)

Suppose we want to learn a policy to set the thermostat target optimally. We'll add a neural component to the thermostat.

```python
import torch.nn as nn
from gismol import NeuralModule

class SetpointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)   # input: (current temp, hour) -> new target

    def forward(self, x):
        return self.fc(x)

setpoint_nn = NeuralModule(SetpointNet(), optimizer_class=torch.optim.Adam, lr=0.01)
thermostat.neural["setpoint_policy"] = setpoint_nn
```

Define an embedding for the home:

```python
def home_embedding(coh):
    temp = coh.children[0].attributes["temperature"]
    hour = coh.attributes.get("hour", 0)
    return np.array([temp, hour], dtype=np.float32)

home.embedding = home_embedding
```

### Add Daemons (D)

Add a daemon to log energy usage every 5 steps.

```python
class EnergyLogger(Daemon):
    def run(self, coh, dt):
        # coh is the home object
        heater = coh.children[0].attributes["heater_on"]
        lights_on = sum(1 for v in coh.children[1].attributes.values() if isinstance(v, bool) and v)
        print(f"Time {coh.time}: heater={heater}, lights_on={lights_on}, energy={heater + lights_on}")

home.daemons.append(EnergyLogger(interval=5.0))
```

---

## Simulating the System

### The Simulator Class

The `Simulator` takes a root COH object and runs it.

```python
from gismol import Simulator

sim = Simulator(home, dt=1.0, max_steps=100)
sim.run(policy=my_policy)
```

- `dt` – time increment per step.
- `max_steps` – optional step limit.
- `real_time` – if `True`, sleep between steps.

### Policies and Action Selection

A policy is a callable that receives the root COH object and returns the name of a method to execute.

```python
def random_policy(coh):
    import random
    return random.choice(list(coh.methods.keys()))
```

For our home, we have two objects with methods. We can define a policy that selects which component to activate:

```python
def home_policy(coh):
    # Decide: either update thermostat or toggle a light
    if coh.attributes.get("hour", 0) % 2 == 0:
        return "thermostat.update"
    else:
        # Choose a random room
        room = random.choice(["bedroom", "livingroom"])
        # We need to call the light's method; we can use apply_method on the child
        # But the simulator only calls methods on the root.
        # So we should define a root method that delegates.
        pass
```

Better: define a root method that calls child methods. For example:

```python
def root_step(state):
    # Update thermostat
    thermostat.apply_method("update")
    # Maybe toggle lights based on some condition
    if random.random() < 0.1:
        lights.apply_method("toggle", room=random.choice(["bedroom", "livingroom"]))
    return state, home.compute_goal()

home.methods["step"] = root_step
```

Then the policy simply returns `"step"`.

### Event Handling

The simulator has an `event_bus`. You can publish and subscribe to events.

```python
def my_event_handler(event):
    print(f"Received event {event.name} with data {event.data}")

sim.event_bus.subscribe("my_event", my_event_handler)
sim.publish("my_event", data=42)
```

Triggers are automatically subscribed to relevant events.

### Accessing Simulation State

During simulation, you can access the current time and step count via `sim.time` and `sim.step_count`. Daemons receive the root COH object as an argument, but note that `coh.parent` is the simulator (for the root) – you can access `coh.parent.time` to get simulation time.

---

## Learning and Adaptation

### Using NeuralModule

`NeuralModule` wraps a PyTorch module. Use it like a regular model:

```python
output = neural_module.forward(input_tensor)
loss = criterion(output, target)
neural_module.step(loss)
```

### Training with ConstrainedRL

GISMOL provides a simple reinforcement learner `ConstrainedRL` for systems with a neural policy.

```python
from gismol.learning import ConstrainedRL

rl = ConstrainedRL(home, home.neural["setpoint_policy"], gamma=0.99, constraint_cost=10.0)

for episode in range(100):
    loss = rl.train_episode(max_steps=50)
    print(f"Episode {episode}, loss={loss:.4f}")
```

The learner assumes:
- The root COH has an embedding function.
- The policy network outputs logits for actions (number of actions = number of root methods).
- Methods return rewards.

You can extend `ConstrainedRL` or write your own learning algorithm.

### Custom Learning Algorithms

You can implement your own learning by:
- Running episodes manually (calling `sim.step(action)`).
- Collecting experience.
- Updating neural components via `neural_module.step(loss)`.

---

## Advanced Composition: Categorical Operations

### Product (Parallel Composition)

Combine two or more COH objects into a single system where they run in parallel.

```python
from gismol.category import product

combined = product(home, another_system)
```

The product:
- Merges children, attributes, methods (with prefixes), constraints, daemons, and neural components.
- Identity constraints are combined with logical AND.
- Goals are summed.

### Coproduct and Exponential (Experimental)

These are placeholders for future releases. Coproduct will represent alternative composition; exponential will represent function spaces.

---

## Serialization and Persistence

### Saving to JSON

```python
from gismol.utils import to_json

to_json(home, "home.json")
```

This saves the hierarchy and attributes, but **not** methods, neural components, or constraints (they are not serializable by default). You need to re‑attach them after loading.

### Loading from JSON

```python
from gismol.utils import from_json

loaded_home = from_json("home.json")
# Now re‑attach methods, constraints, etc.
```

For full persistence, consider using `pickle` (but be aware of security implications).

---

## Visualization Tools

### Drawing the Hierarchy

If `matplotlib` is installed, you can visualize the component DAG.

```python
from gismol.visualization import draw_hierarchy

draw_hierarchy(home)   # displays interactively
draw_hierarchy(home, filename="home_graph.png")   # saves to file
```

---

## Best Practices and Design Patterns

1. **Keep attributes simple** – Use numbers, strings, or small lists/tuples for easy serialization.
2. **Define methods as pure functions** – They should take a state dict and return a new state dict, not mutate external objects.
3. **Use trigger constraints for reactive behavior** – They keep your code clean and decoupled.
4. **Leverage daemons for monitoring and logging** – Don’t clutter methods with logging code.
5. **Design hierarchies that mirror the real system** – This makes the model intuitive.
6. **Test identity constraints thoroughly** – They are your safety net.
7. **When learning, ensure the embedding provides enough information** – The policy network can only see what you give it.
8. **Use `apply_method` to call child methods** – This ensures identity constraints are checked.
9. **For large systems, use factories to create components** – Keep your code DRY.

---

## Troubleshooting Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `ConstraintViolation` during simulation | An identity constraint failed | Check your methods – are they maintaining invariants? Use daemons to log state before violation. |
| Simulation runs forever | No `max_steps` and no termination condition | Set `max_steps` or include a goal‑based termination in your policy. |
| Neural network not learning | Embedding may be poor, or learning rate wrong | Visualize embeddings, tune hyperparameters. |
| Events not triggering | Event name mismatch or condition always false | Print event names and condition results. |
| `is_dag` returns `False` | Cycle in hierarchy | Use `add_child` to maintain parent references correctly. |
| Product operation produces name collisions | Attributes or methods have same name in different components | The product prefixes names automatically; check that you’re using the prefixed names. |

---

## API Quick Reference

| Module | Class / Function | Description |
|--------|------------------|-------------|
| `gismol.core` | `COH` | Main class. |
| `gismol.core` | `NeuralModule` | PyTorch wrapper. |
| `gismol.core` | `Trigger` | ECA rule. |
| `gismol.core` | `Daemon` | Base for monitors. |
| `gismol.core` | `ConstraintViolation` | Exception. |
| `gismol.constraints` | `IdentityConstraint` | Wraps invariant. |
| `gismol.constraints` | `GoalConstraint` | Wraps objective. |
| `gismol.simulation` | `Simulator` | Simulation engine. |
| `gismol.simulation` | `Event`, `EventBus` | Event system. |
| `gismol.learning` | `ConstrainedRL` | RL learner. |
| `gismol.category` | `product` | Parallel composition. |
| `gismol.utils` | `to_json`, `from_json` | Serialization. |
| `gismol.utils` | `default_embedding` | Basic embedding. |
| `gismol.utils` | `is_dag` | Cycle check. |
| `gismol.visualization` | `draw_hierarchy` | Graph drawing. |

---

## Next Steps and Further Resources

- Explore the `examples/` directory in the repository for more complex systems (gridworld, quantum control, biological cell, etc.).
- Read the [COH paper](https://example.com) for theoretical background.
- Join the GISMOL community on GitHub to ask questions, report issues, or contribute.
- Consider extending GISMOL with new constraint types, learning algorithms, or visualizations.

We hope GISMOL empowers you to model, simulate, and understand intelligent systems in a principled way. Happy modelling!

---

*This user guide is a living document. For the latest version, visit [https://gismol.readthedocs.io](https://gismol.readthedocs.io).*