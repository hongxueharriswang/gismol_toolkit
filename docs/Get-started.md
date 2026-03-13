

## 🧪 Usage Tutorial

This tutorial gets you from **zero to running simulations** (interactive and training) and shows how to build your **own COH environment** with constraints, daemons, and (optionally) a PyTorch policy.

> **Prerequisites**
>
> *   Python 3.10+
> *   A virtual environment (`python -m venv .venv && source .venv/bin/activate`)
> *   Dependencies: `pip install -r requirements.txt`
> *   (Optional for MP4) `ffmpeg` installed on your system

***

### 1) Verify your install

```bash
# From repo root
python -c "import gismol; print('GISMOL OK:', gismol.__version__)"
```

If you see `GISMOL OK: <version>`, you’re ready.

***

### 2) Run the Advanced Gridworld (Interactive)

```bash
python examples/gridworld_advanced.py --mode interactive --max-steps 300
```

**Controls**

*   **W / A / S / D** or **Arrow keys** to move
*   **Space** to wait
*   **Hold** keys for auto‑repeat (configurable delay/rate)
*   The window must have focus; click once if keys don’t register

**What you’ll see**

*   **Top‑left**: Grid with **heatmap** (goal distance), **agent**, **trail**, and a **status overlay** (last key, current action)
*   **Top‑right**: **Policy arrow field** showing argmax actions over all cells
*   **Bottom**: Live **metrics panel** (loss per step & return per episode)

To export media at the end, the script **auto‑saves** `gridworld.gif` and (if `ffmpeg` exists) `gridworld.mp4`.

***

### 3) Run the Advanced Gridworld (Training)

```bash
python examples/gridworld_advanced.py --mode train --episodes 30 --max-steps 200
```

*   Trains a **Q‑network** (small MLP) end‑to‑end with **potential‑based shaping**
*   Updates the **policy arrow field** as the network improves
*   Plots **loss (per step)** and **return (per episode)** in real time
*   At completion: runs a **greedy rollout** using the learned policy and exports **GIF/MP4**

***

### 4) Build a Minimal COH from Scratch

Below is a **tiny** example that shows the essential COH parts: objects, a movement method, an identity constraint, a logging daemon, and a one‑step simulation loop.

```python
# examples/minimal_coh_demo.py
from gismol import COH, Daemon, Simulator

# 1) Define objects
grid = COH(name='Grid', attributes={'width': 5, 'height': 5})
agent = COH(name='Agent', attributes={'x': 0, 'y': 0})
world = COH(name='World', children=[grid, agent])

# 2) Define a method on World that mutates Agent
def move(dx, dy):
    def _m(s):
        x, y = agent.attributes['x'], agent.attributes['y']
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.attributes['width'] and 0 <= ny < grid.attributes['height']:
            agent.attributes['x'], agent.attributes['y'] = nx, ny
        return s, 0.0
    return _m

world.methods['right'] = move(1, 0)

# 3) Add an identity constraint (always hold)
def inside_bounds(_coh):
    x, y = agent.attributes['x'], agent.attributes['y']
    w, h = grid.attributes['width'], grid.attributes['height']
    return 0 <= x < w and 0 <= y < h

agent.identity_constraints.append(inside_bounds)

# 4) A simple daemon (logger)
class Logger(Daemon):
    def run(self, coh, dt):
        print(f"pos=({agent.attributes['x']},{agent.attributes['y']})")

agent.daemons.append(Logger(interval=0.5))

# 5) Simulate a few steps
if __name__ == "__main__":
    sim = Simulator(world, dt=0.5, max_steps=5)
    # Use a fixed policy that always moves right
    sim.run(policy=lambda coh: 'right')
```

**Run it:**

```bash
python examples/minimal_coh_demo.py
```

You should see printed positions stepping to the right until reaching the boundary.

***

### 5) Add a PyTorch Policy (NeuralModule) to Your COH

Below extends the minimal demo by attaching a **NeuralModule** and querying it to choose actions.

```python
# examples/policy_demo.py
from gismol import COH, Daemon, Simulator, NeuralModule
import torch
import torch.nn as nn
import numpy as np

grid = COH(name='Grid', attributes={'width': 5, 'height': 5})
agent = COH(name='Agent', attributes={'x': 0, 'y': 0})
world = COH(name='World', children=[grid, agent])

def move(dx, dy):
    def _m(s):
        x, y = agent.attributes['x'], agent.attributes['y']
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.attributes['width'] and 0 <= ny < grid.attributes['height']:
            agent.attributes['x'], agent.attributes['y'] = nx, ny
        return s, 0.0
    return _m

ACTIONS = ['up', 'down', 'left', 'right', 'wait']
world.methods['up'] = move(0, 1)
world.methods['down'] = move(0, -1)
world.methods['left'] = move(-1, 0)
world.methods['right'] = move(1, 0)
world.methods['wait'] = lambda s: (s, 0.0)

def inside_bounds(_coh):
    x, y = agent.attributes['x'], agent.attributes['y']
    w, h = grid.attributes['width'], grid.attributes['height']
    return 0 <= x < w and 0 <= y < h

agent.identity_constraints.append(inside_bounds)

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, len(ACTIONS))  # input: (x,y)
    def forward(self, x):
        return self.fc(x)

world.neural['policy'] = NeuralModule(TinyNet())

def policy_fn(coh):
    x, y = agent.attributes['x'], agent.attributes['y']
    inp = torch.tensor([x, y], dtype=torch.float32)
    with torch.no_grad():
        logits = world.neural['policy'].forward(inp)
    a = int(torch.argmax(logits))
    return ACTIONS[a]

if __name__ == "__main__":
    sim = Simulator(world, dt=0.2, max_steps=10)
    sim.run(policy=policy_fn)
```

> This example **doesn’t train** the network; it just shows how to **attach and query** a model. See the **Gridworld Advanced** example for fully integrated training.

***

### 6) Create Your Own Environment

1.  **Define COH objects** for the world, agents, resources, etc.
2.  **Attach methods** (actions) where each method:
    *   *Reads* attributes it needs
    *   *Mutates* the appropriate object’s state
    *   Returns `(state, reward)`
3.  **Add constraints** (identity/goal) on the object that *owns* the state.
4.  **Introduce daemons** for logging/visualization/schedulers.
5.  **Simulate** with `Simulator(world, dt, max_steps).run(policy=…)`.

**Directory suggestion**

    my_env/
      __init__.py
      world.py         # defines world/agent objects, methods, constraints
      viz.py           # daemons for visuals/logging
      train.py         # your RL loop (or re‑use the example)
      policies.py      # network definitions

***

### 7) Customize Keyboard Repeat (Interactive UX)

In the advanced demo, **hold‑to‑repeat** is controlled by:

```python
viz.repeat_delay = 0.25   # seconds before repetition starts
viz.repeat_rate  = 0.10   # seconds between repeats
```

Find these fields in `AdvancedGridVisualizer` if you want to tweak responsiveness.

***

### 8) Export Animations (Fixed Canvas)

The advanced demo uses a **fixed 1120×800** canvas to ensure all frames are identical for both GIF and MP4 export. If you adapt the visualizer, keep the **fixed size + padding/cropping** logic to avoid shape mismatch errors.

*   GIF: always available (via `imageio`)
*   MP4: requires `ffmpeg` on PATH

***

### 9) Troubleshooting

*   **No window shows / “non‑interactive backend”**  
    Add at the top of your script (before `import pyplot`):
    ```python
    import matplotlib
    matplotlib.use("TkAgg")
    ```
    And ensure Tk is installed (e.g., `sudo apt install python3-tk` on Ubuntu).

*   **Keys not registering**  
    Click the window once to focus. The demo also forces focus on startup; if your WM blocks that, a single click will do.

*   **GIF/MP4 export fails (“different shapes”)**  
    Use the advanced demo’s **fixed‑canvas** approach (locked figure size and padding/cropping before saving).

*   **Large repo pushes to GitHub fail**  
    Use **Git LFS** for large binaries, add `__pycache__/`, `.venv/`, `.vscode/`, datasets, and media to `.gitignore`.  
    If history already contains those folders, consider `git filter-repo` or **BFG** cleanup (then force‑push).

***

### 10) Best Practices

*   Keep environment logic **agent‑centric** (the agent owns its `(x,y)` state, constraints, and goals).
*   Keep methods close to the object whose state they **mutate** (or mutate explicitly if method is stored elsewhere).
*   Use **daemons** for visualization, logging, and background tasks—this keeps the simulation loop clean.
*   For RL, start with **simple shaping** (e.g., negative Manhattan distance) then iterate.
*   Maintain **reproducibility**: seed your RNGs and log train/test metrics.

***

### 11) Next Steps

*   Add a **Value‑heatmap** panel (visualize max‑Q as value)
*   Add **policy arrows per episode** snapshots to see learning dynamics
*   Swap Q‑learning for **DQN/PPO** with replay or policy gradients
*   Build a **multi‑agent** environment by adding more COH agents to the same world

***

### Further Readings:

*   A **cookie‑cutter “new environment” template**,
*   A **notebook tutorial** mirroring this section, or
*   A **minimal PPO trainer** that plugs into COH with the same visualizer.
