# examples/gridworld.py
from gismol import COH, NeuralModule, Daemon, Simulator
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg" if Qt installed
import matplotlib.pyplot as plt

# ============ WORLD DEFINITION ============

grid = COH(name='Grid', attributes={'width': 5, 'height': 5})
agent = COH(name='Agent', attributes={'x': 0, 'y': 0, 'goal': (4, 4)})
obstacles = COH(name='Obstacles', attributes={'cells': [(2, 2), (3, 3)]})

world = COH(name='World', children=[grid, agent, obstacles])

# ============ MOVEMENT METHODS ON WORLD ============

def move(dx, dy):
    def _move(world_state):
        x, y = agent.attributes['x'], agent.attributes['y']
        new_x, new_y = x + dx, y + dy

        if (
            (new_x, new_y) not in obstacles.attributes['cells']
            and 0 <= new_x < grid.attributes['width']
            and 0 <= new_y < grid.attributes['height']
        ):
            agent.attributes['x'], agent.attributes['y'] = new_x, new_y

        return world_state, -1.0
    return _move

world.methods['up'] = move(0, 1)
world.methods['down'] = move(0, -1)
world.methods['left'] = move(-1, 0)
world.methods['right'] = move(1, 0)
world.methods['wait'] = lambda s: (s, -1.0)

# ============ SIMPLE Q-NET MODULE (OPTIONAL) ============

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)
    def forward(self, x):
        return self.fc(x)

world.neural['q_net'] = NeuralModule(QNet())

# ============ CONSTRAINTS & GOALS ON AGENT ============

def inside_bounds(_coh: COH) -> bool:
    x, y = agent.attributes['x'], agent.attributes['y']
    return 0 <= x < grid.attributes['width'] and 0 <= y < grid.attributes['height']

agent.identity_constraints.append(inside_bounds)

def goal_reward(_coh: COH) -> float:
    x, y = agent.attributes['x'], agent.attributes['y']
    gx, gy = agent.attributes['goal']
    return -float(abs(x - gx) + abs(y - gy))

agent.goal_constraints.append(goal_reward)

# ============ VISUALIZATION DAEMON ============

class GridVisualizer(Daemon):
    def __init__(self, interval=1.0):
        super().__init__(interval)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def run(self, coh: COH, dt: float):
        self.ax.clear()

        width = grid.attributes['width']
        height = grid.attributes['height']

        # Draw grid
        self.ax.set_xlim(-0.5, width - 0.5)
        self.ax.set_ylim(-0.5, height - 0.5)
        self.ax.set_xticks(np.arange(-0.5, width, 1))
        self.ax.set_yticks(np.arange(-0.5, height, 1))
        self.ax.grid(True)

        # Draw obstacles
        for (ox, oy) in obstacles.attributes['cells']:
            self.ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1,
                                            color='black', alpha=0.7))

        # Draw goal
        gx, gy = agent.attributes['goal']
        self.ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1,
                                        color='gold', alpha=0.8))

        # Draw agent
        x, y = agent.attributes['x'], agent.attributes['y']
        self.ax.plot(x, y, 'o', color='blue', markersize=18)

        self.ax.set_title(f"Agent Position: ({x},{y})")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Attach daemon
agent.daemons.append(GridVisualizer(interval=1.0))

# ============ RUN SIMULATION ============

if __name__ == '__main__':
    sim = Simulator(world, dt=1.0, max_steps=20)
    sim.run(policy=lambda coh: np.random.choice(['up', 'down', 'left', 'right', 'wait']))
    print("Simulation finished. Close the plot window to exit.")
    plt.ioff()
    plt.show()