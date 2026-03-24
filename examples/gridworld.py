# examples/gridworld.py
from gismol import COH, NeuralModule, Daemon, Simulator
import numpy as np
import torch.nn as nn

# Components
grid = COH(name='Grid', attributes={'width': 5, 'height': 5})
agent = COH(name='Agent', attributes={'x': 0, 'y': 0, 'goal': (4, 4)})
obstacles = COH(name='Obstacles', attributes={'cells': [(2, 2), (3, 3)]})
world = COH(name='World', children=[grid, agent, obstacles])

# Methods live on WORLD but update AGENT state explicitly
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

# simple Q‑network to align with your design
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)  # (x, y, goal_x, goal_y) -> 5 actions
    def forward(self, x):
        return self.fc(x)

world.neural['q_net'] = NeuralModule(QNet())

# Constraints/goals belong to AGENT (who owns x,y)
def inside_bounds(_coh: COH) -> bool:
    x, y = agent.attributes['x'], agent.attributes['y']
    return 0 <= x < grid.attributes['width'] and 0 <= y < grid.attributes['height']

agent.identity_constraints.append(inside_bounds)

def goal_reward(_coh: COH) -> float:
    x, y = agent.attributes['x'], agent.attributes['y']
    gx, gy = agent.attributes['goal']
    return -float(abs(x - gx) + abs(y - gy))

agent.goal_constraints.append(goal_reward)

# Logger daemon on AGENT now reflects actual movement
class LoggerDaemon(Daemon):
    def run(self, coh: COH, dt: float):
        print(f"pos=({agent.attributes['x']},{agent.attributes['y']})")

agent.daemons.append(LoggerDaemon(interval=1.0))

if __name__ == '__main__':
    sim = Simulator(world, dt=1.0, max_steps=10)
    sim.run(policy=lambda coh: np.random.choice(['up', 'down', 'left', 'right', 'wait']))
