# examples/gridworld_advanced.py
# Enhanced Gridworld demo for gismol with:
# - Heatmap overlay (goal distance)
# - Trail visualization
# - Interactive keyboard control (W/A/S/D/Arrow Keys/Space)
# - GIF/MP4 export
# - Simple Q-learning training loop with live visualization
# - Backend-safe frame capture (TkAgg/Qt5Agg/GTK3Agg/Agg)
# - Status overlay for last keypress & current action

import argparse
import numpy as np
import matplotlib

# Prefer an interactive backend; fallback to Agg if unavailable
if not matplotlib.get_backend().lower().startswith(('tkagg', 'qt5agg', 'gtk3agg')):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')

import matplotlib as mpl
# Disable Matplotlib default key bindings that conflict with WASD/space/arrow keys
try:
    mpl.rcParams['toolbar'] = 'none'
    for key in list(mpl.rcParams.keys()):
        if key.startswith('keymap.'):
            mpl.rcParams[key] = []
except Exception:
    pass

import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.optim as optim

from gismol import COH, NeuralModule, Daemon, Simulator

# ============================================================
# 1. WORLD SETUP
# ============================================================

grid = COH(name='Grid', attributes={'width': 7, 'height': 7})
agent = COH(name='Agent', attributes={'x': 0, 'y': 0, 'goal': (6, 6)})
obstacles = COH(name='Obstacles', attributes={'cells': [(2, 2), (3, 3), (4, 4), (1, 5), (5, 1)]})

world = COH(name='World', children=[grid, agent, obstacles])

# ============================================================
# 2. MOVEMENT METHODS (on world; mutate agent)
# ============================================================

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
        # step cost
        return world_state, -1.0
    return _move

world.methods['up'] = move(0, 1)
world.methods['down'] = move(0, -1)
world.methods['left'] = move(-1, 0)
world.methods['right'] = move(1, 0)
world.methods['wait'] = lambda s: (s, -0.5)

# ============================================================
# 3. POLICY NETWORK (for Q-learning)
# ============================================================

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)

policy = NeuralModule(QNet(), optimizer_class=optim.Adam, lr=5e-3)
world.neural['policy'] = policy

# ============================================================
# 4. CONSTRAINTS & GOALS
# ============================================================

def inside_bounds(_coh):
    x, y = agent.attributes['x'], agent.attributes['y']
    return 0 <= x < grid.attributes['width'] and 0 <= y < grid.attributes['height']

agent.identity_constraints.append(inside_bounds)

# Potential function for shaping
def manhattan_distance():
    x, y = agent.attributes['x'], agent.attributes['y']
    gx, gy = agent.attributes['goal']
    return abs(x - gx) + abs(y - gy)

# ============================================================
# 5. ADVANCED VISUALIZER DAEMON
# ============================================================

class AdvancedGridVisualizer(Daemon):
    """
    - Heatmap overlay (goal distance)
    - Trail visualization
    - Keyboard control (W/A/S/D/Arrows/Space)
    - Backend-safe frame capture (GIF/MP4 export)
    - Status overlay (last key & action)
    """
    ACTIONS = ['up', 'down', 'left', 'right', 'wait']
    KEYMAP = {
        'w': 'up', 'W': 'up', 'up': 'up',
        's': 'down', 'S': 'down', 'down': 'down',
        'a': 'left', 'A': 'left', 'left': 'left',
        'd': 'right', 'D': 'right', 'right': 'right',
        ' ': 'wait'
    }

    def __init__(self, interval=0.2, capture_frames=True, max_frames=2000):
        super().__init__(interval)
        self.capture_frames = capture_frames
        self.max_frames = max_frames
        self.trail = []
        self.frames = []
        self.current_action = 'wait'
        self.last_key = None
        self._init_fig()

    def _init_fig(self):
        try:
            plt.ion()
        except Exception:
            pass
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # Persistent hints (figure-level, drawn once)
        self.hint_text = self.fig.text(
            0.5, 0.02,
            'Controls: W/A/S/D or ←↑↓→, Space=wait. Window must have focus.',
            ha='center', va='bottom', fontsize=9, alpha=0.85
        )
        # Try to force focus to the window for reliable key events
        try:
            self.fig.canvas.manager.window.focus_force()
        except Exception:
            pass

    def on_key(self, event):
        self.last_key = event.key
        if event.key in self.KEYMAP:
            self.current_action = self.KEYMAP[event.key]

    def get_action(self):
        return self.current_action

    def consume_action(self):
        # Return current action once, then reset to 'wait' so key press is one-shot
        a = self.current_action
        self.current_action = 'wait'
        return a

    def reset(self):
        self.trail.clear()
        self.current_action = 'wait'
        self.frames.clear()
        self.last_key = None

    def _compute_heatmap(self):
        gx, gy = agent.attributes['goal']
        w, h = grid.attributes['width'], grid.attributes['height']
        H = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                H[h - 1 - y, x] = abs(x - gx) + abs(y - gy)  # flip y for display
        return H

    def run(self, coh, dt):
        self.ax.clear()
        w, h = grid.attributes['width'], grid.attributes['height']

        # Heatmap
        H = self._compute_heatmap()
        self.ax.imshow(
            H, cmap='Reds', interpolation='nearest',
            extent=[-0.5, w - 0.5, -0.5, h - 0.5], alpha=0.35
        )

        # Obstacles
        for (ox, oy) in obstacles.attributes['cells']:
            self.ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black'))

        # Goal
        gx, gy = agent.attributes['goal']
        self.ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color='gold', alpha=0.85))

        # Trail and Agent
        x, y = agent.attributes['x'], agent.attributes['y']
        self.trail.append((x, y))
        if len(self.trail) > 1:
            xs = [p[0] for p in self.trail]
            ys = [p[1] for p in self.trail]
            self.ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.7)
        self.ax.plot(x, y, 'o', color='blue', markersize=16)

        # Grid lines & cosmetics
        self.ax.set_xticks(np.arange(-0.5, w, 1))
        self.ax.set_yticks(np.arange(-0.5, h, 1))
        self.ax.grid(True, alpha=0.4)
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(-0.5, h - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Agent=({x},{y}) | Dist={manhattan_distance():.0f}")

        # Status overlay (axes-relative at top-left)
        status = f"Last key: {self.last_key or '—'} | Current action: {self.current_action} | Frames: {len(self.frames)}"
        self.ax.text(
            0.01, 0.99, status,
            transform=self.ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7), color='black'
        )

        # SAFE cross-backend frame capture
        try:
            self.fig.canvas.draw()                     # ensure Agg renderer is updated
            renderer = self.fig.canvas.get_renderer()  # Agg renderer behind Tk/Qt/GTK canvas
            buf = renderer.buffer_rgba()               # raw RGBA bytes
            frame = np.asarray(buf, dtype=np.uint8)[:, :, :3]  # drop alpha
            if self.capture_frames and len(self.frames) < self.max_frames:
                self.frames.append(frame.copy())
        except Exception:
            pass

        # Allow GUI to process keypress events each frame
        try:
            self.fig.canvas.start_event_loop(0.001)
        except Exception:
            pass

viz = AdvancedGridVisualizer(interval=0.2, capture_frames=True)
agent.daemons.append(viz)

# ============================================================
# 6. Q-LEARNING UTILS
# ============================================================

def reset_env():
    agent.attributes['x'], agent.attributes['y'] = 0, 0
    viz.reset()

def extract_state():
    x, y = agent.attributes['x'], agent.attributes['y']
    gx, gy = agent.attributes['goal']
    return np.array([x, y, gx, gy], dtype=np.float32)

def select_action_epsilon_greedy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(viz.ACTIONS)
    with torch.no_grad():
        logits = policy.forward(torch.tensor(state))
        a_idx = torch.argmax(logits).item()
        return viz.ACTIONS[a_idx]

def step_env(action_name):
    # Potential-based shaping reward
    d0 = manhattan_distance()
    r = world.apply_method(action_name)
    d1 = manhattan_distance()
    # Encourage moving closer to goal
    shaped = r + (d0 - d1)
    # Bonus on reaching goal
    if d1 == 0:
        shaped += 10.0
    return shaped

def qlearning_train_episode(max_steps=100, gamma=0.99):
    optimizer = policy.optimizer
    total_loss = 0.0

    reset_env()
    for t in range(max_steps):
        s = extract_state()
        a = select_action_epsilon_greedy(s, epsilon=max(0.05, 0.2 - 0.002 * t))
        r = step_env(a)
        s2 = extract_state()

        with torch.no_grad():
            q_next = policy.forward(torch.tensor(s2)).max().item()
            target = r + gamma * q_next

        logits = policy.forward(torch.tensor(s))
        a_idx = viz.ACTIONS.index(a)
        q_sa = logits[a_idx]
        loss = (q_sa - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

        # Update viz by ticking daemons once per step (without altering state)
        try:
            Simulator(world, dt=0.2, max_steps=1).step('wait')
        except Exception:
            pass

        if manhattan_distance() == 0:
            break

    return total_loss / (t + 1)

# ============================================================
# 7. MAIN
# ============================================================

def export_animation(out_gif='gridworld.gif', out_mp4='gridworld.mp4', fps=6):
    # Use imageio for both GIF and MP4 (requires ffmpeg for MP4)
    if viz.frames:
        try:
            imageio.mimsave(out_gif, viz.frames, fps=fps)
            print(f"Saved {out_gif}")
        except Exception as e:
            print("GIF export failed:", e)
        try:
            imageio.mimsave(out_mp4, viz.frames, fps=fps)
            print(f"Saved {out_mp4}")
        except Exception as e:
            print("MP4 export failed (install ffmpeg).", e)
    else:
        print("No frames captured; skipping export.")

def main():
    parser = argparse.ArgumentParser(description='Advanced Gridworld demo (COH)')
    parser.add_argument('--mode', choices=['interactive', 'train'], default='interactive')
    parser.add_argument('--episodes', type=int, default=30, help='episodes for training mode')
    parser.add_argument('--max-steps', type=int, default=200, help='max steps in interactive run')
    parser.add_argument('--gif', default='gridworld.gif')
    parser.add_argument('--mp4', default='gridworld.mp4')
    parser.add_argument('--fps', type=int, default=6)
    args = parser.parse_args()

    if args.mode == 'interactive':
        sim = Simulator(world, dt=0.2, max_steps=args.max_steps)
        # One-shot consumption so a single keypress triggers a single move
        sim.run(policy=lambda coh: viz.consume_action())
    else:
        for ep in range(args.episodes):
            loss = qlearning_train_episode()
            print(f"Episode {ep + 1}/{args.episodes} | loss={loss:.4f}")
        # One last run to visualize trained policy greedily
        reset_env()
        sim = Simulator(world, dt=0.2, max_steps=args.max_steps)
        def greedy(_):
            s = extract_state()
            with torch.no_grad():
                a = torch.argmax(policy.forward(torch.tensor(s))).item()
            return viz.ACTIONS[a]
        sim.run(policy=greedy)

    export_animation(args.gif, args.mp4, fps=args.fps)

    try:
        plt.ioff(); plt.show()
    except Exception:
        pass

if __name__ == '__main__':
    main()