# examples/gridworld_advanced.py
# Enhanced Gridworld demo for gismol with:
# - Heatmap overlay (goal distance)
# - Trail visualization
# - Interactive keyboard control (W/A/S/D/Arrow Keys/Space)
# - GIF/MP4 export
# - Simple Q-learning training loop with live visualization
# - Backend-safe frame capture (TkAgg/Qt5Agg/GTK3Agg/Agg)
# - Status overlay for last keypress & current action
# - Option C layout: Gridworld (left), Policy Arrows (right), Reward/Loss plots (bottom)
# - FIXED-CANVAS EXPORT: Locks pixel size (1120x800) so all frames have identical shape

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
    # Disable automatic layout engines that can change canvas size
    mpl.rcParams['figure.constrained_layout.use'] = False
    mpl.rcParams['figure.autolayout'] = False
except Exception:
    pass

import matplotlib.pyplot as plt
import imageio
from collections import deque
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
# 5. ADVANCED VISUALIZER DAEMON (Option C layout + Fixed Canvas)
# ============================================================

class AdvancedGridVisualizer(Daemon):
    """
    - Left: Gridworld (heatmap + trail + agent + status)
    - Right: Policy Arrows overlay (best action per cell from Q-net)
    - Bottom: Live reward/loss plot
    - Keyboard control (W/A/S/D/Arrows/Space)
    - Backend-safe frame capture (GIF/MP4 export)
    - FIXED canvas size for consistent frames
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
        # Fixed canvas target (divisible by 16 for ffmpeg)
        self.dpi = 100
        self.width_px = 1120
        self.height_px = 800
        self.width_in = self.width_px / self.dpi
        self.height_in = self.height_px / self.dpi
        # Input queue to ensure no missed presses (one-shot)
        self._pending_actions = deque()
        # Logging for plots
        self.global_step = 0
        self.loss_hist = []      # per learning step
        self.rew_hist = []       # per episode total
        self.episode_idx = []
        self._init_fig()

    # ---- Public hooks for trainer ----
    def start_episode(self):
        self._ep_return = 0.0

    def record_step(self, reward: float, loss: float | None):
        self._ep_return += float(reward)
        if loss is not None:
            self.loss_hist.append(float(loss))
        self.global_step += 1

    def end_episode(self, ep_number: int):
        self.rew_hist.append(self._ep_return)
        self.episode_idx.append(int(ep_number))
        self._ep_return = 0.0

    # ---- Figure & input handling ----
    def _init_fig(self):
        try:
            plt.ion()
        except Exception:
            pass
        # Create figure with fixed size & dpi
        self.fig = plt.figure(figsize=(self.width_in, self.height_in), dpi=self.dpi)
        # Lock layout engine (matplotlib>=3.8)
        try:
            self.fig.set_layout_engine(None)
        except Exception:
            pass
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], wspace=0.28, hspace=0.28)
        self.ax_grid   = self.fig.add_subplot(gs[0, 0])
        self.ax_arrows = self.fig.add_subplot(gs[0, 1])
        self.ax_plot   = self.fig.add_subplot(gs[1, :])

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # Persistent hint
        self.hint_text = self.fig.text(
            0.5, 0.01,
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
            self._pending_actions.append(self.KEYMAP[event.key])

    def get_action(self):
        return self.current_action

    def consume_action(self):
        # Pop next queued action (if any); otherwise 'wait'
        if self._pending_actions:
            self.current_action = self._pending_actions.popleft()
        else:
            self.current_action = 'wait'
        return self.current_action

    def reset(self):
        self.trail.clear()
        self.current_action = 'wait'
        self.frames.clear()
        self.last_key = None
        self._pending_actions.clear()
        # Do not clear histories so plots persist across episodes unless user restarts process

    # ---- Drawing helpers ----
    def _compute_heatmap(self):
        gx, gy = agent.attributes['goal']
        w, h = grid.attributes['width'], grid.attributes['height']
        H = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                H[h - 1 - y, x] = abs(x - gx) + abs(y - gy)  # flip y for display
        return H

    def _draw_gridworld(self):
        ax = self.ax_grid
        ax.clear()
        w, h = grid.attributes['width'], grid.attributes['height']
        # Heatmap
        H = self._compute_heatmap()
        ax.imshow(H, cmap='Reds', interpolation='nearest',
                  extent=[-0.5, w - 0.5, -0.5, h - 0.5], alpha=0.35)
        # Obstacles
        for (ox, oy) in obstacles.attributes['cells']:
            ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black'))
        # Goal
        gx, gy = agent.attributes['goal']
        ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color='gold', alpha=0.85))
        # Trail & Agent
        x, y = agent.attributes['x'], agent.attributes['y']
        self.trail.append((x, y))
        if len(self.trail) > 1:
            xs = [p[0] for p in self.trail]
            ys = [p[1] for p in self.trail]
            ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.7)
        ax.plot(x, y, 'o', color='blue', markersize=16)
        # Cosmetics
        ax.set_xticks(np.arange(-0.5, w, 1)); ax.set_yticks(np.arange(-0.5, h, 1))
        ax.grid(True, alpha=0.4)
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(-0.5, h - 0.5); ax.set_aspect('equal')
        ax.set_title(f"Gridworld — Agent=({x},{y}) | Dist={manhattan_distance():.0f}")
        # Status overlay
        status = f"Last key: {self.last_key or '—'} | Current action: {self.current_action} | Frames: {len(self.frames)}"
        ax.text(0.01, 0.99, status, transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7), color='black')

    def _compute_policy_field(self):
        """Return (X,Y,U,V,mask_wait) for quiver from current policy over grid cells."""
        w, h = grid.attributes['width'], grid.attributes['height']
        gx, gy = agent.attributes['goal']
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        U = np.zeros((h, w), dtype=np.float32)
        V = np.zeros((h, w), dtype=np.float32)
        mask_wait = np.zeros((h, w), dtype=bool)
        # Map actions to vectors
        vec = {
            'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0), 'wait': (0, 0)
        }
        with torch.no_grad():
            for y in range(h):
                for x in range(w):
                    if (x, y) in obstacles.attributes['cells']:
                        mask_wait[h - 1 - y, x] = True
                        continue
                    s = np.array([x, y, gx, gy], dtype=np.float32)
                    q = policy.forward(torch.tensor(s)).detach().cpu().numpy()
                    a_idx = int(np.argmax(q))
                    a_name = self.ACTIONS[a_idx]
                    dx, dy = vec[a_name]
                    # Flip Y for display row
                    U[h - 1 - y, x] = dx
                    V[h - 1 - y, x] = dy
                    if dx == 0 and dy == 0:
                        mask_wait[h - 1 - y, x] = True
        # Coordinates for quiver should align with displayed orientation
        Xd, Yd = X, (h - 1) - Y
        return Xd, Yd, U, V, mask_wait

    def _draw_policy_arrows(self):
        ax = self.ax_arrows
        ax.clear()
        w, h = grid.attributes['width'], grid.attributes['height']
        # Background heatmap (lighter)
        H = self._compute_heatmap()
        ax.imshow(H, cmap='Reds', interpolation='nearest',
                  extent=[-0.5, w - 0.5, -0.5, h - 0.5], alpha=0.25)
        # Obstacles and goal
        for (ox, oy) in obstacles.attributes['cells']:
            ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black', alpha=0.8))
        gx, gy = agent.attributes['goal']
        ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color='gold', alpha=0.9))
        # Quiver field
        X, Y, U, V, mask_wait = self._compute_policy_field()
        ax.quiver(X, Y, U, V, color='navy', angles='xy', scale_units='xy', scale=1.8, alpha=0.8)
        # Mark wait cells as faint dots
        yy, xx = np.where(mask_wait)
        ax.scatter(xx, yy, s=10, c='navy', alpha=0.5)
        # Cosmetics
        ax.set_xticks(np.arange(-0.5, w, 1)); ax.set_yticks(np.arange(-0.5, h, 1))
        ax.grid(True, alpha=0.4)
        ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(-0.5, h - 0.5); ax.set_aspect('equal')
        ax.set_title("Policy Arrows — argmax_a Q(x,y,·)")

    def _draw_reward_loss(self):
        ax = self.ax_plot
        ax.clear()
        # Loss per step
        if self.loss_hist:
            ax.plot(np.arange(1, len(self.loss_hist) + 1), self.loss_hist, color='tab:blue', label='Loss (per step)')
            ax.set_xlabel('Training step')
            ax.set_ylabel('Loss', color='tab:blue')
        else:
            ax.set_xlabel('Training step'); ax.set_ylabel('Loss')
        # Episode returns on twin axis
        ax2 = ax.twinx()
        if self.rew_hist:
            ax2.plot(self.episode_idx, self.rew_hist, color='tab:orange', marker='o', label='Return (per episode)')
            ax2.set_ylabel('Episode return', color='tab:orange')
        else:
            ax2.set_ylabel('Episode return')
        ax.set_title('Training Metrics — Loss (step) & Return (episode)')
        # Combine legends
        lines = []
        labels = []
        for axes in (ax, ax2):
            L = axes.get_legend_handles_labels()
            lines += L[0]; labels += L[1]
        if lines:
            ax.legend(lines, labels, loc='upper right')

    def _ensure_fixed_canvas(self):
        # Force figure to target inches & dpi each frame
        try:
            self.fig.set_size_inches(self.width_in, self.height_in, forward=True)
            self.fig.set_dpi(self.dpi)
        except Exception:
            pass

    def _fix_frame_size(self, frame: np.ndarray) -> np.ndarray:
        # Pad/crop to exact (height_px, width_px, 3)
        h, w = frame.shape[:2]
        target_h, target_w = self.height_px, self.width_px
        # Crop if larger
        frame = frame[:target_h, :target_w, :]
        # Pad if smaller
        if frame.shape[0] < target_h or frame.shape[1] < target_w:
            out = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            out[:frame.shape[0], :frame.shape[1], :] = frame
            frame = out
        return frame

    def run(self, coh, dt):
        # Draw panels
        self._draw_gridworld()
        self._draw_policy_arrows()
        self._draw_reward_loss()

        # Enforce fixed canvas geometry
        self._ensure_fixed_canvas()

        # SAFE cross-backend frame capture with fixed size guarantee
        try:
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()  # Agg renderer behind Tk/Qt/GTK canvas
            buf = renderer.buffer_rgba()               # raw RGBA bytes
            frame = np.asarray(buf, dtype=np.uint8)[:, :, :3]  # drop alpha
            frame = self._fix_frame_size(frame)
            if self.capture_frames and len(self.frames) < self.max_frames:
                self.frames.append(frame)
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
# 6. Q-LEARNING UTILS (with live plotting callbacks)
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

def qlearning_train_episode(ep_number: int, max_steps=100, gamma=0.99):
    optimizer = policy.optimizer
    viz.start_episode()
    for t in range(max_steps):
        s = extract_state()
        a = select_action_epsilon_greedy(s, epsilon=max(0.05, 0.2 - 0.002 * (ep_number * max_steps + t)))
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

        # Log to visualizer (reward & loss)
        viz.record_step(reward=r, loss=float(loss.item()))

        # Tick daemons to refresh panels
        try:
            Simulator(world, dt=0.2, max_steps=1).step('wait')
        except Exception:
            pass

        if manhattan_distance() == 0:
            break

    viz.end_episode(ep_number)


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
    parser = argparse.ArgumentParser(description='Advanced Gridworld demo (COH) — Option C layout')
    parser.add_argument('--mode', choices=['interactive', 'train'], default='interactive')
    parser.add_argument('--episodes', type=int, default=30, help='episodes for training mode')
    parser.add_argument('--max-steps', type=int, default=200, help='max steps in interactive run')
    parser.add_argument('--gif', default='gridworld.gif')
    parser.add_argument('--mp4', default='gridworld.mp4')
    parser.add_argument('--fps', type=int, default=6)
    args = parser.parse_args()

    if args.mode == 'interactive':
        sim = Simulator(world, dt=0.2, max_steps=args.max_steps)
        sim.run(policy=lambda coh: viz.consume_action())
    else:
        for ep in range(1, args.episodes + 1):
            qlearning_train_episode(ep_number=ep)
            print(f"Episode {ep}/{args.episodes} — return={viz.rew_hist[-1]:.2f}, steps={len(viz.loss_hist)}")
        # Visualize trained policy greedily
        agent.attributes['x'], agent.attributes['y'] = 0, 0
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
