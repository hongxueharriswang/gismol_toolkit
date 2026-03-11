# ============================================================
# gridworld_advanced.py (Part 1/6)
# ============================================================

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
# Disable default key bindings that conflict with WASD/Arrow keys
try:
    mpl.rcParams['toolbar'] = 'none'
    for key in list(mpl.rcParams.keys()):
        if key.startswith('keymap.'):
            mpl.rcParams[key] = []
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

# -----------------------------
# WORLD
# -----------------------------

grid = COH(name='Grid', attributes={'width': 7, 'height': 7})
agent = COH(name='Agent', attributes={'x': 0, 'y': 0, 'goal': (6, 6)})
obstacles = COH(name='Obstacles', attributes={
    'cells': [(2,2),(3,3),(4,4),(1,5),(5,1)]
})
world = COH(name='World', children=[grid, agent, obstacles])

# ============================================================
# gridworld_advanced.py (Part 2/6)
# ============================================================

# -----------------------------
# MOVEMENT METHODS
# -----------------------------

def move(dx, dy):
    def _move(state):
        x, y = agent.attributes['x'], agent.attributes['y']
        nx, ny = x + dx, y + dy
        if ((nx, ny) not in obstacles.attributes['cells']
            and 0 <= nx < grid.attributes['width']
            and 0 <= ny < grid.attributes['height']):
            agent.attributes['x'], agent.attributes['y'] = nx, ny
        return state, -1.0
    return _move

world.methods['up'] = move(0,1)
world.methods['down'] = move(0,-1)
world.methods['left'] = move(-1,0)
world.methods['right'] = move(1,0)
world.methods['wait'] = lambda s: (s, -0.5)

# -----------------------------
# Q-NETWORK
# -----------------------------

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,5)
        )
    def forward(self, x):
        return self.net(x)

policy = NeuralModule(QNet(), optimizer_class=optim.Adam, lr=5e-3)
world.neural['policy'] = policy

# -----------------------------
# COH CONSTRAINTS
# -----------------------------

def inside_bounds(_coh):
    x, y = agent.attributes['x'], agent.attributes['y']
    return 0 <= x < grid.attributes['width'] and 0 <= y < grid.attributes['height']

agent.identity_constraints.append(inside_bounds)

def manhattan_distance():
    x, y = agent.attributes['x'], agent.attributes['y']
    gx, gy = agent.attributes['goal']
    return abs(x-gx) + abs(y-gy)

# ============================================================
# gridworld_advanced.py (Part 3/6)
# ============================================================

class AdvancedGridVisualizer(Daemon):
    """
    Option C Layout:
       [ Gridworld ]   [ Policy Arrows ]
       [      Metrics (Loss/Return)    ]

    Added features:
       • Keyboard queue + repeat delay + repeat rate
       • Fixed-size canvas (1120×800 px @ 100 DPI)
       • Safe consistent frame capture
    """

    ACTIONS = ['up','down','left','right','wait']
    KEYMAP = {
        'w':'up','W':'up','up':'up',
        's':'down','S':'down','down':'down',
        'a':'left','A':'left','left':'left',
        'd':'right','D':'right','right':'right',
        ' ':'wait'
    }

    def __init__(self, interval=0.2, capture_frames=True, max_frames=2000):
        super().__init__(interval)

        # -----------------------------
        # FIXED CANVAS SETTINGS
        # -----------------------------
        self.dpi = 100
        self.width_px = 1120
        self.height_px = 800
        self.width_in = self.width_px / self.dpi
        self.height_in = self.height_px / self.dpi

        # Capture settings
        self.capture_frames = capture_frames
        self.max_frames = max_frames
        self.frames = []

        # Keyboard queues and repeat
        self._pending_actions = deque()
        self.current_action = 'wait'
        self.last_key = None

        self.key_held = None
        self.key_down_time = None
        self.key_last_repeat = None
        self.repeat_delay = 0.25   # seconds before auto-repeat starts
        self.repeat_rate  = 0.10   # seconds between repeats

        # Trajectory trail
        self.trail = []

        # Training curves
        self.loss_hist = []
        self.rew_hist = []
        self.episode_idx = []
        self.global_step = 0

        self._init_fig()

    # ---------------------------------------------------------
    # Episode hooks for training
    # ---------------------------------------------------------
    def start_episode(self):
        self._ep_return = 0.0

    def record_step(self, reward, loss=None):
        self._ep_return += float(reward)
        if loss is not None:
            self.loss_hist.append(float(loss))
        self.global_step += 1

    def end_episode(self, ep):
        self.rew_hist.append(self._ep_return)
        self.episode_idx.append(ep)
        self._ep_return = 0.0

    # ---------------------------------------------------------
    # Figure setup
    # ---------------------------------------------------------
    def _init_fig(self):
        try:
            plt.ion()
        except:
            pass

        self.fig = plt.figure(figsize=(self.width_in, self.height_in), dpi=self.dpi)
        try:
            self.fig.set_layout_engine(None)
        except:
            pass

        gs = self.fig.add_gridspec(2,2, height_ratios=[3,1], wspace=0.28, hspace=0.28)
        self.ax_grid   = self.fig.add_subplot(gs[0,0])
        self.ax_arrows = self.fig.add_subplot(gs[0,1])
        self.ax_plot   = self.fig.add_subplot(gs[1,:])

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Help text
        self.fig.text(
            0.5, 0.01,
            'Controls: W/A/S/D or ←↑↓→, Space=wait. HOLD = auto-repeat.',
            ha='center', va='bottom', fontsize=9, alpha=0.9
        )

        try:
            self.fig.canvas.manager.window.focus_force()
        except:
            pass

# ============================================================
# gridworld_advanced.py (Part 4/6)
# ============================================================

    # ---------------------------------------------------------
    # Keyboard handlers (with repeat control)
    # ---------------------------------------------------------
    def on_key(self, event):
        self.last_key = event.key
        if event.key in self.KEYMAP:
            act = self.KEYMAP[event.key]
            self._pending_actions.append(act)

            # Mark key as held for repeat
            self.key_held = event.key
            self.key_down_time = self._time()
            self.key_last_repeat = self.key_down_time

    def on_key_release(self, event):
        if self.key_held == event.key:
            self.key_held = None
            self.key_down_time = None
            self.key_last_repeat = None

    def _time(self):
        import time
        return time.perf_counter()

    # ---------------------------------------------------------
    # Auto-repeat logic
    # ---------------------------------------------------------
    def _maybe_repeat(self):
        """Return an auto-repeated action if holding key."""
        if self.key_held is None:
            return None
        if self.key_held not in self.KEYMAP:
            return None

        now = self._time()
        held_time = now - (self.key_down_time or now)

        if held_time < self.repeat_delay:
            return None

        if now - (self.key_last_repeat or now) >= self.repeat_rate:
            self.key_last_repeat = now
            return self.KEYMAP[self.key_held]

        return None

    # ---------------------------------------------------------
    # Consuming actions (queue + repeat fallback)
    # ---------------------------------------------------------
    def consume_action(self):
        if self._pending_actions:
            self.current_action = self._pending_actions.popleft()
            return self.current_action

        repeat_action = self._maybe_repeat()
        if repeat_action is not None:
            self.current_action = repeat_action
            return repeat_action

        self.current_action = 'wait'
        return 'wait'

    # ---------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------
    def _compute_heatmap(self):
        gx, gy = agent.attributes['goal']
        w, h = grid.attributes['width'], grid.attributes['height']
        H = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                H[h-1-y, x] = abs(x-gx) + abs(y-gy)
        return H

    # ---------------------------------------------------------
    # Draw gridworld panel
    # ---------------------------------------------------------
    def _draw_gridworld(self):
        ax = self.ax_grid
        ax.clear()
        w, h = grid.attributes['width'], grid.attributes['height']

        H = self._compute_heatmap()
        ax.imshow(H, cmap='Reds', interpolation='nearest',
                  extent=[-0.5,w-0.5,-0.5,h-0.5], alpha=0.35)

        for (ox, oy) in obstacles.attributes['cells']:
            ax.add_patch(plt.Rectangle((ox-0.5, oy-0.5),1,1,color='black'))

        gx, gy = agent.attributes['goal']
        ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5),1,1,color='gold',alpha=0.9))

        x, y = agent.attributes['x'], agent.attributes['y']
        self.trail.append((x,y))
        if len(self.trail) > 1:
            xs = [p[0] for p in self.trail]
            ys = [p[1] for p in self.trail]
            ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.7)

        ax.plot(x, y, 'o', color='blue', markersize=16)

        ax.set_xticks(np.arange(-0.5,w,1))
        ax.set_yticks(np.arange(-0.5,h,1))
        ax.grid(True, alpha=0.4)
        ax.set_xlim(-0.5,w-0.5)
        ax.set_ylim(-0.5,h-0.5)
        ax.set_aspect('equal')

        ax.set_title(f"Gridworld — Agent=({x},{y}) | Dist={manhattan_distance():.0f}")

        status = f"Last key: {self.last_key or '—'} | Action: {self.current_action}"
        ax.text(0.01,0.99,status,
                transform=ax.transAxes,ha='left',va='top',
                bbox=dict(boxstyle='round',fc='white',alpha=0.7))
        
# ============================================================
# gridworld_advanced.py (Part 5/6)
# ============================================================

    # ---------------------------------------------------------
    # Compute policy field (Q argmax)
    # ---------------------------------------------------------
    def _compute_policy_field(self):
        w, h = grid.attributes['width'], grid.attributes['height']
        gx, gy = agent.attributes['goal']
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        U = np.zeros((h,w), dtype=np.float32)
        V = np.zeros((h,w), dtype=np.float32)
        mask_wait = np.zeros((h,w), dtype=bool)

        vec = {
            'up':(0,1),
            'down':(0,-1),
            'left':(-1,0),
            'right':(1,0),
            'wait':(0,0)
        }

        with torch.no_grad():
            for y in range(h):
                for x in range(w):
                    if (x,y) in obstacles.attributes['cells']:
                        mask_wait[h-1-y,x] = True
                        continue

                    s = np.array([x,y,gx,gy], dtype=np.float32)
                    q = policy.forward(torch.tensor(s))
                    idx = int(torch.argmax(q))
                    act = self.ACTIONS[idx]
                    dx, dy = vec[act]
                    U[h-1-y,x] = dx
                    V[h-1-y,x] = dy
                    if dx==0 and dy==0:
                        mask_wait[h-1-y,x] = True

        Xd, Yd = X, (h-1)-Y
        return Xd, Yd, U, V, mask_wait

    # ---------------------------------------------------------
    # Draw policy-arrow panel
    # ---------------------------------------------------------
    def _draw_policy_arrows(self):
        ax = self.ax_arrows
        ax.clear()

        w, h = grid.attributes['width'], grid.attributes['height']
        H = self._compute_heatmap()

        ax.imshow(H, cmap='Reds', interpolation='nearest',
                  extent=[-0.5,w-0.5,-0.5,h-0.5], alpha=0.25)

        for (ox,oy) in obstacles.attributes['cells']:
            ax.add_patch(plt.Rectangle((ox-0.5,oy-0.5),1,1,color='black',alpha=0.7))

        gx, gy = agent.attributes['goal']
        ax.add_patch(plt.Rectangle((gx-0.5,gy-0.5),1,1,color='gold',alpha=0.9))

        X,Y,U,V,mask = self._compute_policy_field()
        ax.quiver(X,Y,U,V, color='navy', angles='xy', scale_units='xy',
                  scale=1.8, alpha=0.8)

        yy,xx = np.where(mask)
        ax.scatter(xx,yy,s=12,c='navy',alpha=0.6)

        ax.set_xticks(np.arange(-0.5,w,1))
        ax.set_yticks(np.arange(-0.5,h,1))
        ax.grid(True, alpha=0.4)
        ax.set_xlim(-0.5,w-0.5)
        ax.set_ylim(-0.5,h-0.5)
        ax.set_aspect('equal')
        ax.set_title("Policy Arrows — argmax Q")

    # ---------------------------------------------------------
    # Draw metrics (loss + returns)
    # ---------------------------------------------------------
    def _draw_metrics(self):
        ax = self.ax_plot
        ax.clear()

        if self.loss_hist:
            ax.plot(np.arange(1,len(self.loss_hist)+1),
                    self.loss_hist,
                    color='tab:blue',
                    label='Loss (per step)')
            ax.set_xlabel("Training step")
            ax.set_ylabel("Loss", color='tab:blue')
        else:
            ax.set_xlabel("Training step")
            ax.set_ylabel("Loss")

        ax2 = ax.twinx()
        if self.rew_hist:
            ax2.plot(self.episode_idx,
                     self.rew_hist,
                     color='tab:orange',
                     marker='o',
                     label='Return (per episode)')
            ax2.set_ylabel("Episode return", color='tab:orange')
        else:
            ax2.set_ylabel("Episode return")

        ax.set_title("Training Metrics — Loss & Episode Return")

        lines = []
        labels = []
        for axes in (ax,ax2):
            L = axes.get_legend_handles_labels()
            lines += L[0]
            labels += L[1]
        if lines:
            ax.legend(lines, labels, loc='upper right')

# ============================================================
# gridworld_advanced.py (Part 6/6)
# ============================================================

    # ---------------------------------------------------------
    # Frame capture (fixed size)
    # ---------------------------------------------------------
    def _ensure_fixed_canvas(self):
        try:
            self.fig.set_size_inches(self.width_in, self.height_in, forward=True)
            self.fig.set_dpi(self.dpi)
        except:
            pass

    def _fix_frame_size(self, frame):
        h,w,_ = frame.shape
        target_h, target_w = self.height_px, self.width_px
        frame = frame[:target_h, :target_w, :]
        if frame.shape[0] < target_h or frame.shape[1] < target_w:
            out = np.zeros((target_h,target_w,3), dtype=frame.dtype)
            out[:frame.shape[0], :frame.shape[1], :] = frame
            frame = out
        return frame

    # ---------------------------------------------------------
    # Main draw/update loop for daemon
    # ---------------------------------------------------------
    def run(self, coh, dt):
        self._draw_gridworld()
        self._draw_policy_arrows()
        self._draw_metrics()

        self._ensure_fixed_canvas()

        try:
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()
            buf = renderer.buffer_rgba()
            import numpy as np
            frame = np.asarray(buf, dtype=np.uint8)[:,:,:3]
            frame = self._fix_frame_size(frame)
            if self.capture_frames and len(self.frames) < self.max_frames:
                self.frames.append(frame)
        except:
            pass

        try:
            self.fig.canvas.start_event_loop(0.001)
        except:
            pass


# ============================================================
# Q-learning utilities
# ============================================================

viz = AdvancedGridVisualizer(interval=0.2, capture_frames=True)
agent.daemons.append(viz)

def reset_env():
    agent.attributes['x'], agent.attributes['y'] = 0,0
    viz.reset()

def extract_state():
    x,y = agent.attributes['x'], agent.attributes['y']
    gx,gy = agent.attributes['goal']
    return np.array([x,y,gx,gy], dtype=np.float32)

def select_action_epsilon_greedy(s, eps=0.1):
    if np.random.rand() < eps:
        return np.random.choice(viz.ACTIONS)
    with torch.no_grad():
        q = policy.forward(torch.tensor(s))
        return viz.ACTIONS[int(torch.argmax(q))]

def step_env(action):
    d0 = manhattan_distance()
    r = world.apply_method(action)
    d1 = manhattan_distance()
    shaped = r + (d0 - d1)
    if d1 == 0:
        shaped += 10.0
    return shaped

def qlearning_train_episode(ep, max_steps=100, gamma=0.99):
    optimizer = policy.optimizer
    viz.start_episode()

    for t in range(max_steps):
        s = extract_state()
        eps = max(0.05, 0.2 - 0.002*(ep*max_steps + t))
        a = select_action_epsilon_greedy(s, eps)
        r = step_env(a)
        s2 = extract_state()

        with torch.no_grad():
            q_next = policy.forward(torch.tensor(s2)).max().item()
            target = r + gamma*q_next

        logits = policy.forward(torch.tensor(s))
        a_idx = viz.ACTIONS.index(a)
        q_sa = logits[a_idx]
        loss = (q_sa - target)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        viz.record_step(reward=r, loss=float(loss.item()))

        try:
            Simulator(world, dt=0.2, max_steps=1).step('wait')
        except:
            pass

        if manhattan_distance() == 0:
            break

    viz.end_episode(ep)

# ============================================================
# Export & main
# ============================================================

def export_animation(gif='gridworld.gif', mp4='gridworld.mp4', fps=6):
    if not viz.frames:
        print("No frames captured.")
        return
    try:
        imageio.mimsave(gif, viz.frames, fps=fps)
        print(f"Saved {gif}")
    except Exception as e:
        print("GIF export failed:", e)
    try:
        imageio.mimsave(mp4, viz.frames, fps=fps)
        print(f"Saved {mp4}")
    except Exception as e:
        print("MP4 export failed (install ffmpeg).", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['interactive','train'], default='interactive')
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--gif', default='gridworld.gif')
    parser.add_argument('--mp4', default='gridworld.mp4')
    parser.add_argument('--fps', type=int, default=6)
    args = parser.parse_args()

    if args.mode == 'interactive':
        sim = Simulator(world, dt=0.2, max_steps=args.max_steps)
        sim.run(policy=lambda coh: viz.consume_action())
    else:
        for ep in range(1, args.episodes+1):
            qlearning_train_episode(ep)
            print(f"Episode {ep}/{args.episodes} Return={viz.rew_hist[-1]:.2f}")
        agent.attributes['x'], agent.attributes['y'] = 0,0
        sim = Simulator(world, dt=0.2, max_steps=args.max_steps)
        def greedy(_):
            s = extract_state()
            with torch.no_grad():
                a = int(torch.argmax(policy.forward(torch.tensor(s))))
            return viz.ACTIONS[a]
        sim.run(policy=greedy)

    export_animation(args.gif, args.mp4, fps=args.fps)

    try:
        plt.ioff(); plt.show()
    except:
        pass

if __name__ == '__main__':
    main()

