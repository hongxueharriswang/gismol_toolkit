
import time
import re
from typing import Callable, Dict, List, Optional, Any
from .core import COH, Trigger, Daemon, ConstraintViolation

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
            self.event_bus.publish(Event('step', self.time))
            # 2. Decide action
            if policy is None:
                action = self._default_policy(self.root)
            else:
                action = policy(self.root)
            # 3. Apply action (method)
            try:
                _ = self.root.apply_method(action)
            except ConstraintViolation as e:
                self.event_bus.publish(Event('constraint_violated', str(e)))
                self._running = False
                break
            # 4. Post‑step events
            self.event_bus.publish(Event('after_step', self.time))
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
        return float(reward)

    def publish(self, event_name: str, data: Any = None):
        self.event_bus.publish(Event(event_name, data))

    def stop(self):
        self._running = False
