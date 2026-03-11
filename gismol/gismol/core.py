
from __future__ import annotations
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn

class ConstraintViolation(Exception):
    """Raised when an identity constraint is violated."""
    pass

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
    def __init__(self, name: str = '', **kwargs):
        self.name = name
        self.parent: Optional[COH] = None
        self.children: List[COH] = kwargs.get('children', [])
        self.attributes: Dict[str, Any] = kwargs.get('attributes', {})
        self.methods: Dict[str, Callable] = kwargs.get('methods', {})
        self.neural: Dict[str, NeuralModule] = kwargs.get('neural', {})
        self.embedding: Optional[Callable[[COH], np.ndarray]] = kwargs.get('embedding')
        self.identity_constraints: List[Callable[[COH], bool]] = kwargs.get('identity_constraints', [])
        self.trigger_constraints: List[Trigger] = kwargs.get('trigger_constraints', [])
        self.goal_constraints: List[Callable[[COH], float]] = kwargs.get('goal_constraints', [])
        self.daemons: List[Daemon] = kwargs.get('daemons', [])
        # Validate hierarchy (no cycles)
        self._validate_hierarchy()

    def _validate_hierarchy(self):
        """Check that the component graph is a DAG."""
        graph = nx.DiGraph()
        def add_edges(obj: 'COH'):
            for child in obj.children:
                graph.add_edge(obj, child)
                add_edges(child)
        add_edges(self)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('Component hierarchy must be a DAG (no cycles).')

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
        state: Dict[str, Any] = dict(self.attributes)
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

    def apply_method(self, name: str, *args, **kwargs):
        """Execute a method by name and update attributes. Returns reward."""
        if name not in self.methods:
            raise ValueError(f"Unknown method '{name}'")
        method = self.methods[name]
        new_state, reward = method(dict(self.attributes), *args, **kwargs)
        # Ensure method returned a mapping to update
        if not isinstance(new_state, dict):
            raise TypeError('Method must return (state_dict, reward).')
        self.attributes.update(new_state)
        if not self.check_identity():
            raise ConstraintViolation(f"Identity constraint failed after method '{name}'")
        return reward

    def compute_goal(self) -> float:
        """Sum of goal constraints (recursively)."""
        total = 0.0
        for g in self.goal_constraints:
            total += float(g(self))
        for child in self.children:
            total += child.compute_goal()
        return total

    def to_dict(self) -> dict:
        """Serialize to a JSON‑compatible dict."""
        return {
            'name': self.name,
            'attributes': self.attributes,
            'children': [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'COH':
        """Deserialize from a dict."""
        children = [cls.from_dict(c) for c in data.get('children', [])]
        obj = cls(
            name=data.get('name', ''),
            attributes=data.get('attributes', {}),
            children=children
        )
        for ch in children:
            ch.parent = obj
        return obj

class NeuralModule:
    """Wrapper for a PyTorch neural network with training utilities."""
    def __init__(self, module: nn.Module, optimizer_class=None, lr: float = 0.001, **optim_kwargs):
        self.module = module
        self.optimizer = None
        if optimizer_class is not None:
            self.optimizer = optimizer_class(module.parameters(), lr=lr, **optim_kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def step(self, loss):
        if self.optimizer is None:
            raise RuntimeError('No optimizer set; cannot perform training step.')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def save(self, path: str):
        torch.save(self.module.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        state = torch.load(path, map_location='cpu')
        self.module.load_state_dict(state, strict=strict)

class Trigger:
    """Event‑condition‑action rule."""
    def __init__(self, event: str, condition: Callable[[COH], bool], action: Callable[[COH], None]):
        self.event = event  # event name or regex pattern
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
