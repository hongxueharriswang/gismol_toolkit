
from typing import Callable
from .core import COH

class Constraint:
    """Base class for constraints."""
    def check(self, coh: COH) -> bool:
        raise NotImplementedError

class IdentityConstraint(Constraint):
    """Invariant that must hold in every state."""
    def __init__(self, func: Callable[[COH], bool], name: str = ''):
        self.func = func
        self.name = name

    def check(self, coh: COH) -> bool:
        return self.func(coh)

class GoalConstraint(Constraint):
    """Objective function returning a scalar reward/cost."""
    def __init__(self, func: Callable[[COH], float], weight: float = 1.0):
        self.func = func
        self.weight = weight

    def __call__(self, coh: COH) -> float:
        return float(self.weight) * float(self.func(coh))
