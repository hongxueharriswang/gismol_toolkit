
"""gismol: General Intelligent Systems Modelling Language.
A Python implementation of the Constrained Object Hierarchies (COH) 9‑tuple framework
for designing, simulating, and learning in hierarchical, constraint‑driven intelligent systems.
"""
from .core import COH, NeuralModule, Trigger, Daemon, ConstraintViolation
from .constraints import IdentityConstraint, GoalConstraint
from .simulation import Simulator
from . import utils, category, learning

__all__ = [
    'COH', 'NeuralModule', 'Trigger', 'Daemon', 'ConstraintViolation',
    'IdentityConstraint', 'GoalConstraint', 'Simulator', 'utils', 'category', 'learning'
]

__version__ = '0.1.0'
