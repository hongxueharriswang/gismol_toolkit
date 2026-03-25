"""
GISMOL: General Intelligent Systems Modelling Language

A Python implementation of the Constrained Object Hierarchies (COH) 9-tuple
framework for modeling, simulating, and learning in hierarchical,
constraint-driven intelligent systems.

Version: 0.1.0 (alpha)
"""

# Core COH abstractions
from gismol.core import (
    COH,
    NeuralModule,
    Trigger,
    Daemon,
    ConstraintViolation,
)

# Constraint wrappers
from gismol.constraints import (
    Constraint,
    IdentityConstraint,
    GoalConstraint,
)

# Simulation and events
from gismol.simulation import (
    Event,
    EventBus,
    Simulator,
)

# Learning
from gismol.learning import (
    ConstrainedRL,
)

# Category-theoretic composition (experimental)
from gismol.category import (
    product,
    coproduct,
    exponential,
)

# Utilities
from gismol.utils import (
    to_json,
    from_json,
    default_embedding,
    is_dag,
)

# Optional visualization (may require matplotlib)
try:
    from gismol.visualization import draw_hierarchy
except Exception:
    draw_hierarchy = None

__all__ = [
    # Core
    "COH",
    "NeuralModule",
    "Trigger",
    "Daemon",
    "ConstraintViolation",

    # Constraints
    "Constraint",
    "IdentityConstraint",
    "GoalConstraint",

    # Simulation
    "Event",
    "EventBus",
    "Simulator",

    # Learning
    "ConstrainedRL",

    # Category
    "product",
    "coproduct",
    "exponential",

    # Utils
    "to_json",
    "from_json",
    "default_embedding",
    "is_dag",

    # Visualization
    "draw_hierarchy",
]

__version__ = "0.1.0"