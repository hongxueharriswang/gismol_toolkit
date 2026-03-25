
import numpy as np
from .core import COH

def product(*coh_objects: COH) -> COH:
    """Parallel composition: Cartesian product of components, conjunction of constraints."""
    if not coh_objects:
        raise ValueError('Need at least one COH object for product.')
    # Combine children (shallow merge)
    combined_children = []
    for obj in coh_objects:
        combined_children.extend(obj.children)
    # Combine attributes with name‑prefixing
    combined_attrs = {}
    for obj in coh_objects:
        for k, v in obj.attributes.items():
            combined_attrs[f'{obj.name}.{k}'] = v
    # Combine methods with name‑prefixing
    combined_methods = {}
    for obj in coh_objects:
        for k, v in obj.methods.items():
            combined_methods[f'{obj.name}.{k}'] = v
    # Combine identity constraints (all must hold)
    combined_identity = []
    for obj in coh_objects:
        combined_identity.extend(obj.identity_constraints)
    # Combine goals (sum)
    combined_goals = []
    for obj in coh_objects:
        combined_goals.extend(obj.goal_constraints)
    # Triggers and daemons are aggregated
    combined_triggers = []
    combined_daemons = []
    for obj in coh_objects:
        combined_triggers.extend(obj.trigger_constraints)
        combined_daemons.extend(obj.daemons)
    # Neural components are merged (prefix names)
    combined_neural = {}
    for obj in coh_objects:
        for k, v in obj.neural.items():
            combined_neural[f'{obj.name}.{k}'] = v
    # Embedding: concatenate component embeddings (if provided)
    def product_embedding(_coh: COH):
        embs = []
        for obj in coh_objects:
            if obj.embedding:
                embs.append(obj.embedding(obj))
        return np.concatenate(embs) if embs else np.array([], dtype=float)
    return COH(
        name='Product',
        children=combined_children,
        attributes=combined_attrs,
        methods=combined_methods,
        neural=combined_neural,
        embedding=product_embedding,
        identity_constraints=combined_identity,
        trigger_constraints=combined_triggers,
        goal_constraints=combined_goals,
        daemons=combined_daemons,
    )

def coproduct(*_coh_objects: COH) -> COH:
    raise NotImplementedError('Coproduct is not yet implemented.')

def exponential(_domain: COH, _codomain: COH) -> COH:
    raise NotImplementedError('Exponential is not yet implemented.')
