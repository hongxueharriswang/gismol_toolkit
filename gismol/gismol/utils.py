
import json
import numpy as np
import networkx as nx
from .core import COH

def to_json(coh: COH, filepath: str):
    """Serialize COH object to JSON."""
    with open(filepath, 'w') as f:
        json.dump(coh.to_dict(), f, indent=2)

def from_json(filepath: str) -> COH:
    """Load COH object from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return COH.from_dict(data)

def default_embedding(coh: COH) -> np.ndarray:
    """Concatenate all numeric attribute values (flatten recursively)."""
    values = []
    for _, v in coh.attributes.items():
        if isinstance(v, (int, float)):
            values.append(float(v))
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            values.extend([float(x) for x in v])
    for child in coh.children:
        child_emb = default_embedding(child)
        values.extend(child_emb.tolist())
    return np.array(values, dtype=np.float32)

def is_dag(coh: COH) -> bool:
    """Check if the component graph is a DAG."""
    graph = nx.DiGraph()
    def add_edges(obj: COH):
        for child in obj.children:
            graph.add_edge(obj, child)
            add_edges(child)
    add_edges(coh)
    return nx.is_directed_acyclic_graph(graph)
