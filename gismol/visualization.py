
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
from .core import COH

def draw_hierarchy(coh: COH, filename: Optional[str] = None):
    """Draw the component DAG."""
    graph = nx.DiGraph()
    def add_nodes(obj: COH):
        graph.add_node(obj.name)
        for child in obj.children:
            graph.add_node(child.name)
            graph.add_edge(obj.name, child.name)
            add_nodes(child)
    add_nodes(coh)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
