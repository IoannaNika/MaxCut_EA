import networkx as nx
import matplotlib.pyplot as plt

import FitnessFunction


def visualize_graph(adjacency_list: dict, weights: dict, genotype=None):
    graph = nx.DiGraph()
    graph.add_nodes_from(adjacency_list.keys())
    color_map = ['green' for x in adjacency_list.keys()]

    if genotype:
        color_map = ['red' if x == 1 else 'blue' for x in genotype]
    for k, v in adjacency_list.items():
        graph.add_weighted_edges_from(([(k, t, weights[(k,t)]) for t in v]))

    pos = nx.spring_layout(graph, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color=color_map)

    # edges
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5)

    # node labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family="sans-serif")

    # edge weight labels
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.figure(1, figsize=(300, 100), dpi=60)
    plt.show()

if  __name__ == '__main__':
    inst = "maxcut-instances/setD/n0000010i00.txt"
    fitness = FitnessFunction.MaxCut(inst)
    visualize_graph(fitness.adjacency_list, fitness.weights)
