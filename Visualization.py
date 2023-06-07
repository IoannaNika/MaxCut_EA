import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import FitnessFunction


def visualize_graph(genotype):
    inst = "maxcut-instances/setD/n0000010i00.txt"
    fitness = FitnessFunction.MaxCut(inst)
    adjacency_list = fitness.adjacency_list
    weights = fitness.weights
    graph = nx.DiGraph()
    graph.add_nodes_from(adjacency_list.keys())
    color_map = ['green' for x in adjacency_list.keys()]


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

def visualize_fitness_generation(data):
    cross_over_data = data.groupby("crossover")
    fig, ax = plt.subplots()

    for i, (group_name, group_data) in enumerate(cross_over_data):
        sorted_generation = group_data.sort_values('generation')
        medians = sorted_generation.groupby('generation')['fitness'].median()

        generation_values = medians.index
        fitness_values = medians.values
        ax.plot(generation_values, fitness_values)
        ax.scatter(group_data['generation'], group_data['fitness'], label=group_name)

    ax.legend()
    ax.set_xlim(0, 30)
    plt.title("fitness vs generation")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.show()


def visualize_fitness_generation_population(data, crossover, names):
    fig, ax = plt.subplots()

    for i, data_population in enumerate(data):
        current_crossover = data_population.loc[data_population['crossover'] == crossover]

        sorted_generation = current_crossover.sort_values('generation')
        medians = sorted_generation.groupby('generation')['fitness'].median()
        generation_values = medians.index
        fitness_values = medians.values
        ax.plot(generation_values, fitness_values, label=names[i])

    ax.legend()
    ax.set_xlim(0, 30)
    plt.title("fitness vs generation for " + crossover + " per population size ")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.show()

def visualize_avg_fitness_population(data, population_size, cross_overs):
    fig, ax = plt.subplots()
    all_fitness = [[] for i in range(len(cross_overs))]

    for i, crossover in enumerate(cross_overs):
        for d in data:
            all_fitness[i].append(d.groupby('crossover')['fitness'].mean()[cross_overs[i]])

    for i, avg_fitness in enumerate(all_fitness):
        ax.plot(population_size, avg_fitness, label=cross_overs[i])

    ax.legend()
    plt.title("avg fitness vs population size")
    plt.xlabel("population size")
    plt.ylabel("fitness")
    plt.show()



if  __name__ == '__main__':
    # inst = "maxcut-instances/setD/n0000010i00.txt"
    # fitness = FitnessFunction.MaxCut(inst)
    # visualize_graph(fitness.adjacency_list, fitness.weights)

    twenty = pd.read_pickle("results/test_experiment/setD_n0000010i00.txt_20_1.pkl")
    thirthy = pd.read_pickle('results/test_experiment/setD_n0000010i00.txt_40_1.pkl')
    fourthy = pd.read_pickle("results/test_experiment/setD_n0000010i00.txt_80_1.pkl")

    visualize_avg_fitness_population([twenty, thirthy, fourthy], [20,40, 80], ["CliqueCrossover", "UniformCrossover", "OnePointCrossover", "CustomCrossover"])
    visualize_fitness_generation_population([twenty, thirthy, fourthy], "CustomCrossover", ["20", '40', '80'])