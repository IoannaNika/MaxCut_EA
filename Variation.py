import numpy as np

from Individual import Individual
from FitnessFunction import FitnessFunction, MaxCut

def uniform_crossover(individual_a: Individual, individual_b: Individual, p = 0.5 ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
    
	m = np.random.choice((0,1), p=(p, 1-p), size=l)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)
	
	return [offspring_a, offspring_b]

def one_point_crossover(individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
    
	l = len(individual_a.genotype)
	m = np.arange(l) < np.random.randint(l+1)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)
	
	return [offspring_a, offspring_b]

def two_point_crossover(individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	offspring_a = Individual()
	offspring_b = Individual()
    
	l = len(individual_a.genotype)
	m = (np.arange(l) < np.random.randint(l+1)) ^ (np.arange(l) < np.random.randint(l+1))
	offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
	offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)
	
	return [offspring_a, offspring_b]

def custom_crossover( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual, offset=0):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
   
   	# Implement your custom crossover here
	offspring_a.genotype = individual_a.genotype.copy()
	offspring_b.genotype = individual_b.genotype.copy()

	# indegree: weight sum of edges incident to vertices in the same set
	# outdegree: weight sum of edges incident to vertices in different sets
	
	# get ajacency matrix
	adjacency_matrix = fitness.adjacency_list

	# get indegree and outdegree
	indegrees = np.zeros(l)
	outdegrees = np.zeros(l)
	for i in range(l):
		for j in range(l):
			if i != j:
				if offspring_a.genotype[i] == offspring_a.genotype[j] and j+offset in adjacency_matrix[i+offset]:
					indegrees[i] += fitness.get_weight(i+offset,j+offset)
				elif offspring_a.genotype[i] != offspring_a.genotype[j] and j+offset in adjacency_matrix[i+offset]:
					outdegrees[i] += fitness.get_weight(i+offset,j+offset)

	# calculate the gain per node defined as the difference between indegree and outdegree
	gains = np.zeros(l)
	for i in range(l):
		gains[i] = indegrees[i] - outdegrees[i]
		if gains[i] < 0:
			gains[i] = 0
	
	# normalize gains as probabilities
	if np.sum(gains) != 0:
		gains = gains / np.sum(gains)
	else:
		gains = np.ones(l) / l

	# adjust probabilities so that the number of 1s and 0s  is balanced
	# count number of 1s and 0s
	num_ones = np.sum(offspring_a.genotype)
	num_zeros = l - num_ones
	# adjust probabilities
	for i in range(l):
		if offspring_a.genotype[i] == 1:
			gains[i] = gains[i] * num_zeros
		else:
			gains[i] = gains[i] * num_ones
	
	# normalize again
	if np.sum(gains) != 0:
		gains = gains / np.sum(gains)
	else:
		gains = np.ones(l) / l


	# perform crossover with probability proportional to gain
	for i in range(l):
		if np.random.uniform() < gains[i]:
			offspring_a.genotype[i] = 1 - offspring_a.genotype[i]
			offspring_b.genotype[i] = 1 - offspring_b.genotype[i]
		
	
	return [offspring_a, offspring_b]

def clique_crossover(individual_a: Individual, individual_b: Individual, p=0.5):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	assert len(individual_a.genotype) % 5 == 0, "instance should be a 5-clique chain"

	cliques = len(individual_a.genotype) // 5

	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)

	m = np.repeat(np.random.choice((0, 1), p=(p, 1 - p), size=cliques), 5)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

	for genotype in [offspring_a.genotype, offspring_b.genotype]:
		for i in range(1, cliques - 1):
			# If chain vertices are in the same partition, flip the genotype of the second
			if genotype[i * 5 - 1] == genotype[i * 5]:
				genotype[i * 5:(i + 1) * 5] = 1 - genotype[i * 5:(i + 1) * 5]

	return [offspring_a, offspring_b]

def k_means_crossover( maxCut: MaxCut, individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)

	offspring_a.genotype = individual_a.genotype.copy()
	offspring_b.genotype = individual_b.genotype.copy()

	clusters = maxCut.k_means_clusters
	for c in np.unique(clusters):
		cluster_indices = np.argwhere(clusters == c)
		c = Individual(cluster_indices.shape[0])
		d = Individual(cluster_indices.shape[0])
		c.genotype = individual_a.genotype[cluster_indices]
		d.genotype = individual_b.genotype[cluster_indices]

		[c, d] = custom_crossover(maxCut, c, d)

		offspring_a.genotype[cluster_indices] = c.genotype
		offspring_b.genotype[cluster_indices] = d.genotype



	m = clusters % 2 == np.random.randint(0,1)


	offspring_a.genotype = np.where(m, offspring_a.genotype, offspring_b.genotype)
	offspring_b.genotype = np.where(1 - m, offspring_a.genotype, offspring_b.genotype)

	return [offspring_a, offspring_b]

def clique_and_custom_crossover(fitness: FitnessFunction, individual_a: Individual, individual_b: Individual, p=0.5):
        # Apply custom crossover on individual cliques
        l = len(individual_a.genotype)

        for i in range(0, l, 5):
                clique_a, clique_b = Individual(5), Individual(5)
                clique_a.genotype = individual_a.genotype[i:i+5]
                clique_b.genotype = individual_b.genotype[i:i+5]

                # i is passed as an offset to custom crossover to match the vertices of the original graph
                clique_a, clique_b = custom_crossover(fitness, clique_a, clique_b, i)

                individual_a.genotype[i:i+5] = clique_a.genotype
                individual_b.genotype[i:i+5] = clique_b.genotype

        # Apply clique crossover on the entire graph
        return clique_crossover(individual_a, individual_b, p)
