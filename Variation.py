import numpy as np

from Individual import Individual
from FitnessFunction import FitnessFunction

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

def custom_crossover( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual ):
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
				if offspring_a.genotype[i] == offspring_a.genotype[j] and j in adjacency_matrix[i]:
					indegrees[i] += fitness.get_weight(i,j)
				elif offspring_a.genotype[i] != offspring_a.genotype[j] and j in adjacency_matrix[i]:
					outdegrees[i] += fitness.get_weight(i,j)

	# calculate the gain per node defined as the difference between indegree and outdegree
	gains = np.zeros(l)
	for i in range(l):
		gains[i] = indegrees[i] - outdegrees[i]
	
	# normalize gains as probabilities
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

