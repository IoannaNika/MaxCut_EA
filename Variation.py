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
	
	return [offspring_a, offspring_b]

def clique_crossover(individual_a: Individual, individual_b: Individual, p=0.5):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	assert len(individual_a.genotype) % 5 == 0, "instance should be a 5-clique chain"

	cliques = len(individual_a.genotype) // 5

	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)

	m = np.repeat(np.random.choice((0,1), p=(p, 1-p), size=cliques), 5)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

	for genotype in [offspring_a.genotype, offspring_b.genotype]:
		for i in range(1, cliques - 1):
			# If chain vertices are in the same partition, flip the genotype of the second
			if genotype[i*5-1] == genotype[i*5]:
				genotype[i*5:(i+1)*5] = 1 - genotype[i*5:(i+1)*5]

	return [offspring_a, offspring_b]
