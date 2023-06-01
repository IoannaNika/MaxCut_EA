import numpy as np

from Individual import Individual

def identity_mutation(individual: Individual):
	return individual

def random_mutation(individual: Individual, p = 0.05 ):
	mutation = Individual(individual.genotype)

	l = len(individual.genotype)
	m = np.random.choice((0,1), p=(p, 1-p), size=l)
	mutation.genotype = np.where(m, 1 - individual.genotype, individual.genotype)

	return mutation
