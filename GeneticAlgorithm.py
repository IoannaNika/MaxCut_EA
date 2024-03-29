import numpy as np
import time
from functools import partial 

import Variation
import Selection
import Mutation
from FitnessFunction import FitnessFunction
from Individual import Individual
from Utils import ValueToReachFoundException

class GeneticAlgorithm:
	def __init__(self, fitness: FitnessFunction, population_size, stats, round, **options ):
		self.fitness = fitness
		self.evaluation_budget = 1000000
		self.variation_operator = Variation.uniform_crossover
		self.mutation_operator = Mutation.identity_mutation
		self.selection_operator = Selection.tournament_selection
		self.population_size = population_size
		self.population = []
		self.number_of_generations = 0
		self.verbose = False
		self.print_final_results = True
		self.stats = stats
		self.round = round
		self.options = options

		if "verbose" in options:
			self.verbose = options["verbose"]

		if "evaluation_budget" in options:
			self.evaluation_budget = options["evaluation_budget"]

		if "variation" in options:
			if options["variation"] == "UniformCrossover":
				self.variation_operator = Variation.uniform_crossover
			elif options["variation"] == "OnePointCrossover":
				self.variation_operator = Variation.one_point_crossover
			elif options["variation"] == "TwoPointCrossover":
				self.variation_operator = Variation.two_point_crossover
			elif options["variation"] == "CustomCrossover":
				self.variation_operator = partial(Variation.custom_crossover, self.fitness)
			elif options["variation"] == "CliqueCrossover":
				self.variation_operator = Variation.clique_crossover
			elif options["variation"] == "CustomCliqueCrossover":
				self.variation_operator = partial(Variation.clique_and_custom_crossover, self.fitness)
			elif options["variation"] == "KMeansCrossover":
				self.variation_operator = partial(Variation.k_means_crossover, self.fitness)

		if "mutation" in options:
			if options["mutation"] == "IdentityMutation":
				self.mutation_operator = Mutation.identity_mutation
			elif options["mutation"] == "RandomMutation":
				self.mutation_operator = Mutation.random_mutation

	def initialize_population( self ):
		self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in range(self.population_size)]
		for individual in self.population:
			self.fitness.evaluate(individual, self.get_best_fitness(), self.number_of_generations, self.population_size, self.options['variation'], self.round, self.stats)

	def make_offspring( self ):
		offspring = []
		order = np.random.permutation(self.population_size)
		for i in range(len(order)//2):
			offspring = offspring + [self.mutation_operator(o) for o in self.variation_operator(self.population[order[2*i]],self.population[order[2*i+1]])]
		for individual in offspring:
			self.fitness.evaluate(individual, self.get_best_fitness(), self.number_of_generations, self.population_size, self.options['variation'], self.round, self.stats)
		return offspring

	def make_selection( self, offspring ):
		return self.selection_operator(self.population, offspring)
	
	def print_statistics( self ):
		fitness_list = [ind.fitness for ind in self.population]
		print("Generation {}: Best_fitness: {:.1f}, Avg._fitness: {:.3f}, Nr._of_evaluations: {}".format(self.number_of_generations,max(fitness_list),np.mean(fitness_list),self.fitness.number_of_evaluations))

	def get_best_fitness( self ):
		return max([ind.fitness for ind in self.population])

	def run( self ):
		try:
			self.initialize_population()
			prev_fitness = self.get_best_fitness()
			patience = 50
			patience_counter = 0
			while( self.fitness.number_of_evaluations < self.evaluation_budget ):
				self.number_of_generations += 1
				if( self.verbose and self.number_of_generations%100 == 0 ):
					self.print_statistics()

				offspring = self.make_offspring()
				selection = self.make_selection(offspring)
				self.population = selection

				best_fitness = self.get_best_fitness()

				# If fitness has not improved, increae patience counter
				if best_fitness <= prev_fitness:
					patience_counter += 1
				else:
					# Otherwise reset patience counter
					patience_counter = 0

				prev_fitness = best_fitness

				if patience_counter == patience:
					print("Stopped early due to low performance, patience counter was reached: ", patience)
					print("Generation", self.number_of_generations)
					break
					
			if( self.verbose ):
				self.print_statistics()
		except ValueToReachFoundException as exception:
			if( self.print_final_results ):
				print(exception)
				print("Best fitness: {:.1f}, Nr._of_evaluations: {}".format(exception.individual.fitness, self.fitness.number_of_evaluations))
			return exception.individual.fitness, self.fitness.number_of_evaluations, self.stats
		if( self.print_final_results ):
			self.print_statistics()
		return self.get_best_fitness(), self.fitness.number_of_evaluations, self.stats

