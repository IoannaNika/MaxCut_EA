import numpy as np
import matplotlib.pyplot as plt

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
	crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
	colors = ['blue', 'green', 'm']
	cross_over_fitnesses = []

	for cx in crossovers:
		inst = "maxcut-instances/setE/n0000020i00.txt"
		with open("output-{}.txt".format(cx),"w") as f:
			population_size = 500
			num_evaluations_list = []
			num_runs = 10
			num_success = 0
			all_fitnesses = []
			for i in range(num_runs):
				fitness = FitnessFunction.MaxCut(inst)
				genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False)
				best_fitness, num_evaluations, fitnesses, num_generations = genetic_algorithm.run()
				if best_fitness == fitness.value_to_reach:
					num_success += 1
				num_evaluations_list.append(num_evaluations)

				# 30 is arbitary number to smaller the graph
				if len(fitnesses) > 30:
					fitnesses = fitnesses[0:30]

				full_fitness = [fitness.value_to_reach for i in range(30)]
				full_fitness[0:len(fitnesses)] = fitnesses
				all_fitnesses.append(full_fitness)

			print("{}/{} runs successful".format(num_success,num_runs))
			print("{} evaluations (median)".format(np.median(num_evaluations_list)))
			percentiles = np.percentile(num_evaluations_list,[10,50,90])
			f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))
			cross_over_fitnesses.append(all_fitnesses)

	generations = list(range(0,30))
	for i in range(len(crossovers)):
		for j in range(len(cross_over_fitnesses[i])):
			plt.scatter(generations, cross_over_fitnesses[i][j], c=colors[i])
		median_line = np.median(cross_over_fitnesses[i], axis=0)
		plt.plot(median_line, c=colors[i], label=str(crossovers[i]))

	plt.xlabel("Generations")
	plt.ylabel("Fitness")
	plt.title("")
	plt.legend()
	plt.show()
