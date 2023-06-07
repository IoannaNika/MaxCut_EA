import numpy as np
import matplotlib.pyplot as plt

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import pandas as pd

if __name__ == "__main__":
	crossovers = ["CliqueCrossover", "UniformCrossover", "OnePointCrossover", "CustomCrossover"]
	save_stats = True
	stats = pd.DataFrame(columns=['fitness', 'generation', 'population size', 'crossover'])

	population_sizes = [20, 40, 80]
	num_evaluations_list = []
	num_runs = 1
	num_success = 0
	inst = "maxcut-instances/setD/n0000010i00.txt"

	for population_size in population_sizes:
		for cx in crossovers:
			with open("output-{}.txt".format(cx),"w") as f:

				for i in range(num_runs):
					fitness = FitnessFunction.MaxCut(inst)
					genetic_algorithm = GeneticAlgorithm(fitness,population_size, stats,variation=cx,evaluation_budget=100000,verbose=False)
					best_fitness, num_evaluations, stats = genetic_algorithm.run()
					if best_fitness == fitness.value_to_reach:
						num_success += 1
					num_evaluations_list.append(num_evaluations)

				print("{}/{} runs successful".format(num_success,num_runs))
				print("{} evaluations (median)".format(np.median(num_evaluations_list)))
				percentiles = np.percentile(num_evaluations_list,[10,50,90])
				f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))

		if save_stats:
			instance_set = inst.split('/')[1]
			instance_name = inst.split('/')[2]
			file_name = instance_set + "_" + instance_name + "_" + str(population_size) + "_" + str(num_runs)
			stats.to_pickle("results/test_experiment/" + file_name + ".pkl")

