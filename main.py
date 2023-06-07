import numpy as np
import matplotlib.pyplot as plt

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import pandas as pd

if __name__ == "__main__":
	crossovers = ["UniformCrossover", "OnePointCrossover", "CustomCrossover"]
	save_stats = True
	stats = pd.DataFrame(columns=['fitness', 'generation', 'population size', 'crossover'])

	population_sizes = [20, 40, 80]
	num_evaluations_list = []
	num_runs = 1
	num_success = 0
	setA = ['n0000006i05.txt', 'n0000012i05.txt', 'n0000025i05.txt', 'n0000050i05.txt', 'n0000100i05.txt']
	setB = ['n0000009i05.txt', 'n0000016i05.txt', 'n0000025i05.txt', 'n0000049i05.txt', 'n0000100i05.txt']
	setC = ['n0000006i05.txt', 'n0000012i05.txt', 'n0000025i05.txt', 'n0000050i05.txt', 'n0000100i05.txt']
	setD = ['n0000010i05.txt', 'n0000020i05.txt', 'n0000040i05.txt', 'n0000080i05.txt', 'n0000160i05.txt']
	setE = ['n0000010i05.txt', 'n0000020i05.txt', 'n0000040i05.txt', 'n0000080i05.txt', 'n0000160i05.txt']
	instances = setA

	for instance_name in instances:
		inst = "maxcut-instances/setA/" + instance_name
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
				file_name = instance_set + "_" + instance_name.split('.')[0] + "_" + str(population_size) + "_" + str(num_runs)
				stats.to_pickle("results/test_experiment/" + file_name + ".pkl")

