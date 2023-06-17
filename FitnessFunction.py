import numpy as np
import itertools as it

import Individual
from Utils import ValueToReachFoundException

class FitnessFunction:
	def __init__( self ):
		self.dimensionality = 1 
		self.number_of_evaluations = 0
		self.value_to_reach = np.inf

	def evaluate( self, individual: Individual, fitness, generation, population_size, cross_over_type, round, stats):
		self.number_of_evaluations += 1
		stats.loc[len(stats)] = [fitness, generation, self.number_of_evaluations, population_size, cross_over_type, round]
		if individual.fitness >= self.value_to_reach:
			fitness = individual.fitness
			stats.loc[len(stats)] = [fitness, generation, self.number_of_evaluations, population_size, cross_over_type, round]
			raise ValueToReachFoundException(individual)

class OneMax(FitnessFunction):
	def __init__( self, dimensionality ):
		super().__init__()
		self.dimensionality = dimensionality
		self.value_to_reach = dimensionality

	def evaluate( self, individual: Individual, fitness, generation, population_size, cross_over_type, round, stats):
		individual.fitness = np.sum(individual.genotype)
		super().evaluate(individual, fitness, generation, population_size, cross_over_type, round, stats)

class DeceptiveTrap(FitnessFunction):
	def __init__( self, dimensionality ):
		super().__init__()
		self.dimensionality = dimensionality
		self.trap_size = 5
		assert dimensionality % self.trap_size == 0, "Dimensionality should be a multiple of trap size"
		self.value_to_reach = dimensionality

	def trap_function( self, genotype ):
		assert len(genotype) == self.trap_size
		k = self.trap_size
		bit_sum = np.sum(genotype)
		if bit_sum == k:
			return k
		else:
			return k-1-bit_sum

	def evaluate( self, individual: Individual,fitness, generation, population_size, cross_over_type, round, stats):
		num_subfunctions = self.dimensionality // self.trap_size
		result = 0
		for i in range(num_subfunctions):
			result += self.trap_function(individual.genotype[i*self.trap_size:(i+1)*self.trap_size])
		individual.fitness = result
		super().evaluate(individual, fitness, generation, population_size, cross_over_type, round, stats)

class MaxCut(FitnessFunction):
	def __init__( self, instance_file ):
		super().__init__()
		self.edge_list = []
		self.weights = {}
		self.adjacency_list = {}
		self.read_problem_instance(instance_file)
		self.read_value_to_reach(instance_file)
		self.preprocess()

	def preprocess( self ):
		self.distance_matrix = self.distance_matrix()
		clusters = int(np.sqrt(self.dimensionality))
		self.k_means_clusters = self.k_means(clusters)


	def read_problem_instance( self, instance_file ):
		with open( instance_file, "r" ) as f_in:
			lines = f_in.readlines()
			first_line = lines[0].split()
			self.dimensionality = int(first_line[0])
			number_of_edges = int(first_line[1])
			for line in lines[1:]:
				splt = line.split()
				v0 = int(splt[0])-1
				v1 = int(splt[1])-1
				assert( v0 >= 0 and v0 < self.dimensionality )
				assert( v1 >= 0 and v1 < self.dimensionality )
				w = float(splt[2])
				self.edge_list.append((v0,v1))
				self.weights[(v0,v1)] = w
				self.weights[(v1,v0)] = w
				if( v0 not in self.adjacency_list ):
					self.adjacency_list[v0] = []
				if( v1 not in self.adjacency_list ):
					self.adjacency_list[v1] = []
				self.adjacency_list[v0].append(v1)
				self.adjacency_list[v1].append(v0)
			assert( len(self.edge_list) == number_of_edges )
	
	def read_value_to_reach( self, instance_file ):
		bkv_file = instance_file.replace(".txt",".bkv")
		with open( bkv_file, "r" ) as f_in:
			lines = f_in.readlines()
			first_line = lines[0].split()
			self.value_to_reach = float(first_line[0])

	def get_weight( self, v0, v1 ):
		if( not (v0,v1) in self.weights ):
			return 0
		return self.weights[(v0,v1)]

	def get_degree( self, v ):
		return len(self.adjacency_list(v))

	def get_shortest_paths(self, v):
		shortest_path = np.full((self.dimensionality), float('inf'))
		shortest_path[v] = 0
		queue = [v]
		
		while len(queue) > 0:
			node = queue.pop()
			for next_node in self.adjacency_list[node]:
				distance = shortest_path[node] + self.weights[(node, next_node)]
				if  distance < shortest_path[next_node]:
					shortest_path[next_node] = distance
					queue.append(next_node)
		return shortest_path

	def distance_matrix(self):
		distance_matrix = np.zeros((self.dimensionality, self.dimensionality))
		for i in range(self.dimensionality):
			distance_matrix[i] = self.get_shortest_paths(i)
		return distance_matrix

	def k_means(self, k, max_iters=100):
		clusters = np.zeros(self.dimensionality)
		# randomly choose the centroids
		centroids = np.random.choice(self.dimensionality, size=k, replace=False)

		for iteration in range(max_iters):

			# Create a matrix only keeping distances to selected centroids
			centroid_dists = np.where(np.isin(np.arange(self.dimensionality), centroids),
						  self.distance_matrix, np.full_like(self.distance_matrix, float('inf')))

			# Put each node in a cluster by choosing the centroid with the minimum distance to it
			# This array contains the index of the cluster for each node
			node_centroid = np.argmin(centroid_dists, axis=1)

			mean_dists = np.zeros((k))
			for i in range(k):
				assert(centroids[i] in np.unique(node_centroid))
				clusters[np.argwhere(node_centroid == centroids[i])] = i
				# Set all rows and columns of nodes not in this cluster to 0
				not_in_cluster_indices = np.argwhere(node_centroid != centroids[i])
				dists_cluster = self.distance_matrix.copy()
				dists_cluster[not_in_cluster_indices, :] = 0
				dists_cluster[:, not_in_cluster_indices] = 0

				# Get mean of distance to all other nodes in cluster
				mean_dist = np.mean(dists_cluster, axis=0)
				mean_dist[not_in_cluster_indices] = float('inf')

				mean_dists[i] = np.min(mean_dist)
				# The node with the minimum mean distance to all other nodes in the cluster is the new centroid
				centroids[i] = np.argmin(mean_dist)
			#print("Mean distance sum: ", np.sum(mean_dists))
		return clusters

	def evaluate( self, individual: Individual, fitness, generation, population_size, cross_over_type, round, stats):
		result = 0
		for e in self.edge_list:
			v0, v1 = e
			w = self.weights[e]
			if( individual.genotype[v0] != individual.genotype[v1] ):
				result += w

		individual.fitness = result
		super().evaluate(individual, fitness, generation, population_size, cross_over_type, round, stats)

