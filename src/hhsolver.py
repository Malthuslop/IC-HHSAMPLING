# HHsolver = tem como objetivo rodar o algoritmo genético 
# para resolver o problema de alocação de forma otimizada

from utils import *
from binary_with_time_control import *

import pygad
import pandas as pd
import statsmodels.api
from statsmodels import stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

import pyswarms as ps
import math

# Compute best allocation for pre-combined data.
class HHSolver:
    def __init__(self, data, sample_size):#):, info):
        self.data = data
        self.sample_size = sample_size
        self.init_pop = []


    def fitness(self, ga_instance, solution, solution_idx):
        return -1*obj_func(solution, self.data)

    def get_best_alloc(self):

        fitness_function = self.fitness
        mutation_type = mutation_func
        crossover_type = crossover_func

        num_generations = 50
        num_parents_mating = 2

        gene_type = int

        allow_duplicate_genes = False

        parent_selection_type="sss"
        keep_parents=1
        keep_elitism=0

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               parent_selection_type=parent_selection_type,
                               keep_elitism=keep_elitism,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               initial_population=self.init_pop,
                            #    sol_per_pop=sol_per_pop,
                            #    num_genes=num_genes,
                            #    init_range_high=init_range_high,
                            #    init_range_low=init_range_low,
                               gene_type=gene_type,
                               allow_duplicate_genes=allow_duplicate_genes,
                               on_start=on_start,
                               on_stop=on_stop,
                              )
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        return solution, obj_func(solution, self.data)

class HHSolverPSO(HHSolver):
    def __init__(self, data, sample_size):#):, info):
        super().__init__(data,sample_size)
    
    def fitness(self,sample):
        return obj_func_pso(sample = sample,data = self.data,sample_size = self.sample_size, verbose = False)
    
    def get_best_alloc(self,init_pop, time_limit, *args):
        # params
        dimensions_num = self.data.shape[0] #Rows' number
        options = {'c1': args[1], 'c2': args[2], 'w': args[3], 'k':1, 'p':2}
        iter_num = 100000000000
        particles_num = args[0]
        init_pop_list = []

        #convert index list to binary numpy array
        for i in range(particles_num):
            init_pop_list.append(index_to_binary(init_pop[i],dimensions_num))
        
        init_pop = np.array(init_pop_list)

        optimizer1 = BinaryPSOWithTimeStopCriteria(n_particles=particles_num, 
                                           dimensions=dimensions_num,
                                           init_pos=init_pop, 
                                           options=options)
        
        cost, pos, time = optimizer1.optimize(objective_func = self.fitness, verbose = False, iters = iter_num, time_limit = time_limit)
        index_solution = np.where(pos == 1)[0]
        return index_solution
    # TODO: get_best_alloc_GU