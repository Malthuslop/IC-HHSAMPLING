from params import *
import pygad

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       parent_selection_type=parent_selection_type,
                       keep_elitism=keep_elitism,
                       keep_parents=keep_parents,
                       save_best_solutions=save_best_solutions,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_high=init_range_high,
                       init_range_low=init_range_low,
                       gene_type=gene_type,
                       allow_duplicate_genes=allow_duplicate_genes,
                       on_start=on_start,
                       on_stop=on_stop,
                       on_generation=callback_generation
                      )
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parâmetros da melhor solução encontrada: {solution}".format(solution=np.sort(solution)))
print("Valor Fitness da melhor solução = {solution_fitness}".format(solution_fitness=solution_fitness*-1))
ga_instance.plot_fitness()
# ga_instance.summary()
