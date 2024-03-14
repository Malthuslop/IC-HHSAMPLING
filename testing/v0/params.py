from utils import *

# z = generate_noise(data.shape[0], 2)
# data_z = combine_matrix(data, z, lambda_noise)



file_path = '../data/Candidatos20230803_n.txt'
data = collect_data(file_path=file_path).to_numpy()
sample_size = 120
lambda_noise = 0.2
### Costum Functions
noisy_data(data, lambda_noise)
fitness_function=fitness_func ## checar o nome certo da esquerda
mutation_type=mutation_func
crossover_type=crossover_func
callback_generation = func_generation

#########################################
# Creating initial pop
sol_per_pop = 250 # tamanho da população
num_genes = sample_size # número de genes no cromossomo
init_range_low=0
init_range_high=601
#########################################
allow_duplicate_genes=False
num_generations=250
num_parents_mating=2
# initial_population=generate_pop(pop_size=50, sample_size=sample_size, n=data.shape[0])
gene_type=int
parent_selection_type="sss"
keep_parents=1
keep_elitism=0
save_best_solutions=False
stop_criteria = "reach_0"
parallel_processing=['process', 10]
on_start=on_start
on_stop=on_stop