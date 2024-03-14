import pandas as pd
import numpy as np
# import pygad
import time
import random as rd
from statsmodels.stats.inter_rater import *
import math
def collect_data(file_path):
    df = pd.read_table(file_path, sep='\t', header=None, skiprows=[0])
    df = df.drop(columns=0)
    df.columns = range(df.shape[1])
    return df

# data.shape[0], mz = 2
def generate_noise(n, mz):
    z = np.random.default_rng().normal(size=(n,mz))
    return z

def combine_matrix(x, z, lamb): 
    xf = x @ inversecholcov(x)
    zf = z @ inversecholcov(z)
    xz = np.concatenate(((1-lamb)*xf, lamb*zf), axis=1)
    return xz

def inversecholcov(x): # tradução do código do marcelo
    if x.shape[1] == 1: return np.eye(1)
    cov = np.cov(x.T)
    cholcov = np.linalg.cholesky(cov) # lower triangular
    invchol = np.linalg.inv(cholcov)
    return invchol

def obj_func(w_index, x): # w_index = indices do cromossomo, x = Candidatos transformados
    #data_z = combine_matrix(data, z, lambda_noise) 
    #return -1*obj_func(solution, noisy_data.data_z)
    n = x.shape[0] # n = número de linhas de x
    nt = len(w_index) # nt = sample size, tamanho do grupo 1
    nc = n - nt # nc = tamanho do grupo 0

    w = np.zeros(n) # transformação do índice em vetor binário
    w[w_index] = 1  

    mt = (1/nt) * w@x     # calcula as médias com multiplicação de matriz do grupo 1
    mc = (1/nc) * (1-w)@x # calcula as médias com multiplicação de matriz do grupo 0

    d = pow(np.sum((mt-mc)**2), 1/2) # calculo da distância entre as médias,

    return d

def obj_func_pso(sample, data, sample_size, verbose = True):
        index_list = list(np.where(sample == 1)[0])    
        cost = obj_func(w_index=index_list,x = data) + 1000*abs(sample_size - len(index_list))
        return cost

class global_time:
    start = 0
    # last_elapsed_time = 0#elapsed_time()#= 0
    def __init__(self):
        pass
    def set_time(self):
        global_time.start = time.time()
    def elapsed_time(self):
        elapsed = time.time() - global_time.start
        print(elapsed, " segundos passados")
        # last_elapsed_time = elapsed

# gerador de cromossomos de index
def w_generator(sample_size, n):
    w = []
    vec = list(range(n))
    np.random.shuffle(vec)
    w = vec[:sample_size]
    return w

# gerador de populações de cromossomos
def generate_pop(pop_size, sample_size, n):
    pop = []
    for _ in range(pop_size):
        pop.append(np.array(w_generator(sample_size=sample_size, n=n)))
    return np.array(pop)    

def func_generation(ga_instance):
    tempo_limite = 30 
    if (time.time()-global_time.start >= tempo_limite):
        print(tempo_limite, " segundos passados.")
        ga_instance.save('ga_instance')
        return "stop"

class noisy_data:
    data_z = []
    # z = []
    def __init__(self, data, lambda_noise):
        z = generate_noise(data.shape[0], 2)
        noisy_data.data_z = combine_matrix(data, z, lambda_noise)

######### All costum functions
### Costum fitness function
def fitness_func(ga_instance, solution, solution_idx):
    return -1*obj_func(solution, noisy_data.data_z)
    # return d2max - obj_func(solution, xz)

### Costum Crossover function
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    all_genes=list(set(parents[0]) | set(parents[1]))
    while len(offspring) != offspring_size[0]: ## ou seja, no nosso caso, enquanto isso for != 2
        try:
            all_genes = list(set(parents[idx]) | set(parents[idx+1]))
        except IndexError:
            all_genes = list(set(parents[idx]) | set(parents[0]))
            idx = 0
        np.random.shuffle(all_genes)
        son=all_genes[:offspring_size[1]]
        offspring.append(son)
        idx += 1
    return np.array(offspring)

### Costum Mutation function
def mutation_func(offspring, ga_instance):

    fix = list(range(602))
    for c in offspring:
        a_list = fix
        a_list = list(set(a_list)-set(c))
        for i in range(len(c)):
            random_prob = rd.random()
            if random_prob <= 0.01:
                var = np.random.choice(a=a_list)
                a_list.append(c[i])
                c[i] = var
                a_list.remove(var)
    return offspring

def on_start(ga_instance):
    garbage = global_time()
    garbage.set_time()

def on_stop(ga_instance, offspring):
    garbage = global_time()
    garbage.elapsed_time()

def fleiss_kappa_allocs(allocs, ngroups=2):
    """
    Computer fleiss kappa for ngroups to measure level of concordance
    between allocations.
    Each column of alocs is an allocation vector.
    
    """
    counts_table, _ = aggregate_raters(allocs, n_cat=ngroups)
    return fleiss_kappa(counts_table)


def absolute(x):
    if x < 0:
        x = x *-1
    return x


def index_to_binary(w_index, n):
    w = np.zeros(n, dtype=int)
    w[w_index] = 1
    return w
