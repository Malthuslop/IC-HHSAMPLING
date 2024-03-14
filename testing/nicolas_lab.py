# from scipy.spatial import distances
import numpy as np
import pandas as pd
import random as rd
import pygad

# function that recieves the w(binary array), the original data and two dataframes that will each recieve or the selected samples or the discarded ones
def sample_selection(bin_arr, df, selected_samples, discarded_samples):
    # print(df)
    # print(bin_arr)
    for i in range(len(df)):
        if bin_arr[i] == 1: selected_samples = selected_samples._append(df.loc[i], ignore_index=True)
        elif bin_arr[i] == 0: discarded_samples = discarded_samples._append(df.loc[i], ignore_index=True)
    print(selected_samples)
    print(discarded_samples)
    vec_mean1 = selected_samples.mean()
    vec_mean2 = discarded_samples.mean()
    print(vec_mean1)    
    print(vec_mean2)
    vec_diff = vec_mean1 - vec_mean2  
    euclid_dist = np.sqrt(np.dot(vec_diff.T, vec_diff))
    print(euclid_dist)

# gets the data

def collect_data(file_path):
    df = pd.read_table(file_path, sep='\t', header=None, skiprows=[0])
    df = df.drop(columns=0)
    df.columns = range(df.shape[1])
    return df

# generates random w

def w_generator(sample_size, n):
    w = []
    vec = list(range(n))
    np.random.shuffle(vec)
    w = vec[:sample_size]
    return w


# pathToFile = 'data/Candidatos20230803_n.txt'
# data_frame = collect_data(pathToFile)
# w = w_generator(df=data_frame)

# defining group 1 and 0 
# selected_samples = pd.DataFrame() # group 1
# discarded_samples = pd.DataFrame() # group 0 

# sample_selection(bin_arr=w, df=data_frame, selected_samples=pd.DataFrame(), discarded_samples=pd.DataFrame())
# ^ implementação primária
#############################################################################
# v pygad

#### costum crossover_func
# parentes = lista de pais
# offspring_size = número de filhos gerados
# 0 1 2 3 4 5 6 7 8 9, n=10;nt=3
# [1 4 5] | [0 2 5] (pais, ex com repetição)
# junta as duas listas, mantendo apenas uma cópia de cada elemento = [0 1 2 4 5]
# sorteia 3 elementos dessa lista para cada filho gerado.
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]: ## ou seja, no nosso caso, enquanto isso for != 2
        try:
            all_genes = list(set(parents[idx] + parents[idx+1]))
        except IndexError:
            idx = 0
            all_genes = list(set(parents[idx] + parents[0]))
        np.random.shuffle(all_genes)
        son=all_genes[:offspring_size[1]]
        offspring.append(son)
        idx += 1
    return np.array(offspring)

offspring_size= [2, 120]
parent1 = w_generator(sample_size=120, n=602)
parent2 = w_generator(sample_size=120, n=602)
parents = [parent1, parent2]
ga_instance = []
final = crossover_func(parents=parents, offspring_size=offspring_size, ga_instance=ga_instance)
print(final)
print(len(final))
print(len(final[0]))
print(len(final[1]))
# a princípio é isso aqui
# def crossover_test():
#     parent1 = [1, 4, 5]
#     parent2 = [0, 2, 5]
#     parents = [parent1, parent2]
#     all_genes = list(set(parent1 + parent2))
#     print(all_genes)
#     for parent in parents:
#         np.random.shuffle(all_genes)
#         son=all_genes[:3]
#         print(son)
# crossover_test()

#### costum mutation_func
# 0 1 2 3 4 5 6 7 8 9, n=10;nt=5
# [0, 3, 4, 6, 9]
# para cada elemento dessa lista, a gente vai sortear quais index vão ser alterados. Caso positivo, troca por algum índice fora do vetor. Caso negativo, passa
# [0, 3, 4, 6, 9]
#  1  0  0  1  0 -> definir a probabilidade de mutação = 0.01 -> 1 troca, 0 não troca
# [1, 3, 4, 5, 9]
# def mutation_func(offspring, ga_instance):
#
#     for chromosome_idx in range(offspring.shape[0]):
#         random_gene_idx = np.random.choice(range(offspring.shape[1]))
#
#         offspring[chromosome_idx, random_gene_idx] += np.random.random()
#
#     return offspring

def mutation_func(offspring, ga_instance):



    # gera lista complementar do cromossomo
    a_list = list(range(602))
    a_list = list(set(a_list)-set(offspring))

    # loop que aplica a mutação com base em probabilidade 
    for i in range(len(offspring)):
        random_prob = rd.random()
        if random_prob <= 0.01:
            var = np.random.choice(a=a_list)
            a_list.append(offspring[i])
            offspring[i] = var
            a_list.remove(var)

    return offspring

# def mutation_func(offspring, ga_instance):

#     for chromosome_idx in range(offspring.shape[0]):
#         random_gene_idx = np.random.choice(range(offspring.shape[1]))

#         offspring[chromosome_idx, random_gene_idx] += np.random.random()

#     return offspring

# print(mutation_func(offspring=parent1, ga_instance=1))

