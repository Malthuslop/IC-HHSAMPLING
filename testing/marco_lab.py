import numpy as np

from itertools import combinations

from utils import *

import time

def test_fleiss_kappa():
    ngroups = 5
    # data from Fleiss 1971 extracted from R package 'irr' diagnoses
    allocs = [ 
         [3, 3, 3, 3, 3, 3],
         [1, 1, 1, 4, 4, 4],
         [1, 2, 2, 2, 2, 4],
         [4, 4, 4, 4, 4, 4],
         [1, 1, 1, 3, 3, 3],
         [0, 0, 2, 2, 2, 2],
         [2, 2, 2, 2, 4, 4],
         [0, 0, 2, 2, 2, 3],
         [0, 0, 3, 3, 3, 3],
         [4, 4, 4, 4, 4, 4],
         [0, 3, 3, 3, 3, 3],
         [0, 1, 3, 3, 3, 3],
         [1, 1, 1, 2, 2, 2],
         [0, 3, 3, 3, 3, 3],
         [1, 1, 3, 3, 3, 4],
         [2, 2, 2, 2, 2, 4],
         [0, 0, 0, 3, 4, 4],
         [0, 0, 0, 0, 0, 1],
         [1, 1, 3, 3, 3, 3],
         [0, 2, 2, 4, 4, 4],
         [4, 4, 4, 4, 4, 4],
         [1, 3, 3, 3, 3, 3],
         [1, 1, 3, 4, 4, 4],
         [0, 0, 3, 3, 3, 3],
         [0, 3, 3, 3, 3, 4],
         [1, 1, 1, 1, 1, 3],
         [0, 0, 0, 0, 4, 4],
         [1, 1, 3, 3, 3, 3],
         [0, 2, 2, 2, 2, 2],
         [4, 4, 4, 4, 4, 4]
    ]

    ans = fleiss_kappa_allocs(allocs, ngroups)
    assert(round(ans,2) == 0.43)

def complete_search_example():
    rng = np.random.default_rng(1)

    x = np.array([[0, 1, 1],
                  [2, 1, 1],
                  [0, 3, 1],
                  [2, 1, 5]])
    n = x.shape[0]
    m = x.shape[1]

    sample_size = int(n/2)
    assert 0 < sample_size < n, f"invalid sample size = {sample_size}"

    lamb = 0.2

    mz = 2
    z = rng.normal(size=(n,mz))

    xz = combine_matrix(x, z, lamb)
    print(f"transformed data & noise lambda={lamb}")
    print(xz)
    print()

    # apply obj to all combinations - ONLY FOR SMALL CASES
    combs = [list(c) for c in combinations(range(n), sample_size)]
    allobj = [(obj_func(wi, xz), wi)  for wi in combs]
    print(allobj)
    print()

    sol = min(allobj)
    print("optimal solution:")
    print(f"wi = {sol[1]}")
    w = [1 if i in sol[1] else 0 for i in range(n)]
    print(f"w = {w}")
    print(f"f(wi) = {sol[0]}")

def test_crossover_runtime(crossover_candidates):

    # build random populations for testing
    #pop_size = (50, 100, 200)
    pop_size = (200, 400, 800)
    n_units = (50, 100, 300, 600, 800, 1000)
    proportion = (0.25, 0.5, 0.75)

    rng = np.random.default_rng(1)

    pop = []
    for size in pop_size:
        for n in n_units:
            for p in proportion:
                pop.append((generate_pop(size, int(p*n), n), n))

    # run and measure runtime for each candidate over testing
    # populations
    npop = len(pop)
    mean_time_ratio = {}
    for cx in crossover_candidates:
        timings = np.empty(npop)
        for i in range(npop):
            # generate offspring with same size as population
            offspring_size = pop[i][0].shape

            st = time.time()
            cx(pop[i][0], offspring_size, None)
            ft = time.time()

            timings[i] = ft - st

        print(cx.__name__)
        mean_time_ratio[cx.__name__] = timings.mean()
        print(f"mean = {timings.mean()} secs")
        print(f"std = {timings.std()} secs\n")

    max_time = max([mean_time_ratio[k] for k in mean_time_ratio])
    for k in mean_time_ratio:
        mean_time_ratio[k] /= max_time
    print(mean_time_ratio)

def crossover_alt(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=int)
    n_p = parents.shape[0]
    idx = 0
    for i in range(offspring_size[0]):
        all_genes = np.union1d(parents[idx], parents[(idx+1)%n_p])
        offspring[i] = np.random.choice(all_genes, offspring_size[1])
        idx = (idx + 1) % n_p
    return offspring

if __name__ == '__main__':
    # test_fleiss_kappa()
    # print("fleiss kappa ok")
    # complete_search_example()
    # print("ok")

    test_crossover_runtime([crossover_func, crossover_alt])
    test_crossover_runtime([crossover_alt, crossover_func])
