from hhsolver import *

# HHsampling = função para gerar as alocações com o método do HH (haphazard)
# x is transformed data
# lamb is pure lambda in [0,1]
# sample_size is number of elements to sample
# noise is list of size n, alloc with transformed noise matrices
# returns a dictionary with entries:
# allocs: matrix n x n alloc where each column j is an allocation associated to noise j
# distq95: 95th percentile of Mahalanobis distances for each allocation using only original data x
# fkappa: fleiss kappa of allocations

def hhsampling(x, lamb, nalloc, sample_size, noise, populations, hh_solver_strategy, time_limit, *args):

    ma = noise[0].shape[1] # number of columns in noise matrix
    m = x.shape[1]

    dist = []

    # CHECK IF LAMBDA IS CORRECT
    lamb = lamb/(lamb*(1-ma/m) + ma/m)

    dist = []

    allocs = np.zeros((nalloc, x.shape[0]), dtype=int)

    for j in range(nalloc):

        xz = np.concatenate(((1-lamb)*x, lamb*noise[j]), axis=1)

        solver = hh_solver_strategy(xz, sample_size) 

        w_index = solver.get_best_alloc(populations[j], time_limit, *args)
        d = obj_func(w_index,x)
        dist.append(d)


    return dist
