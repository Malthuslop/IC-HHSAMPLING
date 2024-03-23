# HHsampling = função para gerar as alocações com o método do HH (haphazard)

from hhsolver import *
import aspose.words as aw
from PIL import Image

# x is transformed data
# lamb is pure lambda in [0,1]
# sample_size is number of elements to sample
# noise is list of size n, alloc with transformed noise matrices
# returns a dictionary with entries:
#   allocs: matrix n x n alloc where each column j is an allocation associated to noise j
#   distq95: 95th percentile of Mahalanobis distances for each allocation using only original data x
#   fkappa: fleiss kappa of allocations
def hhsampling(x, lamb, nalloc, sample_size, noise, populations, hh_solver_strategy, time_limit):

    ma = noise[0].shape[1] # number of columns in noise matrix
    m = x.shape[1]

    res = {"allocs": None, "distq95": 0, "fkappa": 0, "dist": None}

    # CHECK IF LAMBDA IS CORRECT
    lamb = lamb/(lamb*(1-ma/m) + ma/m)

    dist = []

    allocs = np.zeros((nalloc, x.shape[0]), dtype=int)

    for j in range(nalloc):

        xz = np.concatenate(((1-lamb)*x, lamb*noise[j]), axis=1)

        solver = hh_solver_strategy(xz, sample_size) # 
        solver.init_pop = populations[j] # generate_pop(50, solver.sample_size, solver.data.shape[0]) # !!

        w_index = solver.get_best_alloc(time_limit)
        d = obj_func(w_index,x)
        allocs[j][w_index] = 1
        dist.append(d)

    res["allocs"] = allocs.T
    res["dist"] = dist
    distq95 = np.percentile(np.array(dist), 95)
    
    res["distq95"] = distq95
    fkappaval = fleiss_kappa_allocs(allocs.T)
    res["fkappa"] = fkappaval

    return res