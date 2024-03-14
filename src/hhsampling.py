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













if __name__ == '__main__': 

    file_path = '../data/Candidatos20230803_n.txt'
    data = collect_data(file_path=file_path).to_numpy()

    lambdas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    nalloc = 300

    sample_size = [60, 120, 180, 240] 
    
    fileNames = []

    # hhsamplingGA require transformed data
    x = data @ inversecholcov(data)#X*

    for sample in sample_size:
        # pre-generate noises used in each batch
        ma = 2
        noise = []
        populations = []
        for i in range(nalloc):
            z = generate_noise(x.shape[0], ma)
            # hhsamplingGA require transformed noise
            zt = z @ inversecholcov(z)
            noise.append(zt)
            populations.append(generate_pop(50, sample, x.shape[0]))

        stats_aux = {"distq95": [], "fkappa": []}

        for i in range(len(lambdas)):
            res = hhsampling(x, lambdas[i], nalloc, sample, noise, populations, HHSolverPSO)
            for k in stats_aux:
                stats_aux[k].append(res[k])

        stats = pd.DataFrame(stats_aux)

        # br = '-----------------------------'
        # print(br)
        # print(stats)
        # print(br)


        fig, axs = plt.subplots(2)
        fig.suptitle('Allocations stats for haphazard with GA')

        axs[0].plot(lambdas, stats["distq95"])
        axs[0].set_title("Mahalanobis 95th quantile")

        axs[1].plot(lambdas, stats["fkappa"])
        axs[1].set_title("Fleiss Kappa")

        fig.tight_layout()

        figname = str(sample) + ".png"
        fileNames.append("../output/img" + figname)
        plt.savefig("../output/img" + figname, format="png", bbox_inches="tight")

    images = []
    for f in fileNames:
        # Image.open(f)
        png = Image.open(f)
        png.load()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        images.append(background)

    pdf_path = "../output/Output.pdf"
    images[0].save(
      pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
    )
    doc = aw.Document()
    builder = aw.DocumentBuilder(doc)

    # for fileName in fileNames:
    #     builder.insert_image(fileName)
    #     # Insert a paragraph break to avoid overlapping images.
    #     builder.writeln()

    # doc.save("../output/Output.pdf")
    # plt.show()