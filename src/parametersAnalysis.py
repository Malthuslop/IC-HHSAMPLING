#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[13]:


from hhsolver import *
from PIL import Image
from utils import *
import seaborn as sns
import pandas as pd
from hhsampling_params import *
import csv


# In[14]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[15]:


get_ipython().run_line_magic('autoreload', '2')


# In[16]:


file_path = '../data/Candidatos20230803_n.txt'
data = collect_data(file_path=file_path).to_numpy()


# In[17]:


lambda_n = 0#, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

nalloc = 100

sample_size_list = [60, 120, 180, 240] #4
time_limit_list = [120]#[5,10,20,30,60,120]#,240] #7
particles_number_list = [25,50,100]#,150,200] #5
c1_list = [0.0001, 0.1,1, 10, 100, 1000, 10000] #7
c2_list = [0.0001, 0.1, 1, 10, 100, 1000, 10000] #7
w_list = [0.0001, 0.1, 1, 10, 100, 1000, 10000] #7

# hhsamplingGA require transformed data
x = data @ inversecholcov(data)
solver = HHSolverPSO


# In[18]:


df = pd.DataFrame(columns = ["sample size", "time limit", "particles number", "cognitive param", "social param", "allocation number", "fitness score"])


# In[19]:


with open('PSO_dataframe001.csv', 'w', newline='') as arquivo_csv:
    escritor = csv.writer(arquivo_csv)
    escritor.writerow(["sample size", "time limit", "particles number", "cognitive param", "social param", "weight", "allocation number", "fitness score"])
    for sample_size in sample_size_list:
        for time_limit in time_limit_list:
            for particles_number in particles_number_list:
                for c1 in c1_list:
                    for c2 in c2_list:
                        for w in w_list:
                            # pre-generate noises used in each batch
                            ma = 2
                            noise = []
                            populations = []
                            for i in range(nalloc):
                                z = generate_noise(x.shape[0], ma)
                                # hhsamplingGA require transformed noise
                                zt = z @ inversecholcov(z)
                                noise.append(zt)
                                populations.append(generate_pop(particles_number, sample_size, x.shape[0]))


                            dist = hhsampling(x, lambda_n, nalloc, sample_size, noise, populations, solver, time_limit, particles_number, c1, c2, w)
                            for j in range(nalloc):
                                    #["sample size", "time limit", "particles number", "cognitive param", "social param", "allocation number", "fitness score"])
                                    escritor.writerow([sample_size, time_limit, particles_number, c1, c2, w, j, dist[j]])




