import numpy as np
import glob
import os
from natsort import natsorted
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool

ntasks = 12
paths = glob.glob('/lustre/astro/rsx187/isolinescalingdata/fullvorticitydata/simon_data/nematic_simulation2048/*counter_0/defects/')
paths = natsorted(paths)

def get_ndefs(path):
    load = np.loadtxt(path, delimiter=',')
    if load.size==0:
        ndefs = 0
    else:
        charge = load.T[2]
        ndefs = np.sum(np.abs(charge)==0.5)
    return ndefs


d_dic = {}
for path in paths:
    print(path)
    dfiles = natsorted(os.listdir(path))
    dlist = []
    #for df in dfiles:
        # load = np.loadtxt(path+df, delimiter=',')
        # if load.size==0:
        #     ndefs = 0
        # else:
        #     charge = load.T[2]
        #     ndefs = np.sum(np.abs(charge)==0.5)
        # dlist.append(ndefs)


    with Pool(ntasks) as p:
        dlist = p.map(get_ndefs, [path+df for df in dfiles]) 
        
    dmean = np.mean(dlist)
    dstd = np.std(dlist)
    z = float(path.split('/')[-3].split('_')[1])
    d_dic[z] = [dmean, dstd]


zs = natsorted(list(d_dic.keys()))
arr = np.zeros((len(zs), 3))
for i, z in enumerate(zs):
    dm, ds = d_dic[z]
    arr[i] = [z, dm, ds]

np.save('defect_list_simon_z_mean_std.npy', arr)