import numpy as np
import scipy.io as sio
import glob
from pathlib import Path
import json



destbin = '/lustre/astro/rsx187/isolinescalingdata/epsilondata/olgadata/'
destfull = '/lustre/astro/rsx187/isolinescalingdata/fullepsilondata/olgadata/'

names = glob.glob('/lustre/astro/rsx187/datasets/olgadata/Ex*/*.mat')

for name in names:
    print(name)
    foldername = name.split('/')[-2].replace(" ", "")
    dat = sio.loadmat(name)
    vort = dat['vorticity'][:, 0]
    vx = dat['u_smoothed'][:, 0]
    vy = dat['v_smoothed'][:, 0]
    for ivort, arvort in enumerate(vort):

        print(ivort)
        vxi = vx[ivort]
        vyi = vy[ivort]
        epsilon = arvort*(vxi**2+vyi**2)
        epsilonbin = epsilon>0
            
        finaldestbin = destbin+foldername+'/epsilon/'
        finalfullbin = destfull+foldername+'/epsilon/'
        Path(finaldestbin).mkdir(parents=True, exist_ok=True)
        Path(finalfullbin).mkdir(parents=True, exist_ok=True)

        np.save(f'{finaldestbin}/frame{ivort}.npy', epsilonbin)
        with open(f'{finalfullbin}/frame{ivort}.json', 'w') as f:
            json.dump(epsilon.tolist(), f)

