import numpy as np
import scipy.io as sio
import glob
from pathlib import Path
import json



destbin = '/lustre/astro/rsx187/isolinescalingdata/vorticitydata/olgadata/'
destfull = '/lustre/astro/rsx187/isolinescalingdata/fullvorticitydata/olgadata/'

names = glob.glob('/lustre/astro/rsx187/datasets/olgadata/Ex*/*.mat')

for name in names:
    print(name)
    foldername = name.split('/')[-2]  
    dat = sio.loadmat(name)
    vort = dat['vorticity'][:, 0]
    for ivort, arvort in enumerate(vort):
        #continue
        print(ivort)
        vortbin = arvort>0
        finaldestbin = destbin+foldername+'/vorticity/'
        finalfullbin = destfull+foldername+'/vorticity/'
        Path(finaldestbin).mkdir(parents=True, exist_ok=True)
        Path(finalfullbin).mkdir(parents=True, exist_ok=True)
        np.save(f'{finaldestbin}/frame{ivort}.npy', vortbin)
        with open(f'{finalfullbin}/frame{ivort}.json', 'w') as f:
            json.dump(arvort.tolist(), f)
