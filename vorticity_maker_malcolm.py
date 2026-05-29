import numpy as np
import scipy.ndimage as scn
import glob
from pathlib import Path
import json
import os
from multiprocessing import Pool
import sys

def get_psi(dataPath,step, nPts=2048):
    thetaArray = []
    ii = 0
    holder = []
    with open(dataPath+f"/psim/{step}.txt") as f:
        for line in f:
            if ii < nPts:
                holder.append(float(line.split()[0]))
                ii += 1
            if ii == nPts:
                thetaArray.append(holder)
                holder = []
                ii = 0
    return(np.array(thetaArray))

def get_theta(dataPath,step, nPts=2048):
    thetaArray = []
    ii = 0
    holder = []
    with open(dataPath+f"/thetam/{step}.txt") as f:
        for line in f:
            if ii < nPts:
                holder.append(float(line.split()[0]))
                ii += 1
            if ii == nPts:
                thetaArray.append(holder)
                holder = []
                ii = 0
    return(np.array(thetaArray))

def get_vort(psi):
    w = -scn.laplace(psi)
    return w

def get_and_write_vort(dataBaseDir, nPts, dry_run, finaldestbin, finalfullbin):
    print(dataBaseDir)
    nsteps = len(glob.glob(dataBaseDir+'/psim/*'))
    print(nsteps)
    supp = '_'.join(dataBaseDir.split('/')[-2:])
    finaldestbin = f'{finaldestbin}/{supp}/vorticity/'
    finalfullbin = f'{finalfullbin}/{supp}/vorticity/'
    print(finaldestbin)
    if dry_run: return
    Path(finaldestbin).mkdir(parents=True, exist_ok=True)
    Path(finalfullbin).mkdir(parents=True, exist_ok=True)

    for step in range(nsteps):
        psi = get_psi(dataBaseDir,step, nPts=nPts)
        vort = get_vort(psi)
        vortbin = vort>0
        np.save(f'{finaldestbin}/frame{step}.npy', vortbin)
        with open(f'{finalfullbin}/frame{step}.json', 'w') as f:
            json.dump(vort.tolist(), f)

def wrapper(argdic):
    return get_and_write_vort(**argdic)

### SETUP ###
nPts = 1024
dry_run = False

### DATA DESTINATION ###
destpath = f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/hillebrand_long/'
destfull = f'/lustre/astro/rsx187/isolinescalingdata/fullvorticitydata/hillebrand_long/'

### DATA ORIGIN ###
path = "/lustre/astro/rsx187/datasets/hillebrand/erdadata/1024-movies/*/*"
paths = glob.glob(path)
paths = [p for p in paths if os.path.isdir(p)]
print(paths)
ntasks = min(len(paths), np.floor(int(os.environ['SLURM_CPUS_PER_TASK'])).astype(int))
print(f'ncpus = {ntasks}')


#answer = input("continue? [y/n]").lower()
#if answer != 'y':
#    sys.exit()

### EXECUTION ###
#for dataBaseDir in paths:
#    get_and_write_vort(dataBaseDir)

argdics = [{'dataBaseDir': dataBaseDir,
 'nPts': nPts,
 'dry_run': dry_run,
 'finaldestbin': destpath,
 'finalfullbin': destfull} for dataBaseDir in paths]


with Pool(ntasks) as p:
    p.map(wrapper, argdics)
print('finished :)')