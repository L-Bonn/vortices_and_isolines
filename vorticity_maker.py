import numpy as np
import massPy as mp
import matplotlib.pyplot as plt
from shutil import copyfile
from pathlib import Path
import glob
import sys
import os
import json
from multiprocessing import Pool

#size = 2048
#datapath = f'/lustre/astro/rsx187/mmout/active_sample_n10/*'
#paths = ['/lustre/astro/rsx187/datasets/CompressibleAN']
#for zeta in zs:
#for path in paths:

def wrapper(argdic):
    return writevortpath(**argdic)

def writevortpath(path, folder, overwrite=False, scalarjson=False):
    print(path, path.split('/')[-1])
    name = path.split('/')[-1]
    print('starting:', name, flush=True)
    #sys.exit()
    path = path+'/'

    ar = mp.archive.loadarchive(path)

    full = ''
    if scalarjson:
        full = 'full'
    destpath = f"/lustre/astro/rsx187/isolinescalingdata/{full}vorticitydata/{folder}/{name}"
    Path(destpath+"/vorticity").mkdir(parents=True, exist_ok=True)
    copyfile(path+"parameters.json", destpath+"/parameters.json")

    if velocitynpy:
        veldestpath = f"/lustre/astro/rsx187/isolinescalingdata/velocitydata/{folder}/{name}"
        Path(veldestpath+"/velocity").mkdir(parents=True, exist_ok=True)
        copyfile(path+"parameters.json", veldestpath+"/parameters.json")


    nfiles = len(os.listdir(path))
    print(nfiles, flush=True)
    #if nfiles!=182: 
        #print("skipped due to wrong amount of files", flush=True)
        #continue
    nframe = ar.num_frames
    frameis = np.arange(nframe)#[::10]

    for i in frameis:
        print(i, 'of', frameis[-1], name)
        if not overwrite:
            if os.path.isfile(f'{destpath}/vorticity/frame{i}.npy'):
                continue
        try:
            frame = ar._read_frame(i)
        except FileNotFoundError:
            continue
        except json.decoder.JSONDecodeError:
            continue
        LX, LY = frame.LX, frame.LY
        try:
            vort = mp.base_modules.flow.vorticity(frame.ff, LX, LY)
            if velocitynpy:
                vx, vy = mp.base_modules.flow.velocity(frame.ff, LX, LY)
        except AttributeError:
            vx, vy = frame.ux.reshape(LX, LY), frame.uy.reshape(LX, LY)
            vort = mp.base_modules.numdiff.curl2D(vx, vy)
        if scalarjson:
            with open(f'{destpath}/vorticity/frame{i}.json', 'w') as f:
                json.dump(vort.tolist(), f)
        else:
            vortbin = vort>0
            np.save(f'{destpath}/vorticity/frame{i}.npy', vortbin)
        if velocitynpy:
            np.save(f'{veldestpath}/velocity/vx_frame{i}.npy', vx)
            np.save(f'{veldestpath}/velocity/vy_frame{i}.npy', vy)

overwrite = False
scalarjson = False
velocitynpy = False
print(f"overwrite: {overwrite}, scalarjson: {scalarjson}", flush=True)

#folder = "simon_xi_scan"
#folder = "simon_CC_scan"
#folder = "compressibleAN"
#folder = "ns512pd2"
#folder = "polar/L2048"
#folder = "polar/L2048_gam2"
#folder = "polar/test_polar_tt"
#folder = "theta_sample"
#folder = "q_sample"
#folder = "uq_scanq0.05"
#folder = "q_scan"
folder = "deffree_initaligned/"
#folder = "polar/testgam"
datapath = f'/lustre/astro/rsx187/mmout/{folder}/*'
#datapath = f'/lustre/astro/rsx187/{folder}/*'
#datapath = f'/lustre/astro/kpr279/{folder}/*/*'



paths = glob.glob(datapath)
#paths = [p for p in paths if "LX1024" not in p]
#paths = [p for p in paths if "counter0" in p]
#paths = [p for p in paths if ('0.1' in p) or ('0.2' in p)]
print(paths, flush=True)
print(f'n files: {len(paths)}', flush=True)


ntasks = min(len(paths), np.floor(int(os.environ['SLURM_CPUS_PER_TASK'])).astype(int))
#ntasks = 5

argdics = [{'path': path, 'folder': folder, 'overwrite':overwrite, 'scalarjson':scalarjson} for path in paths]

##print(argdics)
#sys.exit()
with Pool(ntasks) as p:
    p.map(wrapper, argdics)
