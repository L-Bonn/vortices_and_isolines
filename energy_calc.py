import numpy as np
import massPy as mp
import matplotlib.pyplot as plt
from shutil import copyfile
from pathlib import Path
import glob
import sys
import os
import json
#size = 2048
#datapath = f'/lustre/astro/rsx187/mmout/active_sample_n10/*'
overwrite = False
folder = "simon_xi_scan"
#folder = "compressibleAN"
datapath = f'/lustre/astro/rsx187/mmout/{folder}/*'
#datapath = f'/lustre/astro/rsx187/{folder}/*'

paths = glob.glob(datapath)
print(paths)
#paths = ['/lustre/astro/rsx187/datasets/CompressibleAN']
#for zeta in zs:
for path in paths:
    print(path, path.split('/')[-1])
    name = path.split('/')[-1]
    print(name)
    #sys.exit()
    path = path+'/'
    #datapath = f'/lustre/astro/rsx187/mmout/active_sample_forperp/qzk1k30.05_K30.05_qkbt0_z{z}_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt{uq}_z0_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/rsx187/mmout/theta_sample/qzk1k30.05_K30.05_qkbt{theta}_z0_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/jayeeta/aniso/datas/size_{size}/zeta-{zeta}-out/'
    #datapath = f'/lustre/astro/bhan/nem2048/'
    ar = mp.archive.loadarchive(path)

    #sdestpath = f"/lustre/astro/rsx187/isolinescalingdata/vorticitydata/active_sample_n10/active_sample_n10{ar.LX}_{ar.zeta}"
    destpath = f"/lustre/astro/rsx187/isolinescalingdata/vorticitydata/{folder}/{name}"

    #destpath = f"/groups/astro/rsx187/isolinescaling/vorticitydata/benjamin2048_zeta0.1"

    Path(destpath+"/vorticity").mkdir(parents=True, exist_ok=True)

    copyfile(path+"parameters.json", destpath+"/parameters.json")


    nframe = ar.num_frames
    frameis = np.arange(nframe)#[::10]

    for i in frameis:
        print(i, 'of', frameis[-1])
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
            vortbin = mp.base_modules.flow.vorticity(frame.ff, LX, LY)>0
        except AttributeError:
            vx, vy = frame.ux.reshape(LX, LY), frame.uy.reshape(LX, LY)
            vortbin = mp.base_modules.numdiff.curl2D(vx, vy)>0
        np.save(f'{destpath}/vorticity/frame{i}.npy', vortbin)