import numpy as np
import sys 
import random
sys.path.append('/groups/astro/rsx187/massPynpz')
import glob
import massPynpz as mp
#import massPy as mp
import matplotlib.pyplot as plt
from shutil import copyfile
from pathlib import Path
import os
from multiprocessing import Pool
import json
#zs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
#zs = [0.018, 0.0195, 0.019, 0.02, 0.0215, 0.021, 0.022, 0.023, 0.024, 0.025,]
zs = [1]
sizes = [2048]

overwrite = False
global scalarjson
scalarjson = False

def vortandE(vx, vy, return_terms=False):
    dxux = mp.base_modules.numdiff.derivX(vx)
    dxuy = mp.base_modules.numdiff.derivX(vy)
    dyux = mp.base_modules.numdiff.derivY(vx)
    dyuy = mp.base_modules.numdiff.derivY(vy)
    vort = dxuy - dyux
    E = dxux + dyuy
    if return_terms:
        return vort, E, dxux, dyux, dxuy, dyuy
    else:    
        return vort, E


def strainrate_to_npy(pathnameout):

    name, out = pathnameout
    print(name, out)
    ar = mp.archive.loadarchive(name)
    nframe = ar.num_frames
    frameis = np.arange(nframe)#[::10]
    Path(out+'/strainrate').mkdir(parents=True, exist_ok=True)
    #if not os.path.exists(name):
    #    os.makedirs(name)
    copyfile(name+"/parameters.json", out+"/parameters.json")
    for i in frameis:
        print(i, 'of', frameis[-1])
        try: # some files just don't exist?
            frame = ar._read_frame(i)
        except FileNotFoundError:
            continue
        lx, ly = frame.LX, frame.LY

        try:
            vx, vy = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
        except AttributeError:
            vx, vy = mp.base_modules.flow.velocity(frame.ff, lx, ly)

        _, E = vortandE(vx, vy)
        if not scalarjson:
            #vx, vy = frame.vx.reshape(LX, LY), frame.vy.reshape(LX, LY)
            #vort, _ = vortandE(vx, vy)
            #vortbin = vort>0
            Ebin = E>0
            np.save(f'{out}/strainrate/frame{i}.npy', Ebin)# does this distinguish frames eveb
        elif scalarjson:
            #try:

            #vort = mp.base_modules.flow.strainrate(frame)
            with open(f'{out}/strainrate/frame{i}.json', 'w') as f:
                json.dump(E.tolist(), f)

for size in sizes:
    for zeta in zs:#    print(zeta)
        #datapath = f'/lustre/astro/rsx187/mmout/active_sample_forperp/qzk1k30.05_K30.05_qkbt0_z{z}_xi1_LX256_counter0/'
        #datapath = f'/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt{uq}_z0_xi1_LX256_counter0/'
        #datapath = f'/lustre/astro/rsx187/mmout/theta_sample/qzk1k30.05_K30.05_qkbt{theta}_z0_xi1_LX256_counter0/'
        datapath = f"/lustre/astro/kpr279/nematic_simulation/ns{size}*/"
        #datapath = f"/lustre/astro/rsx187/mmout/q_sample/"
        #datapath = f"/lustre/astro/rsx187/mmout/q1_sample/"

        


        destpath = f"/lustre/astro/rsx187/isolinescalingdata/strainratedata/simon_data/nematic_simulation{size}"
        #destpath = f"/lustre/astro/rsx187/isolinescalingdata/fullstrainratedata/simon_data/nematic_simulation{size}"
        #destpath = f"/lustre/astro/rsx187/isolinescalingdata/fullstrainratedata/q_sample"
        #destpath = f"/lustre/astro/rsx187/isolinescalingdata/fullstrainratedata/q1_sample"
        #destpath = f"/lustre/astro/rsx187/isolinescalingdata/fullstrainratedata/simon_data/nematic_simulation{size}"
        #destpath = f"/lustre/astro/rsx187/isolinescalingdata/strainratedata2/simon_data/nematic_simulation{size}"

        Path(destpath+"/strainrate").mkdir(parents=True, exist_ok=True)

        #copyfile(datapath+"parameters.json", destpath+"/parameters.json")

        
        names = glob.glob(f'{datapath}*/*')
        #names = glob.glob(f'{datapath}/*')
        #print(names)   

        names = [n for n in names if 'sdn' not in n]
        names = [n for n in names if '.dat' not in n]
        #names = [n for n in names if ]
        #names = [n for n in names if '1024' in n]
        names = [n for n in names if 'counter_0' in n]
        #names = [n for n in names if '0.04' in n]
        #names = [n for n in names if ('counter_2' not in n) &  ('counter_1' not in n) & ('counter_0' not in n)]

        #names = [n for n in names if f'zeta_{zeta}' in n]
        print(names)
        
        #sys.exit()
        outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names] # for simon data
        #outnames = [n.split('/')[-1] for n in names] #  for mmout data

        if not overwrite: # this needs fixing
        #if not overwrite:
        #for name, outname in zip(names, outnames):
            #if os.path.isdir(destpath+'/'+outname):
            #    names.remove(name)
            #    print(f'{name} already exists')
            done = os.listdir(destpath)
            #done = [d.split('/')[-1] for d in done]
            print('n names:', len(names), ', n done:', len(done))
            print(f'skipped: {len([name for name in names if name.split("/")[-1][12:] in done])}')
            #print(f'skipped: {len([name for name in names if name.split("/")[-1] in done])}')
            names = [name for name in names if name.split('/')[-1][12:] not in done] 
            #names = names[:1]
        print(names)
        print(f'number of analyses: {len(names)}\n')
        #sys.exit()

        outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names] # for simon data
        #outnames = [n.split('/')[-1] for n in names] #  for mmout data
        pathnames = [[n, destpath+'/'+outname] for n, outname in zip(names, outnames)]
        print(pathnames)
        print('n paths:', len(pathnames))
        #sys.exit()
        random.shuffle(pathnames)
        if len(pathnames)==0:
            continue
        #ntasks = min(10, len(pathnames))#
        ntasks = min(len(pathnames), int(os.environ['SLURM_CPUS_PER_TASK']))
        ntasks = max(1, ntasks)
        print('ntasks:', ntasks, type(ntasks))
        try:
            csize = int(len(names)/ntasks)+1
        except ZeroDivisionError:
            csize=0
        #sys.exit()
        with Pool(ntasks) as p:
            p.map(strainrate_to_npy, pathnames, chunksize=csize) 
