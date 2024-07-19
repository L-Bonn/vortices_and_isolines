import numpy as np
import sys 
sys.path.append('/groups/astro/rsx187/mass/')
import av_defs_py as adp
sys.path.append('/groups/astro/rsx187/massPynpz')
import glob
import massPynpz as mp
import matplotlib.pyplot as plt
from shutil import copyfile
from pathlib import Path
import os
from multiprocessing import Pool

#zs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
zs = [0.030]
zs = [0.07, 0.06, 0.05, 0.04, 0.032, 0.09, 0.03, 0.15, 0.034, 0.1, 0.028, 0.026, 0.025, 0.024, 0.023, 0.0215, 0.022, 0.018, 0.0195, 0.02, 0.021, 0.019, 0.016, 0.08, 0.014, 0.012, 0.01, 0.007, 0.005, 0.002]

uqs = [0.005, 0.01, 0.02, 0.04]
thetas = [0.001, 0.005, 0.01, 0.013, 0.017]
sizes = [256, 512, 1024, 2048, 4096]
overwrite = False

def pressure_to_npy(pathnameout):

    name, out = pathnameout
    print(name, out)
    ar = mp.archive.loadarchive(name)
    nframe = ar.num_frames
    frameis = np.arange(nframe)#[::10]
    Path(out+'/pressure').mkdir(parents=True, exist_ok=True)
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
        Qxx_dat, Qyx_dat = frame.QQxx.reshape(lx, ly), frame.QQyx.reshape(lx, ly)
        vx_dat, vy_dat = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
        sigmaXX, sigmaXY, sigmaYX, sigmaYY = adp.calcstress(frame, Qxx_dat, Qyx_dat, vx_dat, vy_dat)
        pressure = 0.5*(sigmaXX+sigmaYY)
        pressure0 = pressure>0
        np.save(f'{out}/pressure/frame{i}.npy', pressure0)# does this distinguish frames eveb
        

for size in sizes:
    zeta = zs[0]
    print(zeta)
    #datapath = f'/lustre/astro/rsx187/mmout/active_sample_forperp/qzk1k30.05_K30.05_qkbt0_z{z}_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt{uq}_z0_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/rsx187/mmout/theta_sample/qzk1k30.05_K30.05_qkbt{theta}_z0_xi1_LX256_counter0/'
    datapath = f"/lustre/astro/kpr279/ns{size}*/"

    


    destpath = f"/groups/astro/rsx187/isolinescaling/pressuredata/simon_data/nematic_simulation{size}"
    Path(destpath+"/pressure").mkdir(parents=True, exist_ok=True)

    #copyfile(datapath+"parameters.json", destpath+"/parameters.json")


    names = glob.glob(f'{datapath}*/*')
    #print(names)
    names = [n for n in names if '.dat' not in n]
    #names = [n for n in names if ('counter_0' or 'counter_1') in n]
    #names = [n for n in names if ('counter_2' not in n) &  ('counter_1' not in n) & ('counter_0' not in n)]

    names = [n for n in names if f'zeta_{zeta}' in n]
    print(names)
    
    #sys.exit()
    if not overwrite: # not sure this works?

        done = os.listdir(destpath)
        print('n names:', len(names), ', n done:', len(done))
        print(f'skipped: {len([name for name in names if name+"dur.pickle" in done])}')
        names = [name for name in names if name+"dur.pickle" not in done] 
        #names = names[:1]
    print(names)
    print(f'number of analyses: {len(names)}\n')
    #sys.exit()

    outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names] # what
    pathnames = [[n, destpath+'/'+outname] for n, outname in zip(names, outnames)]
    print(pathnames)
    #sys.exit()
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

    with Pool(ntasks) as p:
        p.map(pressure_to_npy, pathnames, chunksize=csize) 
