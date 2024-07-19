import numpy as np
import sys 

sys.path.append('/groups/astro/rsx187/massPynpz')
import glob
import massPynpz as mp
import matplotlib.pyplot as plt
from shutil import copyfile
from pathlib import Path
import os
from multiprocessing import Pool
import json
#from analyse_vorticity_mpperarchive import vortandE
#zs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
zs = [0.07, 0.06, 0.05, 0.04, 0.032, 0.09, 0.03, 0.15, 0.034, 0.1, 0.028, 0.026, 0.025, 0.024, 0.023, 0.0215, 0.022, 0.018, 0.0195, 0.02, 0.021, 0.019, 0.016, 0.08, 0.014, 0.012, 0.01, 0.007, 0.005, 0.002]
zs = [0.018, 0.0195, 0.019, 0.02, 0.0215, 0.021, 0.022, 0.023, 0.024, 0.025,]
zs = [1]
uqs = [0.005, 0.01, 0.02, 0.04]
thetas = [0.001, 0.005, 0.01, 0.013, 0.017]
sizes = [256, 512, 1024, 2048, 4096]
sizes = [512]
overwrite = False
global scalarjson
scalarjson = True

def vorticity_to_npy(pathnameout):

    name, out = pathnameout
    print(name, out)
    ar = mp.archive.loadarchive(name)
    nframe = ar.num_frames
    frameis = np.arange(nframe)#[::10]
    Path(out+'/vorticity').mkdir(parents=True, exist_ok=True)
    #if not os.path.exists(name):
    #    os.makedirs(name)
    copyfile(name+"/parameters.json", out+"/parameters.json")
    for i in frameis:
        print(i, 'of', frameis[-1])
        try: # some files just don't exist?
            frame = ar._read_frame(i)
        except FileNotFoundError:
            continue
        LX, LY = frame.LX, frame.LY
        if not scalarjson:
            #vx, vy = frame.vx.reshape(LX, LY), frame.vy.reshape(LX, LY)
            #vort, _ = vortandE(vx, vy)
            #vortbin = vort>0
            vortbin = mp.base_modules.flow.vorticity(frame)>0
            np.save(f'{out}/vorticity/frame{i}.npy', vortbin)# does this distinguish frames eveb
        elif scalarjson:
            vort = mp.base_modules.flow.vorticity(frame)
            with open(f'{out}/vorticity/frame{i}.json', 'w') as f:
                json.dump(vort.tolist(), f)

for size in sizes:
    for zeta in zs:#    print(zeta)
        #datapath = f'/lustre/astro/rsx187/mmout/active_sample_forperp/qzk1k30.05_K30.05_qkbt0_z{z}_xi1_LX256_counter0/'
        #datapath = f'/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt{uq}_z0_xi1_LX256_counter0/'
        #datapath = f'/lustre/astro/rsx187/mmout/theta_sample/qzk1k30.05_K30.05_qkbt{theta}_z0_xi1_LX256_counter0/'
        datapath = f"/lustre/astro/kpr279/ns{size}*/"

        


        destpath = f"/lustre/astro/rsx187/isolinescalingdata/vorticitydata/simon_data/nematic_simulation{size}"
        destpath = f"/lustre/astro/rsx187/isolinescalingdata/fullvorticitydata/simon_data/nematic_simulation{size}"
        #destpath = f"/lustre/astro/rsx187/isolinescalingdata/vorticitydata2/simon_data/nematic_simulation{size}"

        Path(destpath+"/vorticity").mkdir(parents=True, exist_ok=True)

        #copyfile(datapath+"parameters.json", destpath+"/parameters.json")

        
        names = glob.glob(f'{datapath}*/*')
        #print(names)

        names = [n for n in names if 'sdn' not in n]
        names = [n for n in names if '.dat' not in n]
        #names = [n for n in names if 'counter_6' in n]
        #names = [n for n in names if '0.03' in n]
        #names = [n for n in names if ('counter_2' not in n) &  ('counter_1' not in n) & ('counter_0' not in n)]

        #names = [n for n in names if f'zeta_{zeta}' in n]
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
        #sys.exit()
        with Pool(ntasks) as p:
            p.map(vorticity_to_npy, pathnames, chunksize=csize) 
