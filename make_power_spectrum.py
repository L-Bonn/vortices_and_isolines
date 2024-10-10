import numpy as np
import sys 

#sys.path.append('/groups/astro/rsx187/massPynpz')
import glob
#import massPynpz as mp
import massPy as mp
from multiprocessing import Pool
import pickle
import copy
import os
from random import shuffle



def vortchanger(vort, func=''):
    match func:
        case '':
            return vort
        case 'abs':
            return np.abs(vort)
        case 'enstrophy':
            return vort**2

def do_frame(path, framei, vort_func=''):

    ar = mp.archive.loadarchive(path)
    lx, ly = ar.LX, ar.LY

    try:
        frame = ar._read_frame(framei)
    except FileNotFoundError: # some frames may be missing
        print(f'skipped frame {framei} due to FileNotFoundError')
        return 
    try:
        vort = mp.base_modules.flow.vorticity(frame)
    except TypeError:
        vx, vy = mp.base_modules.flow.velocity(frame.ff, lx, ly)
        dxuy = mp.base_modules.numdiff.derivX(vy)
        dyux = mp.base_modules.numdiff.derivY(vx)
        vort = dxuy - dyux

    vps, x = mp.base_modules.correlation.power_spectrum(vortchanger(vort, vort_func))
    vps = np.sqrt(vps)

    return x, vps

def wrapper_do_frame(dic):
    return do_frame(**dic)
    
def do_archive(ntasks, path=None, out=None, vort_func=''):

    ar = mp.archive.loadarchive(path)
    nframe = ar.num_frames
    frameis = np.arange(nframe)

    savedic = ar.__dict__
    savedic['vort_func'] = vort_func

    ncpu = min(ntasks, nframe)

    rundics = []
    for framei in frameis:
        dic = {'path':path, 'framei':framei, 'vort_func':vort_func}
        rundics.append(dic)
    with Pool(ncpu) as p:
        res = p.map(wrapper_do_frame, rundics, chunksize=1)

    x_list = []
    vps_list = []

    try:
        for x, vps in res:
            x_list.append(x)
            vps_list.append(vps)
    except TypeError:
        print(f"{path} is empty!")

    savedic['x_list'] = x_list
    savedic['vps_list'] = vps_list

    with open(f'{out}.pickle', 'wb') as f:
        pickle.dump(savedic, f)
    print(f"done {path}")


if __name__ == "__main__":

    overwrite = False
    vort_func = '' # 'abs', 'enstrophy'

    names = ["/lustre/astro/kpr279/ns2048/output_test_zeta_0.04/output_test_zeta_0.04_counter_0"]
    names = ["/lustre/astro/kpr279/ns2048/output_test_zeta_0.030/output_test_zeta_0.030_counter_0",
    "/lustre/astro/kpr279/ns2048l/output_test_zeta_0.0215/output_test_zeta_0.0215_counter_0",
    "/lustre/astro/kpr279/ns2048l/output_test_zeta_0.020/output_test_zeta_0.020_counter_0",
    "/lustre/astro/kpr279/ns2048pd/output_test_zeta_0.010/output_test_zeta_0.010_counter_0",
    "/lustre/astro/kpr279/ns2048pd/output_test_zeta_0.018/output_test_zeta_0.018_counter_0",
    "/lustre/astro/kpr279/ns2048/output_test_zeta_0.04/output_test_zeta_0.04_counter_0"
    ]
    names = ["/lustre/astro/rsx187/mmout/u_sample/uzk1k30.05_K30.05_ukbt0.03_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/u_sample/uzk1k30.05_K30.05_ukbt0.09_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/u_sample/uzk1k30.05_K30.05_ukbt0.1_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/u_sample/uzk1k30.05_K30.05_ukbt0.2_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/u_sample/uzk1k30.05_K30.05_ukbt0.3_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/q_sample/qzk1k30.05_K30.05_qkbt0.005_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/q_sample/qzk1k30.05_K30.05_qkbt0.02_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/q_sample/qzk1k30.05_K30.05_qkbt0.04_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt0.005_z0_xi1_LX1024_counter2",
    "/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt0.011_z0_xi1_LX1024_counter1",
    "/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt0.015_z0_xi1_LX1024_counter0",
    "/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt0.03_z0_xi1_LX1024_counter0"

    ]
    outnames = ['_'.join(n.split('/')[-2:])+f'vortfunc_{vort_func}' for n in names] # what
    destpath = '/lustre/astro/rsx187/isolinescalingdata/power_spectra/'

    pathnames = [[n, destpath+'/'+outname] for n, outname in zip(copy.copy(names), outnames)]



    if not overwrite:
        for name, outname in pathnames:
            if os.path.isfile(outname+'.pickle'):
                names.remove(name)
                print(f'{name} already exists')
        #remake without existing
        outnames = ['_'.join(n.split('/')[-2:]) for n in names] # what
        pathnames = [[n, destpath+'/'+outname] for n, outname in zip(names, outnames)]
    print(names)

    ntasks = min(len(pathnames), int(os.environ['SLURM_CPUS_PER_TASK']))

    params = [{'path':path, 'out':destpath+'/'+outname,  'vort_func':vort_func,} 
        for path, outname in zip(names, outnames)]

    print(pathnames)
    #sys.exit()
    for par in params:
        do_archive(ntasks=int(os.environ['SLURM_CPUS_PER_TASK']), **par)