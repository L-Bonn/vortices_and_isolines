import sys
import numpy as np
import os
import json
import pickle
from multiprocessing import Pool
from scipy import ndimage
import massPy as mp
import glob
from pathlib import Path
#sys.path.append('/groups/astro/rsx187/massPynpz')
#import massPynpz as mp
import massPy as mp
#sys.path.append('../matlabdefects/')

#from archive import archive 
#from plot import lyotropic as ly

def get_F(Qxx, Qyx, C=1, K=0.05, cut=0):

    if cut:
        Qxx = Qxx[cut:-cut, cut:-cut]
        Qyx = Qyx[cut:-cut, cut:-cut]

    Cterm =  C*(1-(Qxx**2+Qyx**2))**2 # C (1-(1/2)Q_ij Q_ji)^2


    dyQxx = mp.base_modules.numdiff.derivY(Qxx)
    dxQxx = mp.base_modules.numdiff.derivX(Qxx)
    dyQyx = mp.base_modules.numdiff.derivY(Qyx)
    dxQyx = mp.base_modules.numdiff.derivX(Qyx)
    dyQxy = dyQyx
    dxQxy = dxQyx
    dyQyy = -dyQxx
    dxQyy = -dxQxx

    Kterm = K/2*(dxQxx*dxQxx + 
                dxQxy*dxQxy + 
                dxQxx*dyQyx + 
                dxQxy*dyQyy + 
                dyQyx*dxQxx + 
                dyQyy*dxQxy + 
                dyQyx*dyQyx + 
                dyQyy*dyQyy)
    #F = np.mean(Cterm + Kterm)#
    return np.mean(Cterm), np.mean(Kterm)

def get_Ekin(vx, vy):
    return np.mean(vx**2+vy**2)

def arch_to_F(nameoutname):
    path, outname = nameoutname
    ar = mp.archive.loadarchive(path)
    ni = ar.num_frames

    FCs = []
    FKs = []
    Ekins = []
    #print(ar.parameters)
    #print(ar.__dict___)
    C = ar.CC
    K = ar.LL
    #assert K1==ar.K3

    for iframe in range(ni):
        try:
            frame = ar._read_frame(iframe)
            LX, LY = frame.LX, frame.LY
            Qxx_dat = frame.QQxx.reshape(LX, LY)
            Qyx_dat = frame.QQyx.reshape(LX, LY)
            try:
                vx, vy = mp.base_modules.flow.velocity(frame.ff, LX, LY)
            except AttributeError:
                vx, vy = frame.vx, frame.vy
            Cterm, Kterm = get_F(Qxx_dat, Qyx_dat, C=C, K=K, cut=12)
            Ekin = get_Ekin(vx, vy)
        except FileNotFoundError:
            print(f'filenotfound for {path}, frame {iframe}')
            Cterm, Kterm, Ekin = np.nan, np.nan, np.nan
        

        FCs.append(Cterm)
        FKs.append(Kterm)
        Ekins.append(Ekin)

    dic = ar.__dict__

    dic['FCs'] = FCs
    dic['FKs'] = FKs
    dic['Ekins'] = Ekins
    with open(outname+'.pickle', 'wb') as file:
        pickle.dump(dic, file)


    print(path)

if __name__ == '__main__':

    overwrite = 0
    size = 512
    #dirpath ='/lustre/astro/rsx187/ptout/dry_QkBT_phasetwist025/'
    #dirpath ='/lustre/astro/rsx187/ptout/wet_z_phasetwist025/'
    #dirpath ='/lustre/astro/rsx187/ptout/dry_QkBT_phasetwist025_CC0/'
    #dirpath ='/lustre/astro/rsx187/ptout/dry_QkBT_phasetwist05/'
    #dirpath ='/lustre/astro/rsx187/ptout/dry_z_phasetwist025/'
    dirpath = f'/lustre/astro/kpr279/ns{size}pd2/*/*'

    fnames = glob.glob(dirpath)
    #outpath = '/groups/astro/rsx187/mass/dry_QkBT_phasetwist025cut/'
    #outpath = '/groups/astro/rsx187/mass/wet_z_phasetwist025/'   
    #outpath = '/groups/astro/rsx187/mass/wet_z_phasetwist025cut/'
    #outpath = '/groups/astro/rsx187/mass/dry_QkBT_phasetwist025_CC0/'
    outpath = f'/lustre/astro/rsx187/isolinescalingdata/energies/simon_{size}_2/'
    #outpath = '/groups/astro/rsx187/mass/dry_QkBT_phasetwist05/'
    #outpath = '/groups/astro/rsx187/mass/dry_z_phasetwist025/'
    #outpath = '/groups/astro/rsx187/mass/wet_QkBT_phasetwist025/'
    #outpath = '/groups/astro/rsx187/mass/wet_QkBTukBT_phasetwist025/'
    #outpath = '/groups/astro/rsx187/mass/wet_z_phasetwist025/'
    #outpath = '/groups/astro/rsx187/mass/wet_theta_phasetwist025/'

    Path(outpath).mkdir(parents=True, exist_ok=True)

    #fnames = [name for name in fnames if 'warmup' not in name]
    #fnames = [name for name in fnames if 'LX2000_' in name]
    #fnames = [name for name in fnames if 'LX500' not in name]
    #fnames = [name for name in fnames if 10<=int(''.join(filter(str.isdigit, name.rstrip('.mat').split('_')[-1][7:])))]
    #fnames = [name for name in fnames if ('z0.003' or 'z0.002') in name]
    #fnames = [name for name in fnames if 'counter0' in name]

    print(fnames)
    outnames = [n.split('/')[-1] for n in fnames]

    #sys.exit()
    if not overwrite:
        done = os.listdir(outpath)
        print(f'skipped: {len([name for name in fnames if name.split("/")[-1] in done])}')
        fnames = [name for name in fnames if name.split('/')[-1] not in done]
    outnames = [outpath+n.split('/')[-1] for n in fnames]
    print(fnames)
    print(f'number of analyses: {len(fnames)}\n')
    #sys.exit()
    ntasks = int(os.environ['SLURM_CPUS_PER_TASK'])
    #ntasks = 40
    print('ntasks:', ntasks, type(ntasks))
    csize = int(len(fnames)/ntasks)+1
    #fnames = [fnames[0]]
    print(outnames)
    
    with Pool(ntasks) as pool:
        pool.map(arch_to_F, zip(fnames, outnames), chunksize=csize)

