import sys
import numpy as np
import os
import json
import pickle
import copy
from multiprocessing import Pool
from scipy import ndimage
import massPy as mp
import glob
from pathlib import Path
sys.path.append('/groups/astro/rsx187/massPynpz')
import massPynpz as mp
#import massPy as mp
#sys.path.append('../matlabdefects/')



def get_viscous_dissipation(vx, vy, eta):
    dxvx = mp.base_modules.numdiff.derivX(vx)
    dyvx = mp.base_modules.numdiff.derivY(vx)
    dxvy = mp.base_modules.numdiff.derivX(vy)
    dyvy = mp.base_modules.numdiff.derivY(vy)
    
    #divergence = dxvx + dyvy
    Exx = dxvx
    Exy = 0.5*(dxvy+dyvx)
    Eyx = Exy
    Eyy = dyvy
    

    D = eta * np.sqrt(Exx**2 + 2*Exy**2 + Eyy**2)

    return D

def get_nematic_dissipation(frame):
    """"
    i want to return Gamma H:H - a lot of calculation? Carenza 2020 EPL
    """

    try:
        L1, L2, L3 = frame.L1, frame.L2, frame.L3
    except AttributeError:
        #K = frame.K
        L1 = frame.LL
        L2 = 0
        L3 = 0
    Gamma = frame.Gamma
    CC = frame.CC
    LX, LY = frame.LX, frame.LY

    Qxx, Qyx = frame.QQxx.reshape(LX, LY), frame.QQyx.reshape(LX, LY)

    #I'm copying from nematic.cpp
    dxQxx = mp.base_modules.numdiff.derivX(Qxx)
    dyQxx = mp.base_modules.numdiff.derivY(Qxx)
    dxQyx = mp.base_modules.numdiff.derivX(Qyx)
    dyQyx = mp.base_modules.numdiff.derivY(Qyx)

    dxdxQxx = mp.base_modules.numdiff.derivX(dxQxx)
    dxdyQxx = mp.base_modules.numdiff.derivX(dyQxx)
    dydyQxx = mp.base_modules.numdiff.derivY(dyQxx)

    dxdxQyx = mp.base_modules.numdiff.derivX(dxQyx)
    dxdyQyx = mp.base_modules.numdiff.derivX(dyQyx)
    dydyQyx = mp.base_modules.numdiff.derivY(dyQyx)

    del2Qxx = dxdxQxx + dydyQxx
    del2Qyx = dxdxQyx + dydyQyx

    dxQxx2 = dxQxx*dxQxx
    dyQxx2 = dyQxx*dyQxx
    dxQyx2 = dxQyx*dxQyx
    dyQyx2 = dyQyx*dyQyx
    term = 1. - Qxx*Qxx - Qyx*Qyx

    Hxx = (CC*term*Qxx +(L1+0.5*L2)*del2Qxx
            +L3*(Qxx*(dxdxQxx-dydyQxx) + 2*Qyx*dxdyQxx + dxQxx*dyQyx + dyQxx*dxQyx
            + 0.5*(dxQxx2-dyQxx2-dxQyx2+dyQyx2)))
    Hyx = (CC*term*Qyx +(L1+0.5*L2)*del2Qyx
            +L3*(Qxx*(dxdxQyx-dydyQyx) + 2*Qyx*dxdyQyx +
            dxQxx*(-dyQxx+dxQyx) + dyQyx*(-dyQxx+dxQyx)))
    
    return Gamma * (2*Hxx**2 + 2*Hyx**2)

def get_polar_dissipation(frame):
    """"
    i want to return h dot h/gamma - a lot of calculation? Carenza 2020 EPL
    """

    CC = frame.CC
    Kp = frame.Kp
    Kn = frame.Kn
    gamma = frame.gamma
    LX, LY = frame.LX, frame.LY
    px, py = frame.Px.reshape(LX, LY), frame.Py.reshape(LX, LY)


    ps = px*px+py*py
    p4 = px*px*px*px+py*py*py*py+2*px*px*py*py

    dxpx = mp.base_modules.numdiff.derivX(px)
    dypx = mp.base_modules.numdiff.derivY(px)
    dxpy = mp.base_modules.numdiff.derivX(py)
    dypy = mp.base_modules.numdiff.derivY(py)
    laplacian_px = mp.base_modules.numdiff.derivX(dxpx) + mp.base_modules.numdiff.derivY(dypx)
    laplacian_py = mp.base_modules.numdiff.derivX(dxpy) + mp.base_modules.numdiff.derivY(dypy)

    term = 1 - p4
    hx = (CC*(1-ps)*px + Kp*laplacian_px + Kn*( 1.*ps*laplacian_px
        +2.*px*(dxpx*dxpx+dypx*dypx-dxpy*dxpy-dypy*dypy)+4.*py*(dxpx*dxpy+dypx*dypy)))
    hy = (CC*(1-ps)*py + Kp*laplacian_py + Kn*( 1.*ps*laplacian_py
            +2.*py*(dxpy*dxpy+dypy*dypy-dxpx*dxpx-dypx*dypx) +4.*px*(dxpx*dxpy+dypx*dypy)))

    return 1/gamma * (hx**2 + hy**2)

def arch_to_D(path, outname):
    #path, outname = nameoutname
    if not os.listdir(path): 
        return

    ar = mp.archive.loadarchive(path)
    ni = ar.num_frames

    Dvs = []
    Dvmeans = []

    DHs = []
    DHmeans = []

    if ar.model_name == 'polar':
        tau = ar.tauPol
    else:
        tau = ar.tau

        
    eta = (2*tau-1)/6 * ar.rho # Carenza 2019 EPJE https://doi.org/10.1140/epje/i2019-11843-6 (B.29)

    for iframe in range(ni):
        #print(iframe, flush=True)
        try:
            frame = ar._read_frame(iframe)
            LX, LY = frame.LX, frame.LY
            try:
                if ar.model_name == 'polar':
                    vx, vy = mp.base_modules.flow.velocity(frame)
                else:
                    vx, vy = mp.base_modules.flow.velocity(frame.ff, LX, LY)
            except AttributeError:
                vx, vy = frame.vx.reshape(LX, LY), frame.vy.reshape(LX, LY)
            except TypeError:
                vx, vy = frame.vx.reshape(LX, LY), frame.vy.reshape(LX, LY)
                
            Dv = get_viscous_dissipation(vx, vy, eta=eta)
            if ar.model_name == 'nematic': 
                DH = get_nematic_dissipation(frame)
            else:
                DH = get_polar_dissipation(frame)

        except FileNotFoundError:
            print(f'filenotfound for {path}, frame {iframe}')
            Dv = np.nan
            DH = np.nan
        

        Dvs.append(Dv)
        Dvmeans.append(np.mean(Dv))
        DHs.append(DH)
        DHmeans.append(np.mean(DH))

    dic = copy.deepcopy(ar.__dict__)
    dic['Dv'] = Dvs
    dic['DH'] = DHs
    with open(outname+'.pickle', 'wb') as file:
        pickle.dump(dic, file)

    meandic = copy.deepcopy(ar.__dict__)
    meandic['Dvmean'] = Dvmeans
    meandic['DHmean'] = DHmeans
    with open(outname+'means.pickle', 'wb') as file:
        pickle.dump(meandic, file)


    print(path)

def wrapper(argdic):
    return arch_to_D(**argdic)


if __name__ == '__main__':

    overwrite = 0
    size = 1024

    #dirpath = f'/lustre/astro/kpr279/ns{size}*/*/*'
    dirpath = f'/lustre/astro/rsx187/polar/L2048_gam2/*'

    fnames = glob.glob(dirpath)
    #outpath = f'/lustre/astro/rsx187/isolinescalingdata/dissipationboth/simon_{size}/'
    outpath = f'/lustre/astro/rsx187/isolinescalingdata/dissipationboth/polar_L2048_gam2/'

    Path(outpath).mkdir(parents=True, exist_ok=True)

    #fnames = [name for name in fnames if 'warmup' not in name]
    #fnames = [name for name in fnames if 'LX2000_' in name]
    #fnames = [name for name in fnames if 'LX500' not in name]
    #fnames = [name for name in fnames if 10<=int(''.join(filter(str.isdigit, name.rstrip('.mat').split('_')[-1][7:])))]
    #fnames = [name for name in fnames if ('z0.003' or 'z0.002') in name]
    #fnames = [name for name in fnames if 'counter_0' in name]
    #fnames = fnames[:3]

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

    argdics = [{'path':n, 'outname':outname} for n, outname in zip(fnames, outnames)]

    with Pool(ntasks) as pool:
        pool.map(wrapper, argdics, chunksize=csize)

