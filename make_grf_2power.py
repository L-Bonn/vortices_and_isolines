import numpy as np
import matplotlib.pyplot as plt
import os
import json
import massPy as mp
import scipy.stats as scs
import sys
from pathlib import Path
import copy
import sys
from multiprocessing import Pool


def Ak_double_power_scale(a, b, s):
    def retfunc(k):
        return 1/((s/k)**-a+(s/k)**-b)
    return retfunc


#https://nkern.github.io/posts/2024/grfs_and_ffts/
def sim_grf(box, dL, pk_func, power=False):
    """
    box : 2D ndarray
    dL : pixel length
    """
    N = len(box)
    
    # k spectrum of each axis
    k = np.fft.fftshift(np.fft.fftfreq(N, dL).astype(np.float32) * 2 * np.pi)

    # get 2D grid
    KX, KY = np.meshgrid(k, k, indexing='ij')

    # get K magnitude
    KMAG = (np.sqrt(KX**2 + KY**2))

    # evaluate the pk function
    pk = pk_func(KMAG)
    if power: # need to avoid /0
        pk[int(N/2), int(N/2)] = 0
    # normalize by the noise equivalent bandwidth scaling (see below for explanation)
    pk /= np.sqrt(dL**2)

    # get the noise equivalent bandwidth of the pk function
    #pk_NEB = np.sqrt(pk.size / (pk**2).sum())
    
    # note we use a symmetric FT convention here, which applies sqrt(L/2pi) for each axis for forward and backward
    bft = np.fft.fft2(box, norm='ortho') # why are we doinf iF here?
    bft = np.fft.fftshift(bft)

    # apply the function
    bft *= pk

    # FFT back and take real (you can see imag is near-zero for yourself)
    grfc = np.fft.ifft2(np.fft.ifftshift(bft), norm='ortho') # should this not be iF
    grf = grfc.real
    ratitor = np.abs(grf.imag/grf.real).mean()

    assert ratitor<1e-10, f'imaginary part is too big!!, {ratitor}'
    assert grf.mean()<1e-10, f'real mean too large, {grf.mean()}'
    
    return grf, bft

def do_frames(a, b, scale, destpath):
    localpath = destpath+f'{scale}/vorticity/'
    Path(localpath).mkdir(parents=True, exist_ok=True)
    Ak = Ak_double_power_scale(a, b, scale)
    rng = np.random.default_rng()

    for i in range(100):
        dataunif = rng.uniform(size=(2*L-1, 2*L-1))
        G, _ = sim_grf(dataunif, 1, Ak, power=True)
        np.save(localpath+f'frame{i}.npy', G>0)

def wrapper(argdic):
    return do_frames(**argdic)



destpath = '/lustre/astro/rsx187/isolinescalingdata/vorticitydata/grf_2powers/'

scales = [0.03, 0.05, 0.07]
a, b = 2.5, -4
L = 2000

argdics = [{'destpath': destpath, 'a':a, 'b':b, 'scale':scale} for scale in scales]

ntasks = min(len(scales), np.floor(int(os.environ['SLURM_CPUS_PER_TASK'])).astype(int))
print(argdics)
#sys.exit()

with Pool(ntasks) as p: # maybe this is not the right place to do multiprocessing
    p.map(wrapper, argdics)
