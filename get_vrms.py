import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import massPy as mp
import sys
import pickle
import scipy.stats as scs
import glob
from natsort import natsorted


names = glob.glob('/lustre/astro/rsx187/datasets/olgadata/Ex*/*.mat')

def get_vrms(vx, vy):
    vrms = []
    for iframe in range(len(vx)):
        vxi, vyi = vx[iframe], vy[iframe]
        vrmsi = np.sqrt(np.mean(vxi**2 + vyi**2))
        vrms.append(vrmsi)
    return vrms

vrmslist = []
for name in names:
    print(name)
    dat = sio.loadmat(name)

    x, y = dat['x'][:, 0], dat['y'][:, 0]
    vx, vy = dat['u_smoothed'][:, 0], dat['v_smoothed'][:, 0]
    #vort = dat['vorticity'][:, 0]
    #vortices = dat['vortex_locator'][:, 0]
    vrms = get_vrms(vx, vy)
    vrmslist.append(vrms)


with open('vrms_olga_ex.pickle', 'wb') as f:
    pickle.dump([names, vrmslist], f)