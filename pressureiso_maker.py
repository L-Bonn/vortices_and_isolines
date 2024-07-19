import sys
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from pathlib import Path

sys.path.append('/groups/astro/rsx187/mass/')

from archive import archive
import av_defs_py as adp
from plot.plot import lyotropic as ly


zs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
uqs = [0.005, 0.01, 0.02, 0.04]
thetas = [0.001, 0.005, 0.01, 0.013, 0.017]
size = 512
for zeta in zs:
    print(zeta)
    #datapath = f'/lustre/astro/rsx187/mmout/active_sample_forperp/qzk1k30.05_K30.05_qkbt0_z{z}_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/rsx187/mmout/uq_sample/qzk1k30.05_K30.05_qkbt{uq}_z0_xi1_LX256_counter0/'
    #datapath = f'/lustre/astro/rsx187/mmout/theta_sample/qzk1k30.05_K30.05_qkbt{theta}_z0_xi1_LX256_counter0/'
    datapath = f'/lustre/astro/jayeeta/aniso/datas/size_{size}/zeta-{zeta}-out/'
    ar = archive.loadarchive(datapath)

    #destpath = f"/groups/astro/rsx187/pressuredata/theta_sample{ar.Q_kBT}"
    destpath = f"/groups/astro/rsx187/pressuredata/simon_data/active1024_{ar.zeta}"
    Path(destpath+"/pressure").mkdir(parents=True, exist_ok=True)

    copyfile(datapath+"parameters.json", destpath+"/parameters.json")

    N = int((ar.nsteps-ar.nstart)/ar.ninfo)
    startframe = 5
    framestep = 1
    for iframe in range(startframe,N,framestep):
        frame = ar.read_frame(iframe)
        LX, LY = ar.LX, ar.LY
        vx, vy = np.array([ly.get_velocity(frame.ff[i, :]) for i in range(frame.ff.shape[0])]).T
        vx_dat = vx.reshape(LX, LY)
        vy_dat = vy.reshape(LX, LY)
        qx = frame.QQxx.reshape(LX, LY)
        qy = frame.QQyx.reshape(LX, LY)
        
        sigmaXX, sigmaXY, sigmaYX, sigmaYY = adp.calcstress(frame, qx, qy, vx_dat, vy_dat)
        pressure = 0.5*(sigmaXX+sigmaYY)
        pressure0 = pressure>0
        np.save(f'{destpath}/pressure/frame{iframe}.npy', pressure0)