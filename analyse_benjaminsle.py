import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import copy
from scipy.stats import norm
from pathlib import Path
import sys
import os
from multiprocessing import Pool
sys.path.append('/groups/astro/rsx187/schrammloewnerevolution/')
import schramm_loewner_evolution as sle


def fit_wa(x, k, a):
    #return a*np.log(x)
    return a+ k*np.log(x)

def remove_flips(drive):
    # sometimes the walk flips to its - value 
    # this doesnt do much for the data
    currdrive = copy.copy(drive)
    var=0
    for istep in range(len(currdrive)):
        steps = np.diff(currdrive)
        newvar = np.var(steps[:istep])
        varrat = newvar/var
        #print(newvar, varrat)
        if varrat>4:
            currdrive[istep:] *= -1

        var = newvar 
    return currdrive

def do_all(name=None, outname=None):

    print(f'starting {name}')
    with open(name, 'rb') as f:
        dat = pickle.load(f)
    res = dat['results']

    if not res[0]:
        # no results
        return
    pairslist = []
    traces = []
    leftpass = []
    drives = []
    drivetimes = []
    for r in res:
        try:
            pairs, trace, hits, phi, drive, drivetime = r
        except TypeError:
            continue
        pairslist.append(pairs)
        traces.append(trace)
        leftpass.append([hits, phi])
        drives.append(drive)
        drivetimes.append(drivetime)
    
    pairs = np.vstack(pairslist)


    savedic = copy.copy(dat)
    del savedic['results']


    ## WINDING ANGLE
    dist = pairs[:, 0]
    wang = pairs[:, 1]

    mask = (50 < dist) #& (dist<10**3)
    wang_cutoff = wang[mask]
    dist_cutoff = dist[mask]

    nbins = 30
    wang_var, dist_bin_edges_log, binnum = binned_statistic(np.log(dist_cutoff), wang_cutoff, np.var, nbins)
    dist_bin_edges = np.exp(dist_bin_edges_log)
    dist_bin_centers = .5 * (dist_bin_edges[:-1] + dist_bin_edges[1:])

    xmin, xmax = dist_bin_edges.min(), dist_bin_edges.max()
    popt, pcov = curve_fit(fit_wa, dist_bin_centers, wang_var)


    savedic['winding_angle_xy'] = [dist_bin_centers, wang_var]
    savedic['winding_angle_popt_pcov'] = [popt, pcov]

    ##WINDING ANGLE HIST
    angs = copy.copy(pairs[:, 1])
    angs /= np.std(angs)
    # rescale angs by ang/var(ang)
    minang, maxang = min(angs), max(angs)
    bins = np.linspace(1.05*minang, 1.05*maxang, 20)
    bin_centres = 0.5*(bins[1:]+bins[:-1])
    hist, _ = np.histogram(angs, bins=bins, density=True)

    savedic['winding_angle_hist'] = [bin_centres, hist]

    ##LEFT PASSAGE
    num_rad = dat['num_rad']
    arleftpass = np.array(leftpass)
    arleftpass.shape

    counts = arleftpass[:, 0]

    phis = arleftpass[:, 1][0]

    arphi = np.array(phis)
    countsnorm = counts/(num_rad)

    mcounts = np.nanmean(countsnorm, axis=0)
    scounts = np.nanstd(countsnorm, axis=0)

    savedic['left_passage_xys'] = [phis, mcounts, scounts]

    ##DRIVING FUNCTION

    analysistimes = np.arange(10, 1001, 10)
    ntraces = len(drives)
    datdrive = np.full([len(analysistimes), ntraces], np.nan)
    for i in range(ntraces):
        drive, time = drives[i], drivetimes[i]
        #drive = remove_flips(drive)
        interdat = np.interp(analysistimes, time, drive)
        datdrive[:, i] = interdat

    m = np.var(datdrive, axis=1)
    savedic['driving_xy'] = [analysistimes, m]

    checktimes = [100, 200]

    idchecktimes = np.argwhere([a in checktimes for a in analysistimes])

    mi, ma = -6, 6
    bins = np.linspace(mi, ma, 30)
    bin_centres = 0.5*(bins[1:]+bins[:-1])

    hists = []
    for i, t in enumerate(checktimes):
        idat = idchecktimes[i]
        dattimei = datdrive[idat].squeeze()
        
        #points = dattimei/np.std(dattimei)
        points = dattimei/np.sqrt(6*checktimes[i])
        hist, _ = np.histogram(points, bins=bins, density=True)
        hists.append(hist)
    savedic['driving_hists'] = [bin_centres, hists]

    pathout = Path(outname)
    Path(pathout.parents[0]).mkdir(parents=True, exist_ok=True)

    with open(f'{outname}', 'wb') as f:
        pickle.dump(savedic, f)
    print(f'finished {name}')


def wrapper(dict):
    return do_all(**dict)
    

if __name__ == '__main__':

    overwrite = False
    names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/benjaminsle/*/*')
    names = [n for n in names if n.endswith('pickle')]


    outnames = [name.replace('benjaminsle', 'benjaminsleanalysed') for name in names]
    print(names)
    if not overwrite:
        print('removing:')
        for i, n in enumerate(names):
            if os.path.isfile(outnames[i]):
                names.remove(n)
                print(n)
        print('finished removing.')
    #sys.exit()
    ntasks = min(len(names), int(os.environ['SLURM_CPUS_PER_TASK']))
    params = [{'name':name, 'outname':outname,} for name, outname in zip(names, outnames)]
    print('starting')
    with Pool(ntasks) as p:
        p.map(wrapper, params)
    #for param in params:
    #    wrapper(param)