import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage as ndi
import pickle
from multiprocessing import Pool
from pathlib import Path
import glob
sys.path.append('/groups/astro/rsx187/massPynpz')
import massPynpz as mp




def vortandE(vx, vy, return_terms=False):
    dxux = mp.base_modules.numdiff.gradient(vx, axis=1)
    dyux = mp.base_modules.numdiff.gradient(vx, axis=0)
    dxuy = mp.base_modules.numdiff.gradient(vy, axis=1)
    dyuy = mp.base_modules.numdiff.gradient(vy, axis=0)
    vort = dxuy - dyux
    E = dxux + dyuy
    
    if return_terms:
        return vort, E, dxux, dyux, dxuy, dyuy
    else:    
        return vort, E


#jeong and hussain 95, zhang 19
def Q_criterion(vort, E):#zhang 2019
    return vort**2 - E**2 

def Okubo_Weiss(vort, E, dxux, dyux, dxuy, dyuy): #martinez prat 2021
    return E**2 - 4*dxux*dyuy + 4*dxuy*dyux

def Omega_criterion(vort, E):#zhang 2019
    OM = vort**2/(vort**2+E**2)


def label_vortices(vortices):
    label, number = ndi.label(vortices)
    vals, counts = np.unique(label, return_counts=True)
    vals = vals[1:]
    counts = counts[1:]
    return label, number, vals, counts

def dilate_image(im):
    #doing a dilation to avoid tensor eigs of 0
    yshape = im.shape[1]
    yrepped = np.repeat(im, 2).reshape(yshape,-1)
    bothrepped = np.repeat(yrepped.T, 2).reshape(yrepped.shape[1], -1).T
    return bothrepped

def anisotropy(labelled_image, values):
    dilated_labelled_image = dilate_image(labelled_image)
    rats = []
    for label in values: # so slow :(
        im = (dilated_labelled_image==label)
        a, b = measure.inertia_tensor_eigvals(im)
        r = max(a/b, b/a)
        rats.append(r)
    return rats

def do_all(pathnameout, r=0.05):

    path, out = pathnameout
    print(f'starting {path}')
    tn, tp = r, 1-r 
    ar = mp.archive.loadarchive(path)

    savedic = ar.__dict__

    lx, ly = ar.LX, ar.LY
    nframe = ar.num_frames
    frameis = np.arange(nframe)


    #outname = path.split('/')[-2]+f"L{lx}"

    #collector lists

    vortsums = []
    vorthists = []
    vorticity_area_ratios = []

    vortex_numbers = []
    vortex_anisotropy = []

    for i in frameis: # this needs to be multiprocessed

        frame = ar._read_frame(i)
        vx, vy = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
        vort, E, dxux, dyux, dxuy, dyuy = vortandE(vx, vy, return_terms=True)
        
        negcut, poscut = np.quantile(vort, [tn, tp])
        vortices = (vort<negcut)*-1 + (vort>poscut)

        #binary fields
        vortices_n = (vortices==-1)
        vortices_p = (vortices==1)

        props =  ['label',  'inertia_tensor_eigvals']#'area',


        label_n, number_n = measure.label(dilate_image(vortices_n), connectivity=connectivity, return_num=True)
        labels = np.arange(1, number_n+1)
        dic_n = measure.regionprops_table(label_n, properties=(props))#
        #areas_n = dic_n['area']/2 # /2 due to dilation

        label_p, number_p = measure.label(dilate_image(vortices_p), connectivity=connectivity, return_num=True)
        labels = np.arange(1, number_n+1)
        dic_p = measure.regionprops_table(label_p, properties=(props))#
        #areas_p = dic_p['area']/2 # /2 due to dilation

        # anisotropy by inertia tensor eig vals
        i0_n, i1_n = dic_n['inertia_tensor_eigvals-0'], dic_n['inertia_tensor_eigvals-1']
        anisotropy_n = np.maximum(i0_n/i1_n, i1_n/i0_n)
        i0_p, i1_p = dic_p['inertia_tensor_eigvals-0'], dic_p['inertia_tensor_eigvals-1']
        anisotropy_p = np.maximum(i0_p/i1_p, i1_p/i0_p)

        vortsum = np.sum(vort) # must be 0
        vortcounts, vortbins = np.histogram(vort.ravel(), 50) #vorticity hist
        vorticity_area_ratio =  np.sum(vort>0)/np.sum(vort<=0) #area ratio of pos and neg

        vortex_numbers.append([number_n, number_p])
        vortex_anisotropy.append([anisotropy_n, anisotropy_p])

        vortsums.append(vortsum)
        vorthists.append([vortcounts, vortbins])
        vorticity_area_ratios.append(vorticity_area_ratio)
        # this can be functioned out

        # label_n, number_n, vals_n, counts_n = label_vortices(vortices_n)
        # label_p, number_p, vals_p, counts_p = label_vortices(vortices_p)

        # vortex_numbers.append([number_n, number_p])

        # rats_n = anisotropy(label_n, vals_n)
        # rats_p = anisotropy(label_p, vals_p)

        # vortex_anisotropy.append([rats_n, rats_p])

        # # some numbers
        # vortsum = np.sum(vort) # must be 0
        # vortcounts, vortbins = np.histogram(vort.ravel(), 50) #vorticity hist
        # vorticity_area_ratio =  np.sum(vort>0)/np.sum(vort<=0) #area ratio of pos and neg

        # vortsums.append(vortsum)
        # vorthists.append([vortcounts, vortbins])
        # vorticity_area_ratios.append(vorticity_area_ratio)

    savedic['vortsums'] = vortsums
    savedic['vorthists'] = vorthists
    savedic['vorticity_area_ratios'] = vorticity_area_ratios
    savedic['vortex_numbers'] = vortex_numbers
    savedic['vortex_anisotropy'] = vortex_anisotropy

    with open(f'{out}.pickle', 'wb') as f:
        pickle.dump(savedic, f)
    print(f'finished {path}, saved to {out}.pickle')


## parameters
r = 0.05 # 5% extreme vorticity
#tn, tp = r, 1-r 
size=1024

datapath = f"/lustre/astro/kpr279/ns{size}*/"
destpath = f"/lustre/astro/rsx187/isolinescalingdata/vortex_statistics/nematic_simulation{size}"
Path(destpath).mkdir(parents=True, exist_ok=True)
names = glob.glob(f'{datapath}*/*')
names = [n for n in names if '.dat' not in n]
names = [n for n in names if ('counter_0' or 'counter_1') in n]
print(names)
print('n names:', len(names))

#sys.exit()



outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names] # what
pathnames = [[n, destpath+'/'+outname] for n, outname in zip(names, outnames)]
ntasks = min(len(pathnames), int(os.environ['SLURM_CPUS_PER_TASK']))
ntasks = max(1, ntasks)
with Pool(ntasks) as p:
    p.map(do_all, pathnames)

    #paths = ['/home/liuzo/simonlustre/ns256/output_test_zeta_0.04/output_test_zeta_0.04_counter_0/',
# '/home/liuzo/simonlustre/ns256/output_test_zeta_0.15/output_test_zeta_0.15_counter_0/']


#paths
# pathc = '/home/liuzo/simonlustre/ns256/output_test_zeta_0.04/output_test_zeta_0.04_counter_0/'
# pathc = '/home/liuzo/simonlustre/ns256/output_test_zeta_0.15/output_test_zeta_0.15_counter_0/'
# pathc = '/home/liuzo/simonlustre/ns256/output_test_zeta_0.0245/output_test_zeta_0.0245_counter_0/'
# pathc = '/home/liuzo/simonlustre/ns256/output_test_zeta_0.05/output_test_zeta_0.05_counter_0/'
# pathc = '/home/liuzo/simonlustre/ns256pd/output_test_zeta_0.018/output_test_zeta_0.018_counter_0/'
# pathc= '/home/liuzo/simonlustre/ns256vl/output_test_zeta_0.022/output_test_zeta_0.022_counter_20/'


# ar = mp.archive.loadarchive(pathc)

# savedic = ar.__dict__

# lx, ly = ar.LX, ar.LY
# nframe = ar.num_frames
# frameis = np.arange(nframe)


# outname = pathc.split('/')[-2]+f"L{lx}"

# #collector lists

# vortsums = []
# vorthists = []
# vorticity_area_ratios = []

# vortex_numbers = []
# vortex_anisotropy = []

# for i in frameis:

#     frame = ar._read_frame(i)
#     vx, vy = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
#     vort, E, dxux, dyux, dxuy, dyuy = vortandE(vx, vy, return_terms=True)
    
#     negcut, poscut = np.quantile(vort, [tn, tp])
#     vortices = (vort<negcut)*-1 + (vort>poscut)

#     #binary fields
#     vortices_n = (vortices==-1)
#     vortices_p = (vortices==1)

#     # this can be functioned out

#     label_n, number_n, vals_n, counts_n = label_vortices(vortices_n)
#     label_p, number_p, vals_p, counts_p = label_vortices(vortices_p)

#     vortex_numbers.append([number_n, number_p])

#     rats_n = anisotropy(label_n, vals_n)
#     rats_p = anisotropy(label_p, vals_p)

#     vortex_anisotropy.append([rats_n, rats_p])

#     # some numbers
#     vortsum = np.sum(vort) # must be 0
#     vortcounts, vortbins = np.histogram(vort.ravel(), 50) #vorticity hist
#     vorticity_area_ratio =  np.sum(vort>0)/np.sum(vort<=0) #area ratio of pos and neg

#     vortsums.append(vortsum)
#     vorthists.append([vortcounts, vortbins])
#     vorticity_area_ratios.append(vorticity_area_ratio)

# savedic['vortsums'] = vortsums
# savedic['vorthists'] = vorthists
# savedic['vorticity_area_ratios'] = vorticity_area_ratios
# savedic['vortex_numbers'] = vortex_numbers
# savedic['vortex_anisotropy'] = vortex_anisotropy

# with open(f'pickles/{outname}.pickle', 'wb') as f:
#     pickle.dump(savedic, f)