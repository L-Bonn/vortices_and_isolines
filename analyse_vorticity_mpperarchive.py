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
#sys.path.append('/groups/astro/rsx187/massPynpz')
#import massPynpz as mp
import massPy as mp
import copy
import json
from random import shuffle
from datetime import datetime
"""
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
"""
def vortandE(vx, vy, return_terms=False):
    dxux = mp.base_modules.numdiff.derivX(vx)
    dxuy = mp.base_modules.numdiff.derivX(vy)
    dyux = mp.base_modules.numdiff.derivY(vx)
    dyuy = mp.base_modules.numdiff.derivY(vy)
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
    return vort**2/(vort**2+E**2)

def get_vortices_extremal(vort, r):
    tn, tp = r, 1-r 
    negcut, poscut = np.quantile(vort, [tn, tp])
    vortices = (vort<negcut)*-1 + (vort>poscut)

    #binary fields
    return (vortices==-1), (vortices==1)

def get_vortices_Q(vort, E, t=0):
    Q = Q_criterion(vort, E)
    vortsign = np.sign(vort)
    vortices = (Q>t)*vortsign
    return (vortices==-1), (vortices==1)


def get_vortices_OW(vort, E, dxux, dyux, dxuy, dyuy):
    OW = Okubo_Weiss(vort, E, dxux, dyux, dxuy, dyuy)
    vortsign = np.sign(vort)
    vortices = (OW<0)*vortsign
    return (vortices==-1), (vortices==1)


def get_vortices_Omega(vort, E, thold=0.5):
    Omega = Omega_criterion(vort, E)
    vortsign = np.sign(vort)
    vortices = (Omega>thold)*vortsign
    return (vortices==-1), (vortices==1)



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

# def do_frame(path, framei, connectivity=1):

#     # very annoying it has to read each time but i cant pass the frame, and at least the memory is kept low
#     tn, tp = r, 1-r 
#     ar = mp.archive.loadarchive(path)
#     lx, ly = ar.LX, ar.LY
#     print(framei)
#     frame = ar._read_frame(framei)
#     vx, vy = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
#     vort, E, dxux, dyux, dxuy, dyuy = vortandE(vx, vy, return_terms=True)
    
#     negcut, poscut = np.quantile(vort, [tn, tp])
#     vortices = (vort<negcut)*-1 + (vort>poscut)

#     #binary fields
#     vortices_n = (vortices==-1)
#     vortices_p = (vortices==1)

#     # # this can be functioned out
#     # label_n, number_n, vals_n, counts_n = label_vortices(vortices_n)
#     # label_p, number_p, vals_p, counts_p = label_vortices(vortices_p)

#     # rats_n = anisotropy(label_n, vals_n)
#     # rats_p = anisotropy(label_p, vals_p)


#     props =  ['label',  'inertia_tensor_eigvals']#'area',


#     label_n, number_n = measure.label(dilate_image(vortices_n), connectivity=connectivity, return_num=True)
#     labels = np.arange(1, number_n+1)
#     dic_n = measure.regionprops_table(label_n, properties=(props))#
#     #areas_n = dic_n['area']/2 # /2 due to dilation

#     label_p, number_p = measure.label(dilate_image(vortices_p), connectivity=connectivity, return_num=True)
#     labels = np.arange(1, number_n+1)
#     dic_p = measure.regionprops_table(label_p, properties=(props))#
#     #areas_p = dic_p['area']/2 # /2 due to dilation

#     # anisotropy by inertia tensor eig vals
#     i0_n, i1_n = dic_n['inertia_tensor_eigvals-0'], dic_n['inertia_tensor_eigvals-1']
#     anisotropy_n = np.maximum(i0_n/i1_n, i1_n/i0_n)
#     i0_p, i1_p = dic_p['inertia_tensor_eigvals-0'], dic_p['inertia_tensor_eigvals-1']
#     anisotropy_p = np.maximum(i0_p/i1_p, i1_p/i0_p)

#     vortsum = np.sum(vort) # must be 0
#     vortcounts, vortbins = np.histogram(vort.ravel(), 50) #vorticity hist
#     vorticity_area_ratio =  np.sum(vort>0)/np.sum(vort<=0) #area ratio of pos and neg

#     return vortsum, [vortcounts, vortbins], vorticity_area_ratio, [number_n, number_p], [anisotropy_n, anisotropy_p]

def do_all(path=None, out=None, r=0.05, connectivity=1, method="extremal", Othold=0.52, Qthold=0, do_area=False, do_orientation=False):
    #path, out = pathnameout
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, starting {path}')
    tn, tp = r, 1-r 
    ar = mp.archive.loadarchive(path)

    savedic = ar.__dict__

    lx, ly = ar.LX, ar.LY
    nframe = ar.num_frames
    frameis = np.arange(nframe)


    #collector lists

    vortsums = []
    vorthists = []
    vorticity_area_ratios = []

    vortex_numbers = []
    vortex_anisotropy = []

    areas = []
    orientations = []

    #for i in frameis:
    try:
        for frame in ar.read_frames():

            print({datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, i)
            #frame = ar._read_frame(i)
            try:
                vx, vy = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
            except AttributeError:
                vx, vy = mp.base_modules.flow.velocity(frame.ff, lx, ly)
            vort, E, dxux, dyux, dxuy, dyuy = vortandE(vx, vy, return_terms=True)
            
            # negcut, poscut = np.quantile(vort, [tn, tp])
            # vortices = (vort<negcut)*-1 + (vort>poscut)

            # #binary fields
            # vortices_n = (vortices==-1)
            # vortices_p = (vortices==1)

            match method:
                case "extremal":
                    vortices_n, vortices_p = get_vortices_extremal(vort, r=r)
                case "Q_criterion":
                    vortices_n, vortices_p = get_vortices_Q(vort, E, t=Qthold)
                case "OW":
                    vortices_n, vortices_p = get_vortices_OW(vort, E, dxux, dyux, dxuy, dyuy)
                case "Omega":
                    vortices_n, vortices_p = get_vortices_Omega(vort, E, thold=Othold)

            props =  ['label',  'inertia_tensor_eigvals']#'area',
            if do_area:
                props.append('area')
            if do_orientation:
                props.append('orientation')
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
            if do_area:
                areas.append([dic_n['area'], dic_p['area']])
            if do_orientation:
                orientations.append([dic_n['orientation'], dic_p['orientation']])
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
        #savedic['vortex_anisotropy'] = vortex_anisotropy#
        savedic['inertia_eigs_n'] = [i0_n, i1_n]
        savedic['inertia_eigs_p'] = [i0_p, i1_p]
        savedic['area'] = areas
        savedic['orientation'] = orientations

        with open(f'{out}.pickle', 'wb') as f:
            pickle.dump(savedic, f)
        print(f'finished {path}')
    except FileNotFoundError:
        print(f'skipped {path} due to FileNotFoundError')

def wrapper_do_all(dic):
    return do_all(**dic)


def do_frame(path, framei, r=0.05, connectivity=1, method="extremal", Othold=0.52, Qthold=0, do_area=False):

    #print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, started do_frame {framei}', flush=True)
    # very annoying it has to read each time but i cant pass the frame, and at least the memory is kept low
    tn, tp = r, 1-r 
    ar = mp.archive.loadarchive(path)
    lx, ly = ar.LX, ar.LY
    print(framei)
    try:
        frame = ar._read_frame(framei)
    except FileNotFoundError: # some frames may be missing
        print(f'skipped frame {framei} due to FileNotFoundError, {path}', flush=True)
        return 
    #print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, loaded frame {framei}', flush=True)
    try:
        vx, vy = frame.vx.reshape(lx, ly), frame.vy.reshape(lx, ly)
    except AttributeError:
        vx, vy = mp.base_modules.flow.velocity(frame.ff, lx, ly)
    #except TypeError:

    
    vort, E, dxux, dyux, dxuy, dyuy = vortandE(vx, vy, return_terms=True)

    match method:
        case "extremal":
            vortices_n, vortices_p = get_vortices_extremal(vort, r=r)
        case "Q_criterion":
            vortices_n, vortices_p = get_vortices_Q(vort, E, t=Qthold)
        case "OW":
            vortices_n, vortices_p = get_vortices_OW(vort, E, dxux, dyux, dxuy, dyuy)
        case "Omega":
            vortices_n, vortices_p = get_vortices_Omega(vort, E, thold=thold)

    props =  ['label',  'inertia_tensor_eigvals']
    if do_area:
        props.append('area')
    if do_orientation:
        props.append('orientation')

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
    #i0_n, i1_n = np.sqrt(dic_n['inertia_tensor_eigvals-0']), np.sqrt(dic_n['inertia_tensor_eigvals-1'])
    #anisotropy_n = np.maximum(i0_n/i1_n, i1_n/i0_n)
    #anisotropy_n = (i0_n-i1_n)/(i0_n+i1_n)
    i0_p, i1_p = dic_p['inertia_tensor_eigvals-0'], dic_p['inertia_tensor_eigvals-1']
    #i0_p, i1_p = np.sqrt(dic_p['inertia_tensor_eigvals-0']), np.sqrt(dic_p['inertia_tensor_eigvals-1'])
    #anisotropy_p = np.maximum(i0_p/i1_p, i1_p/i0_p)
    #anisotropy_p = (i0_p-i1_p)/(i0_p+i1_p)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, created dicts, frame {framei}', flush=True)
    vortsum = np.sum(vort) # must be 0
    vortcounts, vortbins = np.histogram(vort.ravel(), 50) #vorticity hist
    vorticity_area_ratio =  np.sum(vort>0)/np.sum(vort<=0) #area ratio of pos and neg

    # vortex_numbers.append([number_n, number_p])
    # vortex_anisotropy.append([anisotropy_n, anisotropy_p])

    # vortsums.append(vortsum)
    # vorthists.append([vortcounts, vortbins])
    # vorticity_area_ratios.append(vorticity_area_ratio)
    #returnlist = [vortsum, [vortcounts, vortbins], vorticity_area_ratio, [number_n, number_p], [anisotropy_n, anisotropy_p]]
    returndic = {
     'vortsum':vortsum, 
     'vorthist':[vortcounts, vortbins],
     'vorticity_area_ratio':vorticity_area_ratio,
     'vortex_number':[number_n, number_p], 
     'inertia_eigs_n':[i0_n, i1_n],
     'inertia_eigs_p':[i0_p, i1_p],
     #'vortex_anisotropy':[anisotropy_n, anisotropy_p]
     }
    if do_area:
        #returnlist.append([dic_n['area'], dic_p['area']])
        returndic['area'] = [dic_n['area'], dic_p['area']]
    if do_orientation:
        returndic['orientation'] = [dic_n['orientation'], dic_p['orientation']]
    # write returndic here - if it times out before finishing
    return returndic



def do_archive(ntasks=1, path=None, out=None, r=0.05, connectivity=1, method="extremal", Othold=0.52,  Qthold=0, do_area=False, do_orientation=False):
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, starting {path}')

    ar = mp.archive.loadarchive(path)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, loaded archive')
    savedic = ar.__dict__
    savedic['method'] = method
    savedic['Qthold'] = Qthold
    savedic['Othold'] = Othold

    nframe = ar.num_frames
    frameis = np.arange(nframe)#[::10]

    #del ar
    #ntasks = min(ntasks, np.floor(int(os.environ['SLURM_CPUS_PER_TASK'])/2).astype(int))
    ntasks = min(ntasks, int(os.environ['SLURM_CPUS_PER_TASK']))
    #collector lists
    vortsums = []
    vorthists = []
    vorticity_area_ratios = []


    vortex_numbers = []
    #vortex_anisotropies = []
    inertias_n = []
    inertias_p = []

    areas = []
    orientations = []
    ncpu = min(ntasks, nframe)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, ncpus: {ncpu}')

    rundics = []
    for framei in frameis:
        dic = {'path':path, 'framei':framei,
         'r':r, 'connectivity':connectivity, 'method':method, 'Othold':Othold, 'Qthold':Qthold, 'do_area':do_area}
        rundics.append(dic)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, made rundics')

    with Pool(ncpu) as p:
        res = p.map(wrapper_do_frame, rundics, chunksize=1)

    for rpart in res:
        try:
            vortsums.append(rpart['vortsum'])
            vorthists.append(rpart['vorthist'])
            vorticity_area_ratios.append(rpart['vorticity_area_ratio'])
            vortex_numbers.append(rpart['vortex_number'])
            #vortex_anisotropies.append(rpart['vortex_anisotropy'])
            inertias_n.append(rpart['inertia_eigs_n'])
            inertias_p.append(rpart['inertia_eigs_p'])
            if do_area:
                areas.append(rpart['area'])
            if do_orientation:
                orientations.append(rpart['orientation'])
        except:
            continue


    savedic['vortsums'] = vortsums
    savedic['vorthists'] = vorthists
    savedic['vorticity_area_ratios'] = vorticity_area_ratios
    savedic['vortex_numbers'] = vortex_numbers
    #savedic['vortex_anisotropy'] = vortex_anisotropies
    savedic['inertias_n'] = inertias_n
    savedic['inertias_p'] = inertias_p

    if do_area:
        savedic['area'] = areas
    if do_orientation:
        savedic['orientation'] = orientations
    with open(f'{out}.pickle', 'wb') as f:
        pickle.dump(savedic, f)
    print(f'finished {path}, nframes = {len(vortsums)}')


def wrapper_do_frame(dic):
    return do_frame(**dic)

if __name__ == "__main__":


    mpinarchive=True
    ## parameters
    r = 0.05 # 5% extreme vorticity
    Othold = 0.52
    Qthold=2*10**-8
    do_area=True
    do_orientation=True
    vortex_type = 'Q_criterion'
    overwrite = False

    do_area_suff=None
    do_orientation_suff=None
    if do_area:
        do_area_suff='area'
    if do_orientation:
        do_orientation_suff='orientation'


    size=512

    #datapath = f"/lustre/astro/kpr279/ns{size}*/*/*"
    datapath = f"/lustre/astro/kpr279/ns{size}*2/*/*"
    #datapath = f"/lustre/astro/rsx187/mmout/simon_CC_scan/*"
    #datapath = f"/lustre/astro/rsx187/mmout/uq_sample/*"
    #destpath = f"/lustre/astro/rsx187/isolinescalingdata/vortex_statistics/{vortex_type}{do_area_suff}{do_orientation_suff}qthold2e-8_eigs/nematic_simulation{size}"
    destpath = f"/lustre/astro/rsx187/isolinescalingdata/vortex_statistics/{vortex_type}{do_area_suff}{do_orientation_suff}qthold2e-8_eigs/nematic_simulation{size}_2"
    #destpath = f"/lustre/astro/rsx187/isolinescalingdata/vortex_statistics/{vortex_type}{do_area_suff}{do_orientation_suff}qthold2e-8_eigs/simon_CC_scan"
    #destpath = f"/lustre/astro/rsx187/isolinescalingdata/vortex_statistics/{vortex_type}{do_area_suff}{do_orientation_suff}qthold2e-8_eigs/uq_sample"
    print(destpath)
    #par_varied = 'xi'
    #datapath = f"/lustre/astro/rsx187/mmout/simon_{par_varied}_scan/*"
    #destpath = f"/lustre/astro/rsx187/isolinescalingdata/vortex_statistics/{vortex_type}/simon_{par_varied}_scan"

    Path(destpath).mkdir(parents=True, exist_ok=True)
    names = glob.glob(datapath)

    #necessary
    #names = [n for n in names if 'sdn' not in n]
    names = [n for n in names if '.dat' not in n]

    #choice
    #names = [n for n in names if '0.019' not in n]
    #names = [n for n in names if 'z0.024' in n]
    #names = [n for n in names if 'counter0' in n]
    #names = [n for n in names if '1024' in n]
    #names = [n for n in names if 'cc0' not in n]
    #names = [n for n in names if 'xi1' in n]
    #names = [n for n in names if 'xi0_z0.021_LX2048_counter0' in n]
    #names = [n for n in names if 'counter_0' in n]
    #names = [n for n in names if ('counter_0' and 'counter_1' and 'counter_2') not in n]
    #names = [names[0]]
    #names = names[:15]
    names = names[30:]
    print(names)
    print('n names:', len(names))

    #sys.exit()

    #outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names] # for kpr279/ns...
    outnames = [n.split('/')[-1] for n in names] # for mmout/simon_...

    pathnames = [[n, destpath+'/'+outname] for n, outname in zip(copy.copy(names), outnames)]

    #for name, outname in pathnames:
    #    nfiles = len(os.listdir(name))
    #    if nfiles!=182:
    #        names.remove(name)
    #        print(f'removed {name}, {nfiles} files')
    #outnames = [n.split('/')[-1] for n in names] # for mmout/simon_...
    #pathnames = [[n, destpath+'/'+outname] for n, outname in zip(copy.copy(names), outnames)]

    if not overwrite:
        for name, outname in pathnames:
            if os.path.isfile(outname+'.pickle'):
                names.remove(name)
                print(f'{name} already exists')
        #remake without existing
        #outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names] # for kpr279/ns...
        outnames = [n.split('/')[-1] for n in names] # for mmout/simon_...
        pathnames = [[n, destpath+'/'+outname] for n, outname in zip(names, outnames)]
    print(names)
    #sys.exit()

    #pathnames = pathnames[:3]
    #ntasks = min(len(pathnames), np.floor(int(os.environ['SLURM_CPUS_PER_TASK'])/2).astype(int))
    ntasks = 5
    params = [{'path':path, 'out':destpath+'/'+outname,  'method':vortex_type,
               'r':r, 'Othold':Othold, 'Qthold':Qthold, 'do_area':do_area, 'do_orientation':do_orientation}
              for path, outname in zip(names, outnames)]
    #sys.exit()
    shuffle(params)

    if mpinarchive:
        for par in params:
            do_archive(ntasks=int(os.environ['SLURM_CPUS_PER_TASK']), **par)
    else:
        with Pool(ntasks) as p:
            p.map(wrapper_do_all, params)
