import numpy as np
import os
from skimage import measure
from natsort import natsorted
import pickle
from pathlib import Path
from multiprocessing import Pool
import sys
import glob
import copy
import random

#sys.path.append('/groups/astro/rsx187/schrammloewnerevolution/')
#import schramm_loewner_evolution as sle

def perims_and_rgs(binary_image, connectivity=None, border=20, minperim=20):
    """
    get perimeters and radius of gyration for all clusters in image
    
    :param binary_image: 2D numpy binary array, 1 for cluster
    :param connectivity: int, 1 is edge neighbours, 2 is also diagonal neighbours and more
    
    returns list of perimeters, list of radius of gyration
    """
    
    # get clusters
    label_img, num = measure.label(binary_image, connectivity=connectivity, return_num=True)
    labels = np.arange(1, num+1)
    lx, ly = label_img.shape[0], label_img.shape[1]


    # get relevant measures
    dic = measure.regionprops_table(label_img, properties=('label', 'centroid', 'perimeter_crofton', 'area', 'moments_central'))#
    assert np.all(dic['label'] == labels)
    comxs = dic['centroid-1']
    comys = dic['centroid-0']
    pers = dic['perimeter_crofton']
    areas = dic['area']
    M20 = dic['moments_central-2-0']
    M02 = dic['moments_central-0-2']

    # mask because we want central clusters and minimum perimeter
    mask = (comxs>border) & (comxs<lx-border) & (comys>border) & (comys<ly-border) & (pers>minperim)
    borderlabels = np.unique(np.hstack([label_img[0], label_img[-1], label_img[:, 0], label_img[:, -1]]))
    #print(mask.shape, borderlabels.shape)
    if len(borderlabels>0) and len(mask>0):
        mask[borderlabels-1] = 0 # cut clusters that touch the boundary

    # get relevant clusters
    labelsuse = labels[mask]
    #comxuse = comxs[mask]
    #comyuse = comys[mask]
    persuse = pers[mask]
    areasuse = areas[mask]
    M20use = M20[mask]
    M02use = M02[mask]

    # calculate meaures
    I = M20use+M02use
    Rgs = np.sqrt(I/areasuse)

    perims = persuse

    return perims, Rgs


def run_perim_v_rg(foldername, outpath):
    """
    run perim vs rg for an archive
    write a pickled dictionary of the results
    """
    files = natsorted(os.listdir(foldername))[1:]
    #assert len(files)!=0
    if len(files)==0:
        print(f'{foldername} is empty, skipping')
        return
    print(f'start: {outpath}')
    perims, rads = [], []
    for file in files:
        try:
            field = np.load(foldername+'/'+file)
        except ValueError:
            field = np.loadtxt(foldername+'/'+file, delimiter=',')
        p, r = perims_and_rgs(field, connectivity=2)
        perims.extend(p)
        rads.extend(r)

    outdic = {
    'folder': foldername,
    'perims': perims,
    'rads': rads
    }
    with open(outpath+'.pickle', 'wb') as f:
        pickle.dump(outdic, f)
    print(f'finished: {outpath}')


def wrapper(path):
    try:
        field = np.load(path)
    except ValueError:
        field = np.loadtxt(path, delimiter=',')

    res = perims_and_rgs(field, connectivity=2)
    return res

def run_perim_v_rg_multi(foldername, outpath, ntasks):
    """
    run perim vs rg for an archive
    write a pickled dictionary of the results
    """
    files = natsorted(os.listdir(foldername))[1:]
    if len(files)==0:
        print(f'{foldername}, {outpath} did not have any files - skipped')
        return
    #assert len(files)!=0
    paths = [foldername+'/'+file for file in files]

    print(f'start: {outpath}')

    with Pool(ntasks) as p:
        res = p.map(wrapper, paths)

    perims, rads = [], []
    for item in res:
        p, r = item
        perims.extend(p)
        rads.extend(r)

    outdic = {
    'folder': foldername,
    'perims': perims,
    'rads': rads
    }
    with open(outpath+'.pickle', 'wb') as f:
        pickle.dump(outdic, f)
    print(f'finished: {outpath}')


if __name__ == '__main__':

    typeof = 'vorticity'
    typeof = 'pressure'
    #name = 'zeta_0.04_counter_2'
    size = 2048
    overwrite = True
    multiprocessing_of_run = True
    
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data2/simon_data/nematic_simulation{size}/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/wensink2012/3d_data_piv/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/simon_K_scan/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/simon_xi_scan/*/{typeof}')
    folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/pressuredata/colloids_sourav/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/Valeriia_tracking/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/compressibleAN/*')

    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/grf/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/varunBactTurb/*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/backofen/Forcing_*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/u_sample/uzk*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/uq_sample/qzk*/{typeof}')
    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/PIV_mol.perturb/*/{typeof}')

    #folder = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/Stress_Density_PIV_Tracking/*/{typeof}')
    #folder = [f for f in folder if 'counter_0' in f]
    #folder = [f for f in folder if ('counter_0' and 'counter_1' and 'counter_2') not in f]

    folder = [f for f in folder if '.ipynb' not in f]
    #folder = [f for f in folder if 'cut25' in name]

    #folder = folder[20:]

    random.shuffle(folder)
    
    assert len(folder)>0, f'0 names: {folder}'

    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data2/perimvsgyrout/simon_data/nematic_simulation{size}/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/wensink2012_3d_data_piv/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/simon_K_scan/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/simon_xi_scan/'
    outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/colloids_sourav/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/Valeriia_tracking/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/compressibleAN/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/grf/perimvsgyrout/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/varunBactTurb/perimvsgyrout/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/backofen/perimvsgyrout/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/u_sample/perimvsgyrout/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/uq_sample/perimvsgyrout/'
    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/PIV_mol.perturb/perimvsgyrout/'

    #outfolder = f'/lustre/astro/rsx187/isolinescalingdata/{typeof}data/perimvsgyrout/Stress_Density_PIV_Tracking/'
    Path(outfolder).mkdir(parents=True, exist_ok=True)

    # make outnames
    outnames = [name.split('/')[-2] for name in folder]
    outpaths = [outfolder+outname for outname in outnames]
    args = zip(copy.copy(folder), outpaths)

    # avoid overwriting / recalculating
    if not overwrite:
        for file, out in args:
            if os.path.isfile(out+'.pickle'):
                folder.remove(file)
                print(f'removed {file}')
        #remake outs
        outnames = [name.split('/')[-2] for name in folder]
        outpaths = [outfolder+outname for outname in outnames]
        args = zip(folder, outpaths)

    print(folder)
    print(f'number of runs: {len(folder)}')
    print(args)
    #sys.exit()

    ntasks = min(len(folder), int(os.environ['SLURM_CPUS_PER_TASK']))

    if multiprocessing_of_run:
        for arg in args:
            foldername, outpath = arg
            run_perim_v_rg_multi(foldername, outpath, ntasks=ntasks)
    else :
        with Pool(ntasks) as p:
            res = p.starmap(run_perim_v_rg, args)
        

