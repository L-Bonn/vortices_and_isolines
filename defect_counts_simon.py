import numpy as np
import sys
import os
import pickle
from multiprocessing import Pool
import glob
from pathlib import Path
import scipy.stats as scs
import copy

sys.path.append('../matlabdefects/')
import av_defs_py as adp

sys.path.append('/groups/astro/rsx187/massPynpz')
import massPynpz as mp

def def_to_array(defects):
    mainlist = []
    for de in defects:
        mainlist.append([*de['pos'], de['charge'], de['angle']])
    return np.array(mainlist)

def run_archive(name, out, write_each_frame=False, overwrite=False):
    #name, out = pathnameout
    ar = mp.archive.loadarchive(name+'/')
    #ar = archive.loadarchive(name+'/')
    #N = int((ar.nsteps-ar.nstart)/ar.ninfo)
    N = ar.num_frames
    pars = ar.__dict__#ar.parameters
    lx, ly = ar.LX, ar.LY
    dic_out = pars

    Path(out).mkdir(parents=True, exist_ok=True)

    outnameall = out+'_alldefects.pickle'
    if not write_each_frame: # don't redo if overwrite=False
        if not overwrite:
            if os.path.isfile(outname):
                return

    defectlists = []
    #print('running ', name, flush=True)
    for i in range(N): # this could use some parallelisation
        #print('frame ', i, flush=True)

        outname = f'{out}frame{i}_defects.csv'
        if write_each_frame: # don't redo if overwrite=False
            if not overwrite:
                if os.path.isfile(outname):
                    continue
        try:
            frame = ar._read_frame(i)
        except FileNotFoundError:
            continue
        try:
            defects = adp.defectlist_nematic_from_frame(frame, lx, ly)
        except AttributeError:
            continue

        #print('first defect', defects[0], flush=True)
        
        if write_each_frame:
            #print('writing ', i, flush=True)
            #print('to ',  out, flush=True)

            dic_out_i = copy.copy(dic_out)
            dic_out_i['defects'] = defects
            #with open(out+i+'_defects.pickle', 'wb') as f:
            #    pickle.dump(dic_out, f)
            np.savetxt(outname, def_to_array(defects), delimiter=',', header='x, y, charge, angle')

        else:
            defectlists.append(defects)

    if not write_each_frame:
        dic_out['defectlists'] = defectlists

        with open(outnameall, 'wb') as f:
            pickle.dump(dic_out, f)
    
    print('done', name)

def wrapper(argdic):
    return run_archive(**argdic)

if __name__ == '__main__':

    overwrite=False
    write_each_frame=True

    size = 2048
    #path = "/lustre/astro/rsx187/mmout/" + "uq_sample/"
    #path = "/lustre/astro/jayeeta/aniso/datas/size_1024/"
    path = f"/lustre/astro/kpr279/ns{size}*/"
    #path = "/lustre/astro/jayeeta/aniso/datas/size_2048/"
    #path = "/lustre/astro/robinboe/HiddenTransitions/"
    #path = "/lustre/astro/rsx187/mmout/" + "theta_sample/"
    #path = "/lustre/astro/robinboe/"
    #path = "/lustre/astro/ardash/FrictionProject23/lyotropic/forLasse/"
    #out = "/lustre/astro/rsx187/durbig/jayeeta_data/size_1024/"
    #out = f"/groups/astro/rsx187/mass/dur/simon_data/ns{size}/"
    #out = f"/lustre/astro/rsx187/durbig/simon_data_local/ns{size}"
    out = f"/lustre/astro/rsx187/isolinescalingdata/fullvorticitydata/simon_data/nematic_simulation{size}"


    Path(out).mkdir(parents=True, exist_ok=True)


    names = glob.glob(f'{path}*/*')

    names = [n for n in names if '.dat' not in n]
    names = [n for n in names if 'counter_0' in n]
    #names = [names[20]]

    
    print(names)
    #names = names[:2]
    #sys.exit()
    #if not overwrite: # not sure this works?

    #    done = os.listdir(out)
    #    print('n names:', len(names), ', n done:', len(done))
    #    print(f'skipped: {len([name for name in names if name+"_alldefects.pickle" in done])}')
    #    names = [name for name in names if name+"dur.pickle" not in done] 
    #    #names = names[:1]

    print(names)
    print(f'number of analyses: {len(names)}\n')
    #sys.exit()

    outnames = ['_'.join(n.split('/')[-1].split('_')[-4:]) for n in names]
    pathnames = [[n, out+'/'+outname] for n, outname in zip(names, outnames)]
    argdics = [{'name':n, 'out':out+'/'+outname+'/defects/', 'write_each_frame': write_each_frame} for n, outname in zip(names, outnames)]

    print(pathnames)
    #sys.exit()

    #ntasks = min(10, len(pathnames))#
    ntasks = min(len(pathnames), int(os.environ['SLURM_CPUS_PER_TASK']))
    #ntasks = 1
    print('ntasks:', ntasks, type(ntasks))
    csize = int(len(names)/ntasks)+1

    with Pool(ntasks) as p:
#       p.map(dur_pickle, pathnames, chunksize=csize)
        p.map(wrapper, argdics, chunksize=csize) 