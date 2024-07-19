import glob
from multiprocessing import Pool
import pickle
import sle_helpers as sleh
import os
import sys
from pathlib import Path
import copy

def make_outname(name):
    return outpath+'/'.join(name.split('/')[-2:])+'.pickle'

#SETTINGS
data_type = 'vorticity'

args = {'nperfile':10,
'minsle':25,
'maxsle':30,
'border':5,
'num_rad':100,
'num_ang':30}
outpath = "/lustre/astro/rsx187/isolinescalingdata/benjaminsle/"
overwrite = False

names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{data_type}data/simon_data/*/*')
print(names)
#names = [n for n in names if 'counter_1' in n]

assert len(names)>0
#names = [names[0]]
print(names)

ncpu=int(os.environ['SLURM_CPUS_PER_TASK'])
print(f'ncpu: {ncpu}')

if not overwrite:
    print('removing:')
    for n in names:
        if os.path.isfile(make_outname(n)):
            names.remove(n)
            print(n)
    print('finished removing.')

print(f'n names now: {len(names)}')
#sys.exit()

for name in names:
    print(f'starting {name}')
    #files = glob.glob(name+f'/{data_type}/')
    folder=f'{name}/{data_type}/'
    files = os.listdir(name+f'/{data_type}/')
    params = sleh.make_params(folder=folder, files=files, **args)
    chunksize = int(len(params)/ncpu)
    with Pool(ncpu) as p:
        res = p.starmap(sleh.sle_stats, params, chunksize=chunksize)

    # what to save
    savedic = copy.copy(args)
    savedic['data_type'] = data_type
    savedic['name'] = name
    savedic['results'] = res

    #create dir if needed
    outname = make_outname(name)
    pathout = Path(outname)
    Path(pathout.parents[0]).mkdir(parents=True, exist_ok=True)

    #save
    with open(outname, 'wb') as f:
        pickle.dump(savedic, f)
    print(f'finished {name}')
