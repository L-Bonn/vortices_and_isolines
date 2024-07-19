# functions I use to run the SLE package
# Lasse 2024

import numpy as np 
import copy
import sys
sys.path.append('/groups/astro/rsx187/massPynpz')
import massPynpz as mp
sys.path.append('/groups/astro/rsx187/schrammloewnerevolution/')
import schramm_loewner_evolution as sle

def sle_stats(folder, file, rot=False, px=None, py=None, minsle=25, maxsle=30, num_rad=30, num_ang=30, max_trace_len=2*10**4):
    """
    run sle measurements: find contour, winding angle, left passage prob, driving function

    return: pairs, trace, hits, phi, drive, drivetime
    """
    #print(file)

    rng = np.random.default_rng()

    if rot:
        fieldbef = np.load(folder+file).T
    else:
        fieldbef = np.load(folder+file)
    
    field = draw_border(fieldbef, cut=px, y_realline=py)
    
    lx, ly = np.shape(field)
    assert lx == ly # would need to decide which is x and y otherwise

    if (px==None) & (py == None): # for basic version
        px = np.floor(lx/2).astype(int)
        py = 0
    px = int(px)
    py = int(py)
    trace = sle.measure.trace_contour(field, pixel_init=(px-1, py))#[:10**4]
    
    #if trace.shape[0]< 1000:
    #    return 
    if rng.random()>0.5: #doesn't seem to make much of a difference
        tracewind = trace[::-1]
    else:
        tracewind = trace

    wang, dist = sle.measure.winding_angle(tracewind[:max_trace_len])

    # more short data - this messes with beginning
    for i in range(10):
        continue # skip this
        cutlen = 300
        maxran = trace.shape[0]-cutlen-1
        try:
            istart = rng.integers(cutlen, maxran)
        except ValueError:
            continue
        wangshort, distshort = sle.measure.winding_angle(trace[istart:istart+cutlen])
        wang = np.hstack([wang, wangshort])
        dist = np.hstack([dist, distshort])

    pairs = np.vstack([dist, wang]).T
    
    centretrace = trace - trace[0]
    ctrace = centretrace[:, 0] + 1j*centretrace[:, 1]
    rmax = max(np.sqrt((ctrace.real)**2+ctrace.imag**2))
    hits, phi = sle.measure.left_passage_probability(ctrace, 
                                                     minsle, maxsle, num_rad=num_rad, num_ang=num_ang)
    # rad could be max rad
    
    drive, drivetime = sle.measure.driving_function(ctrace[:max_trace_len])
        
    return pairs, trace, hits, phi, drive, drivetime



def draw_border(fieldbef, cut=None, y_realline=None):
    
    fieldout = copy.copy(fieldbef)
    leftval = 1
    rightval = 0
    lx, ly = np.shape(fieldout)
    
    if not cut:
        cut = np.floor(lx/2).astype(int)
    if not y_realline:
        y_realline = 0
    cut = int(cut)
    y_realline = int(y_realline)
    fieldout[:cut, y_realline] = leftval
    fieldout[cut:, y_realline] = rightval
    fieldout[:, :y_realline] = leftval
    #fieldout[-1] = rightval
    #fieldout[:cut, -1] = leftval
    #fieldout[cut:, -1] = rightval
    return fieldout

def make_params(folder, files, nperfile=10, minsle=25, maxsle=30, border=5, num_rad=30, num_ang=30):
    """
    make a list of parameter lists to put into Pool.starmap
    unrotated, rotated pi/2, and then random start and random rotation

    return: params nrun x nparam
    """

    rng = np.random.default_rng()
    nfile = len(files)

    file = files[0]
    fieldbef = np.load(folder+file)
    lx, ly = fieldbef.shape
    #print(f'nfiles: {nfile}, nper: {nperfile}, n sle: {nfile*nperfile}')

    # plain
    folderrepped = np.repeat(folder, nfile)
    filesrepped = np.repeat(files, 1)
    rotchoice = np.tile(np.array([0]), nfile)
    pxs = np.tile(np.array([None]), nfile)
    pys = np.tile(np.array([None]), nfile)
    minsles = np.tile(np.array([minsle]), nfile)
    maxsles = np.tile(np.array([maxsle]), nfile)
    numrads = np.tile(np.array([num_rad]), nfile)
    numangs = np.tile(np.array([num_ang]), nfile)
    paramsbasic = np.array([folderrepped, filesrepped, rotchoice, pxs, pys, minsles, maxsles, numrads, numangs], dtype='object')

    # rotated
    folderrepped = np.repeat(folder, nfile)
    filesrepped = np.repeat(files, 1)
    rotchoice = np.tile(np.array([1]), nfile)
    pxs = np.tile(np.array([None]), nfile)
    pys = np.tile(np.array([None]), nfile)
    minsles = np.tile(np.array([minsle]), nfile)
    maxsles = np.tile(np.array([maxsle]), nfile)
    numrads = np.tile(np.array([num_rad]), nfile)
    numangs = np.tile(np.array([num_ang]), nfile)
    paramsrot = np.array([folderrepped, filesrepped, rotchoice, pxs, pys, minsles, maxsles, numrads, numangs], dtype='object')

    # random
    folderrepped = np.repeat(folder, nfile*(nperfile-2))
    filesrepped = np.repeat(files, nperfile-2)
    rotchoice = rng.integers(0, 2, nfile*(nperfile-2))
    pxs = rng.integers(0+border, lx-border, nfile*(nperfile-2))
    pys = rng.integers(0+border, lx-border, nfile*(nperfile-2))
    minsles = np.tile(np.array([minsle]), nfile*(nperfile-2))
    maxsles = np.tile(np.array([maxsle]), nfile*(nperfile-2))
    numrads = np.tile(np.array([num_rad]), nfile*(nperfile-2))
    numangs = np.tile(np.array([num_ang]), nfile*(nperfile-2))
    paramsrandom = np.array([folderrepped, filesrepped, rotchoice, pxs, pys, minsles, maxsles, numrads, numangs], dtype='object')


    params = np.hstack([paramsbasic, paramsrot, paramsrandom]).T

    return params