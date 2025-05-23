# making Francisco's ipynb into a python script, Lasse
# please choose options and data at beginning of main

import numpy as np
import glob
from skimage import measure
import random
from scipy.stats import norm,sem
import matplotlib.pyplot as plt
import math
#import cv2 
from collections import deque
import numexpr as ne
import scipy
from scipy import optimize
from mpmath import *
from mpmath import cot
from numpy import ndarray
from scipy.special import gamma
import scipy.special as sc
import os
import copy
import sys
sn = ellipfun('sn')
from multiprocessing import Pool
sys.path.append('/groups/astro/rsx187/schrammloewnerevolution/')
import schramm_loewner_evolution as sle



def clean_func(FOLDER):
    #clean out previous results
    for folder in FOLDER:
        files = os.listdir(folder)
        todel = [f for f in files if f.endswith('.npy')]
        for t in todel:
            if os.path.isfile(folder+'/'+t):
                os.remove(folder+'/'+t)


def pad_ragged(ar):
    # pad list of lists into nlist x maxlenlist array 
    try:
        max_length = max([a.shape[0] for a in ar])
    except:
        max_length = max(len(a) for a in ar)
    arout = np.full((len(ar), max_length), np.nan)
    for i, t in enumerate(ar):
        try:
            l = t.shape[0]
        except:
            l = len(t)
        arout[i] = np.pad(t, pad_width=(0, max_length-l), constant_values=np.nan)
    return arout

def trace_pad(Trace):
    Tracex = pad_ragged(Trace[0])
    Tracey = pad_ragged(Trace[1])
    return [Tracex, Tracey]

def Draw(matrix):
    # can be improved but no need
    lx,ly=matrix.shape
    for i in range(lx): # putting 1 on the bottom and 0 on the top
        matrix[i][0]=1
        matrix[i][ly-1]=0
    for j in range(ly):
        if j/ly<0.5:
            matrix[0][j]=1
            matrix[lx-1][j]=1
        else:
            matrix[0][j]=0
            matrix[lx-1][j]=0
        
    return matrix

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
    fieldout[-1] = rightval
    fieldout[-0] = leftval
    fieldout[:cut, -1] = leftval
    fieldout[cut:, -1] = rightval
    return fieldout

def rotate_matrix(matrix, degrees):
    if degrees == 0:
        return matrix
    elif degrees == 90:
        return np.rot90(matrix, k=1)
    elif degrees == 180:
        return np.rot90(matrix, k=2)
    elif degrees == 270:
        return np.rot90(matrix, k=3)


def remove_loops(X_ALL_loops, Y_ALL_loops):
    """
    to remove contours that loop - by removing those that end at y=0 (implicit)
    """
    endsatinf = [True if loop[-1]!=0 else False for loop in Y_ALL_loops ]# x will end at 0 due to Draw
    X_ALL_loops = [loop for end, loop in zip(endsatinf, X_ALL_loops) if end]
    Y_ALL_loops = [loop for end, loop in zip(endsatinf, Y_ALL_loops) if end]
    
    return X_ALL_loops, Y_ALL_loops

def SLE_Trace(foldertypeanal, removeloops=True):
    folder, type_anal = foldertypeanal
    X_curve_final=[]
    Y_curve_final=[]

    Data_list=glob.glob(str(folder)+f"/{type_anal}/*")
    for i,link in enumerate(Data_list): # for file
        
        try:
            data=np.load(link)
        #except ValueError: 
        #    data = np.load(link)
        except UnicodeDecodeError:
            data = np.load(link)
        except ValueError: #not gna get used
            data = np.loadtxt(link, delimiter=',')
        
        data = data.astype('int')

        #data=1-np.where(data==-1, 0, data)
        data=np.where(data==-1, 0, data)
        data2=data.copy()

        for angle in [0,90]: # do it at different rotations
            data3=data2.copy()
            data3=rotate_matrix(data3, angle)
            data3=Draw(data3)
            
            contours = measure.find_contours(data3, 0.5, fully_connected='high') #strange way of doing things
            counter2=[counter for counter in contours if counter[0][0]==0] # if it has x=0 at i=0
            indices = np.flip(np.argsort([len(sublist) for sublist in counter2]))
            sorted_list = [counter2[i] for i in indices]
            for j,counter in enumerate(sorted_list):
                if j<1: # just take the first (longest)
                    X_curve_final.append(counter[:, 1]-counter[0, 1]) # why only subtract from x?? (maybe y[0] is already 0)
                    Y_curve_final.append(counter[:, 0])
            #print(link, len(X_curve_final), flush=True)
            del indices,sorted_list,contours
    if removeloops:
        X_curve_final,Y_curve_final = remove_loops(X_curve_final,Y_curve_final)
    return  X_curve_final,Y_curve_final

def chunks(lst, n):
    LIST=[]
    for i in range(0, len(lst), n):
        if i+n>len(lst):
            break
        LIST.append( lst[i:i + n])
    return LIST

def df_first(Tracex_emsemble,Tracey_emsemble):
    L_stick=10**np.arange(0.7, 5, 0.1)
    nummax=len(L_stick)
    To_save=[]
    for X_par,Y_par in zip(Tracex_emsemble,Tracey_emsemble):
        N=len(X_par)
        num_sticks_loop=np.zeros((nummax,), dtype=int)
        for j in range(len(L_stick)):
            num=0
            x0,y0=X_par[0],Y_par[0]
            l=L_stick[j]
            for i in range(N):
                x1,y1=X_par[i],Y_par[i]
                if((x0-x1)**2+(y0-y1)**2)**0.5>l:
                    x0,y0=x1,y1
                    num+=1
            num_sticks_loop[j]=num
        To_save.append(num_sticks_loop)
    return L_stick,To_save

def df_last(X_ALL_loops,Y_ALL_loops,cut):
    L_stick=10**np.arange(0.2,2, 0.1)
    num_total=len(L_stick)
    num_sticks_final=np.zeros((num_total,), dtype=int)
    To_save=[]
    lll=0
    for X,Y in zip(X_ALL_loops,Y_ALL_loops):
        X_list=[X]
        Y_list=[Y]
        for X_par,Y_par in zip(X_list,Y_list):
            num_sticks_loop=np.zeros((num_total,), dtype=int)
            X_par,Y_par=X_par[:cut],Y_par[:cut]
            N=len(X_par)
            for k in range(len(L_stick)):#ver todos os sticks
                num=0
                x0,y0=X_par[0],Y_par[0]
                l=L_stick[k]
                i=0
                while i<N-1:
                    j=N-1
                    while j>i:
                        x1,y1=X_par[j],Y_par[j]
                        if((x0-x1)**2+(y0-y1)**2)**0.5<l:
                            x0,y0=x1,y1
                            num+=1
                            i=j
                            break
                        j=j-1
                num_sticks_loop[k]=num
            if num_sticks_loop[0]>1 and num_sticks_loop[-1]>=0:
                lll+=1
                To_save.append(num_sticks_loop)
                if lll>100:
                    return L_stick,To_save
    return L_stick,To_save

def angle_conterclock(a,b):
    cosTh = np.dot(a,b)
    sinTh = np.cross(a,b)
    return np.arctan2(sinTh,cosTh)

def winding(X,Y):
    VEC=np.array((np.diff(X),np.diff(Y))).T
    thetai=0
    THETA=[]
    for j in range(len(VEC)-1): #this doesnt have to be a for loop
        A=VEC[j]
        B=VEC[j+1]
        alphai=angle_conterclock(A,B)
        if alphai>3.1415:
            alphai=-alphai
        thetai=thetai+alphai
        
        THETA.append(thetai)
    return np.array(THETA)
    

# def winding_statistics(loops_to_save,cut):
#     L_list=10**np.arange(1,3,0.1)
#     L_list =L_list.astype('int32')
#     THETA_list=[]
#     X_ALL_loops,Y_ALL_loops=loops_to_save
    

    
#     f=0
#     for X_par,Y_par in zip(X_ALL_loops,Y_ALL_loops):
        
#         N=len(X_par)
        
        
#         X_list=chunks(X_par, cut)
#         Y_list=chunks(Y_par, cut)
        
#         for X,Y in zip(X_list,Y_list):
#             N=len(X)
#             theta_prov=np.zeros((len(L_list),))
#             for j,L in enumerate(L_list):
                
#                 dist=0
#                 x0,y0=X[0],Y[0]
#                 X_wind=[x0]
#                 Y_wind=[y0]
            
#                 for i in range(N):
#                     x1,y1=X[i],Y[i]
#                     X_wind.append(x1)
#                     Y_wind.append(y1)
#                     dist0=((x1-x0)**2+(y1-y0)**2)**0.5
#                     dist=dist+dist0
#                     #print(dist)
#                     x0,y0=x1,y1
#                     if dist>=L:
#                         p=random.random()
#                         if p<0.5:  
#                             X_wind,Y_wind=X_wind[::-1],Y_wind[::-1] 
#                         winding_L=winding(X_wind,Y_wind)
#                         theta_prov[j]=np.var(winding_L)
#                         break
#             f+=1
#             THETA_list.append(theta_prov)
#         if f>1000:
#                 break
#     return L_list,np.array(THETA_list)

def winding_statistics(loops_to_save, cut, do_chunks):
    if do_chunks:
        evalLs = 10**np.arange(1,np.log10(cut),0.1)
    else:
        evalLs = 10**np.arange(1,4,0.1)

    dat = [[] for i in range(len(evalLs))]
    nloops = loops_to_save.shape[1]
    for i in range(nloops):
        trace = loops_to_save[:, i].T

        #chunking
        if do_chunks:
            #lennonnan = np.sum(~np.isnan(trace))
            nch = np.floor(len(trace)/cut).astype(int)
            trace_chunks = (np.array_split(trace, (np.arange(nch-1)+1)*cut))
        else:
            trace_chunks = [trace]
        for tr in trace_chunks:
            wang_, dist_ = sle.measure.winding_angle(tr)
            for j, L in enumerate(evalLs):
                va = np.var(wang_[dist_<=L])
                #if np.isnan(va): continue
                dat[j].append(va)
                #assert ~np.isnan(va), f'{L}, {j}'
    return evalLs, np.array(dat).T


def coarsegrain_first(X,Y,l):
    N=len(X)
    x0,y0=X[0],Y[0]
    X_new=[x0]
    Y_new=[y0]
    for i in range(N):
        x1,y1=X[i],Y[i]
        if((x0-x1)**2+(y0-y1)**2)**0.5>l:
            x0,y0=x1,y1
            X_new.append(x0)
            Y_new.append(y0)
    return X_new,Y_new


def winding_at_L(loops_to_save,L):
    X_ALL_loops,Y_ALL_loops=loops_to_save
    THETA_forL=[]
    f=0
    for X_par,Y_par in zip(X_ALL_loops,Y_ALL_loops):
        cut=200
        X_par,Y_par=coarsegrain_first(X_par,Y_par,3)
        X_list=chunks(X_par, cut)
        Y_list=chunks(Y_par, cut)

        
        for X,Y in zip(X_list,Y_list):
            N=len(X)
            dist=0
            f=f+1
            x0,y0=X[0],Y[0]
            X_wind=[x0]
            Y_wind=[y0]
            for i in range(1,N):
                x1,y1=X[i],Y[i]
                X_wind.append(x1)
                Y_wind.append(y1)
                dist0=((x1-x0)**2+(y1-y0)**2)**0.5
                dist=dist+dist0
                x0,y0=x1,y1
                if dist>=L:

                    p=random.random()

                    if p<0.5:

                        X_wind,Y_wind=X_wind[::-1],Y_wind[::-1]

                    winding_L=winding(X_wind,Y_wind)

                    THETA_forL.extend(winding_L-np.mean(winding_L))
                    dist=0
                    X_wind=[x0]
                    Y_wind=[y0]
                    break
        if f>1000:
                break
            
        

    return L,THETA_forL

def inverse_sc_all(X_ALL_loops, Y_ALL_loops, lx, ly, cut=5_000, rescale=100):
    """
    params: X_ALL_loops, Y_ALL_loops, scalar box length lx, scalar box length ly
    perform an inverse Schwarz Christoffel transformation as defined in sle package
    includes scaling down to a box and back to size rescale
    """
    transformed_X_ALL, transformed_Y_ALL = [], []
    for X, Y in zip(X_ALL_loops, Y_ALL_loops):
        trace = np.vstack([X, Y]).T
        centretrace = trace - trace[0] # centering

        centretrace /= np.array([lx, ly]) # shrinking
        ctrace = centretrace[:, 0] + 1j*centretrace[:, 1] # complexification
        sc_ctrace = sle.measure.schwarz_christoffel_exact(ctrace) # magic happens here
        if cut:
            sc_ctrace = sc_ctrace[:cut] 
        x, y = np.real(sc_ctrace)*rescale, np.imag(sc_ctrace)*rescale # realisation and rescaling
        transformed_X_ALL.append(x)
        transformed_Y_ALL.append(y)
    return transformed_X_ALL, transformed_Y_ALL

def cot(x):
    return 1/math.tan(x)


def P_k(k,phi):
    return 1/2+(1/(math.pi**0.5))*gamma(4/k)/(gamma((8-k)/(2*k)))*(cot(phi))*sc.hyp2f1(1/2, 4/k, 3/2, -((cot(phi))**2))

def number_at_y0(y0,X,Y):
    X_at0=[]
    for i in range(len(Y)-1):
        if (Y[i]>y0 and Y[i+1]<y0) or (Y[i]<y0 and Y[i+1]>y0):
            X_at0.append(X[i])
    return X_at0

def Pontosaavaliar(y0,Phase,n1):
    X_S=[]
    Y_S=np.linspace(y0,y0,n1)
    for i in range(len(Y_S)):
        X_S.append(Y_S[i]/np.tan(Phase[i]))
    return X_S,Y_S

def leftorright(y0,X_S,Y_S,X,Y):
    X_at0=number_at_y0(y0,X,Y)
    scoreleft=np.zeros(len(X_S))
    for i,xs in enumerate(X_S):
        n=0
        for x0 in X_at0:
            if xs>x0:
                n+=1
            if xs==x0:
                p=random.random()
                if p<0.5:
                    n+=1
        if n%2==0:
            scoreleft[i]=0
        else:
            scoreleft[i]=1
    return scoreleft

def simulate_P_list(X_ALL_loops,Y_ALL_loops,y0):
    t=0
    Score=[]
    n1=30
    PHASE=np.linspace(0.01,3.1315,n1)
    X_S,Y_S=Pontosaavaliar(y0,PHASE,n1)

    for X,Y in zip(X_ALL_loops,Y_ALL_loops):
        n=random.random()
        if n<0.5:
            X=-X
        scoreleft=leftorright(y0,X_S,Y_S,X,Y)
        t+=1
        Score.append(scoreleft)
        if t>999:
            break
    return PHASE,np.array(Score)

def g_t(z, x, y):
    return ne.evaluate("1j * sqrt(-(z - x) ** 2 - y * y)")

def directsle(X,Y,t_max):
    z=[]
    for x,y in zip(X,Y):
        z.append(complex(x,y))
    z=np.array(z)
    u_list=[]
    t_list=[]
    u=0
    t=0
    for i, w in enumerate(z[1:], start=1):
        z[i + 1:] = g_t(z[i + 1:], w.real, w.imag)
        u=u+w.real
        t=t+w.imag ** 2 * 0.25
        u_list.append(u)
        t_list.append(t)
        if t>t_max:
            break
    
    return t_list, u_list
"""
def noise_plot(X, Y, t_max):

    T_sample=[]
    W_T_sample=[]

    ctraces = X + 1j*Y
    for ctrace in ctraces:
        drive, drivetime = sle.measure.driving_function(ctrace)
        cut = np.min(np.argwhere(drivetime>t_max))
        T_sample.append(drivetime[:cut])
        W_T_sample.append(drive[:cut])
        
    return np.array(pad_ragged(T_sample)),np.array(pad_ragged(W_T_sample))
"""
def noise_plot(loops_to_save_half_plane,t_max):
    X_curve_final,Y_curve_final=loops_to_save_half_plane
    T_sample=[]
    W_T_sample=[]

    for X,Y in zip(X_curve_final,Y_curve_final):

        
        T,W_T=directsle(X,Y,t_max)
        if T[-1]>t_max: # ? 
            T_sample.append(T)
            W_T_sample.append(W_T)
    return np.array(pad_ragged(T_sample)),np.array(pad_ragged(W_T_sample))


def c_auxiliar(a,b,c,d,e):
    return (a-b*c)/(((d-b**2)*(e-c**2))**0.5)


def c(Noise_lists,t,tau):
 
    List__t_tau=[]
    List__t=[]
    List__t__t=[]
    List__t_tau__t_tau=[]
    List__t__t_tau=[]
    
    for lst in Noise_lists:
        Noise_list_intervals=np.diff(lst)
        interalo__t=Noise_list_intervals[t]
        interalo__t_tau=Noise_list_intervals[t+tau]
    
        List__t_tau.append(interalo__t_tau)
        List__t.append(interalo__t)
        List__t__t.append(interalo__t**2)
        List__t_tau__t_tau.append(interalo__t_tau**2)
        List__t__t_tau.append(interalo__t_tau*interalo__t)
    
    Avg__t_tau=np.mean(List__t_tau)
    Avg__t=np.mean(List__t)
    Avg__t__t=np.mean(List__t__t)
    Avg__t_tau__t_tau=np.mean(List__t_tau__t_tau)
    Avg__t__t_tau=np.mean(List__t__t_tau)

    c=c_auxiliar(Avg__t__t_tau,Avg__t_tau,Avg__t,Avg__t_tau__t_tau,Avg__t__t)

    return c

def c_different_tau(t,lista_tau,Noise_lists):
    c_list=[]

    for tau in lista_tau:
        c1=c(Noise_lists,t,tau)
        c_list.append(c1)
    return c_list


def c_different_tau_average_in_t(list_t,lista_tau,Noise_lists):
    c_toT=[]
    c_list_final=np.zeros((len(lista_tau),))
    Error=np.zeros((len(lista_tau),))
    for t in list_t:
        c_list=c_different_tau(t,lista_tau,Noise_lists)
        c_list_final=c_list_final+c_list
        c_toT.append(c_list)
    c_toT_Trans=np.array(c_toT).T
    for i in range(len(Error)):
        Error[i]=sem(c_toT_Trans[i])
    return Error,c_list_final/len(list_t)

def W_tdescretize(T,W_T,T_save):
    W_Tnew=np.zeros(len(T_save))
   
    for k in range(len(T_save)):
        t=T_save[k]
        for j in range(len(T)-1):
            if T[j]<=t<T[j+1]:
                W_Tnew[k]=W_T[j]+((W_T[j+1]-W_T[j])/(T[j+1]-T[j]))*(t-T[j])
    return W_Tnew

def descretize_sample(T_sample,W_T_sample,T_save):
    W_T_sample_dis=[]

    for T,W_T in zip(T_sample,W_T_sample):
        W_Tnew=W_tdescretize(T,W_T,T_save)
        W_T_sample_dis.append(W_Tnew)
    return W_T_sample_dis


if __name__ == '__main__':

    ##OPTIONS
    multi = True # use multiprocessing?

    clean = False # clean out directories (fresh calculation)
    overwrite = False # overwrite existing files?
    
    do_chunks = True # subdivide long traces for statistics purposes (when low on data)

    removeloops = True
    do_inverse_schwarz_christoffel = False
    rescale = 100 # how to rescale trace after inverse schwarz christoffel


    #type_anal - will look into folders with this name (so you could just as well look at pressure or anything)
    type_anal = 'vorticity'
    #type_anal = 'pressure'  
    #type_anal = 'binary_vorticity'

    ##FILE ORIGIN
    # files should be as foldername/{type_anal}/*.npy 
    # glob folder/* will take all files in folder
    # otherwise do names = ['folder'] to do individual folders

    print('beginning globbing', flush=True)

    #names = glob.glob('/groups/astro/rsx187/isolinescaling/pressuredata/simon_data/*/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/pressuredata/simon_data/isc_nematic_simulation2048/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/simon_data/isc*/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/olgadata/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/isc_olgadata/*')
    names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/olgadata_tol/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/isc_olgadata_tol/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/olgadata_ntol/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/iscolgadata_ntol/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/polar/iscL2048/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/polar/L2048_gam2_isc/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/q_sample/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/q_sample_isc/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/uq_scanq_isc/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/backofen/iscAllForcingsFromPaper/*')
    #
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/grf3/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/pressuredata/colloids_sourav/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/varunBactTurbv2/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/Valeriia_tracking/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/smukherjee/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/mitomyocinC/*/*')

    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/vorticitydata/wensink2012/3d_data_piv/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/data-nematic')
    #names = glob.glob('/groups/astro/adoostmo/projects/SLE/data-nematic/')
    #names = glob.glob('/groups/astro/adoostmo/projects/SLE_Lasse/compressibleAN/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/vorticitydata/Stress_Density_PIV_Tracking/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/vorticitydata/guangyin/Oxygen_induced_turbulence_transision/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/vorticitydata/guangyin/Antibiotic_Induced_Turbulence_Transition/*')

    #names = ['/lustre/astro/rsx187/isolinescalingdata/vorticitydata/test_rem_loops_and_isc/critical_percolation_remove_loops_isc/']
    #names = ['/lustre/astro/rsx187/isolinescalingdata/vorticitydata/test_rem_loops_and_isc/critical_percolation_isc/']

    # checks and cleaning
    assert len(names)>0, f'0 names: {names}'
    names = [name for name in names if 'README' not in name]
    names = [name for name in names if '.ipynb' not in name]
    names = [name for name in names if 'perimvsgyrout' not in name]

    if not do_inverse_schwarz_christoffel:
        assert 'isc' not in names[0], f'isc in chosen files but do_inverse_schwarz_christoffel=False!, filename: {names[0]}'
    #assert not do_inverse_schwarz_christoffel and not 'isc' in names[0], 
    #choosing specific files
    #names = [name for name in names if '2048' in name]
    #names = [name for name in names if 'counter_0' in name]
  #   choice = ['z0.001params', 'z0.1params', 'z1.2params', 'z0.0005params', 'z0.3params', 'z1.0params',
  # 'z0.003params', 'z0.05params', 'z1.1params', 'z0.2params',
  # 'z0.002params', 'z0.9params', 'z0.005params', 'z0.5params', 'z0.0001params',
  # 'z0.8params', 'z0.02params', 'z0.4params', 'z5e-05params',
  # 'z0.7params', 'z0.01params', 'z0.007params', 'z1e-05params', 'z0.6params', 'z0.0002params']
  #   names = [name for name in names if name.split('/')[-1] in choice]

    FOLDER = names
    print("folders:", FOLDER, flush=True)
    #sys.exit() 

    for f in FOLDER: # plain vorticity / pressure folders
        if f.endswith(type_anal):
            FOLDER.remove(f)
            print('removed:', f , flush=True)

    if not overwrite:
        for n in FOLDER:
            if os.path.isfile(n+'/correlation.npy'):
                FOLDER.remove(n)
                print(n, flush=True)
    print('n files:', len(FOLDER), flush=True)

    if clean:
        clean_func(FOLDER) # clean out dest folders
    #sys.exit()
    print('start', flush=True)

    if not multi:
        #get trace
        for folder in FOLDER:
            Trace=SLE_Trace([folder, type_anal], removeloops=removeloops)
            # obviously the traces are not all the same length so np.array() doesn't work
            # I'm going to just pad with nan lets see.
            
            arTrace = np.array(trace_pad(Trace)) 
            np.save(str(folder)+"/SLE_Trace.npy",arTrace)
            del Trace
            del arTrace

        print('done traces', flush=True)
        #DF
        for folder in FOLDER:
           #print(folder)
           X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
           L_stick,num_sticks_final=df_first(X_ALL_loops,Y_ALL_loops)
           np.save(str(folder)+"/df.npy",np.array([L_stick, *num_sticks_final]))
           
           del X_ALL_loops,Y_ALL_loops
        print('done DF', flush=True)

        CUT=[400,400,400,400]
        #DF*
        for folder,cut in zip(FOLDER,CUT):
            #print(folder)
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            L_stick,num_sticks_final=df_last(X_ALL_loops,Y_ALL_loops,cut)
            np.save(str(folder)+"/df_last.npy",np.array([L_stick, *num_sticks_final]))
            del X_ALL_loops,Y_ALL_loops
        print('done DF*', flush=True)

        cut = 1000
        

        #winding varianve with time
        for folder in FOLDER:
           loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
           L_list,THETA_list=winding_statistics(loops_to_save, cut=cut, do_chunks=do_chunks)
           np.save(str(folder)+"/winding.npy",np.array([L_list,*THETA_list]))
           del loops_to_save
        print('done wind', flush=True)

        for folder in FOLDER:
           L_list=[80,500]
           for L in L_list:
               loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
               L,THETA_forL=winding_at_L(loops_to_save,L)
               np.save(str(folder)+"/windingmormaml_L="+str(L)+".npy",np.array([L,*THETA_forL]))
               del loops_to_save
               del THETA_forL
        print('done wind pdf', flush=True)

        for folder in FOLDER:
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            if do_inverse_schwarz_christoffel:
                # for a square
                #ly = np.nanmax(Y_ALL_loops[0])+1
                #lx = ly
                # hack for a rectangle
                ly = np.nanmax(Y_ALL_loops[0])+1
                lx = (np.nanmax(X_ALL_loops[0])+1)*2
                lx = ly = max(lx, ly)
                X_ALL_loops,Y_ALL_loops = inverse_sc_all(X_ALL_loops,Y_ALL_loops, lx, ly, cut=False, rescale=rescale)
            Y_0=[]
            for i in range (2,10,1):
                Y_0.append(i+0.1)
            Score_final=[]
            for y0 in Y_0:
                PHASE,Score=simulate_P_list(X_ALL_loops,Y_ALL_loops,y0)
                Score_final.extend(Score)

            np.save(str(folder)+"/Left_passage.npy",np.array([PHASE,*Score_final]))
            del PHASE,Score_final
            del X_ALL_loops
            del Y_ALL_loops
        print('done left passage', flush=True)

        for folder in FOLDER:
            loops_to_save_half_plane=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            if do_inverse_schwarz_christoffel:
                X_ALL_loops,Y_ALL_loops = loops_to_save_half_plane
                # for a square
                #ly = np.nanmax(Y_ALL_loops[0])+1
                #lx = ly
                # hack for a rectangle
                ly = np.nanmax(Y_ALL_loops[0])+1
                lx = (np.nanmax(X_ALL_loops[0])+1)*2
                lx = ly = max(lx, ly)
                X_ALL_loops,Y_ALL_loops = inverse_sc_all(X_ALL_loops,Y_ALL_loops, lx, ly, cut=False, rescale=rescale)
                loops_to_save_half_plane = X_ALL_loops,Y_ALL_loops
            t_max=1000
            T_sample, W_T_sample= noise_plot(loops_to_save_half_plane,t_max)

            np.save(str(folder)+"/noise.npy",np.array([T_sample,W_T_sample]))
            del T_sample,W_T_sample,loops_to_save_half_plane
        print('done drive', flush=True)

        for folder in FOLDER:
            list_t=np.arange(0,20,1)
            lista_tau=np.arange(0,19,1)
            T_sample,W_T_sample=np.load(str(folder)+"/noise.npy",allow_pickle=True)
            
            t_max_save=1000
            T_save=np.linspace(0,t_max_save,50)

            W_T_sample_dis=descretize_sample(T_sample,W_T_sample,T_save)
            
            
            Error2,c_list_final=c_different_tau_average_in_t(list_t,lista_tau,W_T_sample_dis) 
            res = np.array(pad_ragged([T_save,Error2,c_list_final]))
            np.save(str(folder)+"/correlation.npy", res)
        print('done correlation', flush=True)


    else:
        ncpu=min(int(os.environ['SLURM_CPUS_PER_TASK']), len(FOLDER))
        print(f'ncpu: {ncpu}', flush=True)
        def do_trace(foldertypeanal):
            #print(foldertypeanal, flush=True)
            folder, type_anal = foldertypeanal
            if not overwrite:
                if os.path.isfile(str(folder)+"/SLE_Trace.npy"):
                    return

            Trace=SLE_Trace(foldertypeanal, removeloops=removeloops)
            #print(folder, flush=True)
            # obviously the traces are not all the same length so np.array() doesn't work
            # I'm going to just pad with nan lets see.
            #print(Trace, flush=True)
            try:
                arTrace = np.array(trace_pad(Trace)) 
            except ValueError:
                print(f'ValueError for padding {folder} trace, len trace {len(Trace)}', flush=True) # why? - no traces?
                return folder
            #arTrace = np.array([0])
            #FOLDER.remove(folder)
            print(f'done {folder}', flush=True)
            np.save(str(folder)+"/SLE_Trace.npy",arTrace)
            #del Trace
            #del arTrace

        with Pool(ncpu) as p:
            res = p.map(do_trace, zip(FOLDER, [type_anal]*len(FOLDER)))
        if len(res)>0: #what is going on here
            for folder in res:
                try:
                    FOLDER.remove(folder)
                except:
                    pass
        print('done SLE', flush=True)

        def do_DF(folder):
            if not overwrite:
                if os.path.isfile(str(folder)+"/df.npy"):
                    return
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            L_stick,num_sticks_final=df_first(X_ALL_loops,Y_ALL_loops)
            np.save(str(folder)+"/df.npy",np.array([L_stick, *num_sticks_final]))
            #del X_ALL_loops,Y_ALL_loops
        with Pool(ncpu) as p:
            p.map(do_DF, FOLDER)
        print('done DF', flush=True)


        #CUT=[400,400,400,400]
        #DF*
        #def do_DFstar(folder, cut):
        #    if not overwrite:
        #        if os.path.isfile(str(folder)+"/df_last.npy"):
        #            return
        #    X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
        #    L_stick,num_sticks_final=df_last(X_ALL_loops,Y_ALL_loops,cut)
        #    np.save(str(folder)+"/df_last.npy",np.array([L_stick, *num_sticks_final]))
            #del X_ALL_loops,Y_ALL_loops
        #with Pool(ncpu) as p:
        #    p.starmap(do_DFstar, zip(FOLDER, CUT))
        #print('done DF*')

        cut = 1000
        def do_winding(folder, cut, do_chunks):
            if not overwrite:
                if os.path.isfile(str(folder)+"/winding.npy"):
                    return
            loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            L_list,THETA_list=winding_statistics(loops_to_save, cut, do_chunks)
            np.save(str(folder)+"/winding.npy",np.array([L_list,*THETA_list]))
            #del loops_to_save
        with Pool(ncpu) as p:
            p.starmap(do_winding, zip(FOLDER, np.repeat(cut, len(FOLDER)), np.repeat(do_chunks, len(FOLDER)) ))
        print('done winding', flush=True)

        #winding gaussianity

        def do_winding_pdf(folder):
            L_list=[80,500]
            for L in L_list:
                if not overwrite:
                    if os.path.isfile(str(folder)+"/windingmormaml_L="+str(L)+".npy"):
                        continue
                loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
                L,THETA_forL=winding_at_L(loops_to_save,L)
                np.save(str(folder)+"/windingmormaml_L="+str(L)+".npy",np.array([L,*THETA_forL]))
                #del loops_to_save
                #del THETA_forL
        with Pool(ncpu) as p:
            p.map(do_winding_pdf, FOLDER)
        print('done winding pdf', flush=True)

        #left passage

        def do_left_passage(folder):
            if not overwrite:
                if os.path.isfile(str(folder)+"/Left_passage.npy"):
                    return
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            if do_inverse_schwarz_christoffel:
                # for a square
                #ly = np.nanmax(Y_ALL_loops[0])+1
                #lx = ly
                # hack for a rectangle
                ly = np.nanmax(Y_ALL_loops[0])+1
                lx = (np.nanmax(X_ALL_loops[0])+1)*2
                lx = ly = max(lx, ly)
                X_ALL_loops,Y_ALL_loops = inverse_sc_all(X_ALL_loops,Y_ALL_loops, lx, ly, cut=False, rescale=rescale)
            Y_0=[]
            if do_inverse_schwarz_christoffel:
                yrange = range(3,1000,100)
            else:
                yrange = range(2,10,1)
            for i in yrange:
                Y_0.append(i+0.1)
            Score_final=[]
            for y0 in Y_0:
                PHASE,Score=simulate_P_list(X_ALL_loops,Y_ALL_loops,y0)
                Score_final.extend(Score)

            np.save(str(folder)+"/Left_passage.npy",np.array([PHASE,*Score_final]))
            #del PHASE,Score_final
            #del X_ALL_loops
            #del Y_ALL_loops
        with Pool(ncpu) as p:
            p.map(do_left_passage, FOLDER)
        print('done left passage', flush=True)

        # drive and drivetimes
       
        def do_driving(folder):
            if not overwrite:
                if os.path.isfile(str(folder)+"/noise.npy"):
                    return
            loops_to_save_half_plane=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            if do_inverse_schwarz_christoffel:
                X_ALL_loops,Y_ALL_loops = loops_to_save_half_plane
                # for a square
                #ly = np.nanmax(Y_ALL_loops[0])+1
                #lx = ly
                # hack for a rectangle
                ly = np.nanmax(Y_ALL_loops[0])+1
                lx = (np.nanmax(X_ALL_loops[0])+1)*2
                lx = ly = max(lx, ly)
                X_ALL_loops,Y_ALL_loops = inverse_sc_all(X_ALL_loops, Y_ALL_loops, lx, ly, cut=False, rescale=rescale)
                loops_to_save_half_plane = X_ALL_loops,Y_ALL_loops
            t_max=1000
            T_sample, W_T_sample= noise_plot(loops_to_save_half_plane,t_max)

            np.save(str(folder)+"/noise.npy",np.array([T_sample,W_T_sample]))
            #del T_sample,W_T_sample,loops_to_save_half_plane
        with Pool(ncpu) as p:
            p.map(do_driving, FOLDER)
        print('done drive', flush=True)


        # correlation time

        def do_correlation(folder):
            if not overwrite:
                if os.path.isfile(str(folder)+"/correlation.npy"):
                    return
            list_t=np.arange(0,20,1)
            lista_tau=np.arange(0,19,1)
            T_sample,W_T_sample=np.load(str(folder)+"/noise.npy",allow_pickle=True)
            
            t_max_save=1000
            T_save=np.linspace(0,t_max_save,50)

            W_T_sample_dis=descretize_sample(T_sample,W_T_sample,T_save)
            
            
            Error2,c_list_final=c_different_tau_average_in_t(list_t,lista_tau,W_T_sample_dis) 
            res = np.array(pad_ragged([T_save,Error2,c_list_final]))
            np.save(str(folder)+"/correlation.npy", res)

        with Pool(ncpu) as p:
            p.map(do_correlation, FOLDER)

        print('done correlation', flush=True)
