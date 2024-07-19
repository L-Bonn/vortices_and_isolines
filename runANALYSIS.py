# making Francisco's ipynb into a python script

import numpy as np
import glob
from skimage import measure
import random
from scipy.stats import norm,sem
import matplotlib.pyplot as plt
import matplotlib
import math
import cv2 
from collections import deque
import numexpr as ne
import scipy
from scipy import optimize
from mpmath import *
import mpmath
from mpmath import cot
from numpy import ndarray
from scipy.special import gamma
import scipy.special as sc
import os
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

def rotate_matrix(matrix, degrees):
    if degrees == 0:
        return matrix
    elif degrees == 90:
        return np.rot90(matrix, k=1)
    elif degrees == 180:
        return np.rot90(matrix, k=2)
    elif degrees == 270:
        return np.rot90(matrix, k=3)

def SLE_Trace(foldertypeanal):
    folder, type_anal = foldertypeanal
    X_curve_final=[]
    Y_curve_final=[]

    Data_list=glob.glob(str(folder)+f"/{type_anal}/*")
    for i,link in enumerate(Data_list): # for file
        
        data=np.load(link)
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
                    X_curve_final.append(counter[:, 1]-counter[0, 1])
                    Y_curve_final.append(counter[:, 0])
            del indices,sorted_list,contours
   
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

def noise_plot(loops_to_save_half_plane,t_max):
    X_curve_final,Y_curve_final=loops_to_save_half_plane
    T_sample=[]
    W_T_sample=[]

    for X,Y in zip(X_curve_final,Y_curve_final):

        
        T,W_T=directsle(X,Y,t_max)
        if T[-1]>t_max:
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

    type_anal = 'vorticity',#pressure'#binary_vorticity' # pressure
    #names = glob.glob('/groups/astro/rsx187/isolinescaling/pressuredata/simon_data/*/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/pressuredata/simon_data/*/*')
    #names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/{type_anal}data/simon_data/*/*')
    names = glob.glob(f'/lustre/astro/rsx187/isolinescalingdata/vorticitydata/grf/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/vorticitydata/wensink2012/3d_data_piv/*')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/data-nematic')
    #names = glob.glob('/lustre/astro/rsx187/isolinescalingdata/vorticitydata/Stress_Density_PIV_Tracking/*')
    assert len(names)>0, f'0 names: {names}'
    names = [name for name in names if 'README' not in name]
    FOLDER = names
    #FOLDER = [n for n in names if '0.04_' in n]
    #
    #FOLDER = [n for n in FOLDER if 'counter_4' in n]
    #FOLDER =[n for n in FOLDER if '2048' in n] 


    print(FOLDER)
    multi = True

    clean = True
    overwrite = False
    
    do_chunks = True

    for f in FOLDER: # plain vorticity / pressure folders
        if f.endswith(type_anal):
            FOLDER.remove(f)
            print('removed:', f )

    if not overwrite:
        for n in FOLDER:
            if os.path.isfile(n+'/correlation.npy'):
                FOLDER.remove(n)
                print(n)
    print('n files:', len(FOLDER))

    if clean:
        clean_func(FOLDER) # clean out dest folders
    #sys.exit()
    print('start')

    if not multi:
        #get trace
        for folder in FOLDER:
            Trace=SLE_Trace(folder)
            # obviously the traces are not all the same length so np.array() doesn't work
            # I'm going to just pad with nan lets see.
            
            arTrace = np.array(trace_pad(Trace)) 
            np.save(str(folder)+"/SLE_Trace.npy",arTrace)
            del Trace
            del arTrace

        print('done traces')
        #DF
        for folder in FOLDER:
           #print(folder)
           X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
           L_stick,num_sticks_final=df_first(X_ALL_loops,Y_ALL_loops)
           np.save(str(folder)+"/df.npy",np.array([L_stick, *num_sticks_final]))
           
           del X_ALL_loops,Y_ALL_loops
        print('done DF')

        CUT=[400,400,400,400]
        #DF*
        for folder,cut in zip(FOLDER,CUT):
            #print(folder)
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            L_stick,num_sticks_final=df_last(X_ALL_loops,Y_ALL_loops,cut)
            np.save(str(folder)+"/df_last.npy",np.array([L_stick, *num_sticks_final]))
            del X_ALL_loops,Y_ALL_loops
        print('done DF*')

        cut = 1000
        

        #winding varianve with time
        for folder in FOLDER:
           loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
           L_list,THETA_list=winding_statistics(loops_to_save, cut=cut, do_chunks=do_chunks)
           np.save(str(folder)+"/winding.npy",np.array([L_list,*THETA_list]))
           del loops_to_save
        print('done wind')

        for folder in FOLDER:
           L_list=[80,500]
           for L in L_list:
               loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
               L,THETA_forL=winding_at_L(loops_to_save,L)
               np.save(str(folder)+"/windingmormaml_L="+str(L)+".npy",np.array([L,*THETA_forL]))
               del loops_to_save
               del THETA_forL
        print('done wind pdf')

        for folder in FOLDER:
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
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
        print('done left passage')

        for folder in FOLDER:
            loops_to_save_half_plane=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            t_max=1000
            T_sample, W_T_sample= noise_plot(loops_to_save_half_plane,t_max)

            np.save(str(folder)+"/noise.npy",np.array([T_sample,W_T_sample]))
            del T_sample,W_T_sample,loops_to_save_half_plane
        print('done drive')

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
        print('done correlation')


    else:
        ncpu=min(int(os.environ['SLURM_CPUS_PER_TASK']), len(FOLDER))
        print(f'ncpu: {ncpu}')
        def do_trace(foldertypeanal):
            folder, type_anal = foldertypeanal
            if not overwrite:
                if os.path.isfile(str(folder)+"/SLE_Trace.npy"):
                    return

            Trace=SLE_Trace(foldertypeanal)
            print(folder)
            # obviously the traces are not all the same length so np.array() doesn't work
            # I'm going to just pad with nan lets see.
            try:
                arTrace = np.array(trace_pad(Trace)) 
            except ValueError:
                print(f'ValueError for padding {folder} trace, len trace {len(Trace)}') # why? - no traces?
                return folder
            #arTrace = np.array([0])
            #FOLDER.remove(folder)
            print(f'done {folder}')
            np.save(str(folder)+"/SLE_Trace.npy",arTrace)
            #del Trace
            #del arTrace

        with Pool(ncpu) as p:
            res = p.map(do_trace, zip(FOLDER, type_anal*len(FOLDER)))
        if len(res)>0: #what is going on here
            for folder in res:
                try:
                    FOLDER.remove(folder)
                except:
                    pass
        print('done SLE')

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
        print('done DF')


        CUT=[400,400,400,400]
        #DF*

        def do_DFstar(folder, cut):
            if not overwrite:
                if os.path.isfile(str(folder)+"/df_last.npy"):
                    return
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            L_stick,num_sticks_final=df_last(X_ALL_loops,Y_ALL_loops,cut)
            np.save(str(folder)+"/df_last.npy",np.array([L_stick, *num_sticks_final]))
            #del X_ALL_loops,Y_ALL_loops
        with Pool(ncpu) as p:
            p.starmap(do_DFstar, zip(FOLDER, CUT))
        print('done DF*')

        cut = 1000
        def do_winding(folder, cut, do_chunks):
            loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            L_list,THETA_list=winding_statistics(loops_to_save, cut, do_chunks)
            np.save(str(folder)+"/winding.npy",np.array([L_list,*THETA_list]))
            #del loops_to_save
        with Pool(ncpu) as p:
            p.starmap(do_winding, zip(FOLDER, np.repeat(cut, len(FOLDER)), np.repeat(do_chunks, len(FOLDER)) ))
        print('done winding')

        #winding gaussianity

        def do_winding_pdf(folder):
            L_list=[80,500]
            for L in L_list:
                loops_to_save=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
                L,THETA_forL=winding_at_L(loops_to_save,L)
                np.save(str(folder)+"/windingmormaml_L="+str(L)+".npy",np.array([L,*THETA_forL]))
                #del loops_to_save
                #del THETA_forL
        with Pool(ncpu) as p:
            p.map(do_winding_pdf, FOLDER)
        print('done winding pdf')

        #left passage

        def do_left_passage(folder):
            X_ALL_loops,Y_ALL_loops=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            Y_0=[]
            for i in range (2,10,1):
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
        print('done left passage')

        # drive and drivetimes
       
        def do_driving(folder):
            loops_to_save_half_plane=np.load(str(folder)+"/SLE_Trace.npy",allow_pickle=True)
            t_max=1000
            T_sample, W_T_sample= noise_plot(loops_to_save_half_plane,t_max)

            np.save(str(folder)+"/noise.npy",np.array([T_sample,W_T_sample]))
            #del T_sample,W_T_sample,loops_to_save_half_plane
        with Pool(ncpu) as p:
            p.map(do_driving, FOLDER)
        print('done drive')


        # correlation time

        def do_correlation(folder):
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

        print('done correlation')
