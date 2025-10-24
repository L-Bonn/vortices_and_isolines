import numpy as np
import schramm_loewner_evolution as sle

from scipy.special import gamma
import scipy.special as sc


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
                p=np.random.random()
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
    n1=100
    PHASE=np.linspace(0.01,3.1315,n1)
    X_S,Y_S=Pontosaavaliar(y0,PHASE,n1)

    for X,Y in zip(X_ALL_loops,Y_ALL_loops):
        n=np.random.random()
        if n<0.5:
            X=-X
        scoreleft=leftorright(y0,X_S,Y_S,X,Y)
        t+=1
        Score.append(scoreleft)
        if t>999:
            break
    return PHASE,np.array(Score)


if __name__ == '__main__':
	traces = []
	lx = ly = 1_000
	doSC = True
	for i in range(50):
	    fieldbef = np.random.uniform(size=(lx, ly))<0.592
	    field = draw_border(fieldbef, cut=None, y_realline=0)
	    px = np.floor(lx/2).astype(int)
	    py = 0
	    px = int(px)
	    py = int(py)
	    trace = sle.measure.trace_contour(field, pixel_init=(px-1, py))
	    if doSC
		    centretrace = trace - trace[0]
		    #traces.append(centretrace)
		    centretrace /= np.array([lx, ly])
		    ctrace = centretrace[:, 0] + 1j*centretrace[:, 1]
		    
		    sc_ctrace = sle.measure.schwarz_christoffel_exact(ctrace)
		    traces.append(np.real(sc_ctrace), np.imag(sc_ctrace))
	    else:
	    	traces.append([trace[:,0], trace[:1]])

	Y_0=[]
	for i in range (2,10,1):
	    Y_0.append(i+0.1)
	Score_final=[]
	for y0 in Y_0:
	    PHASE,Score=simulate_P_list(X_ALL_loops,Y_ALL_loops,y0)
	    Score_final.extend(Score)