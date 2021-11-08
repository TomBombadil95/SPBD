import numpy as np
import cvxpy as cp

#This function allows us to choose the shape we want to consider from the Ground-Truth and the Routing Matrix
def set_data(Q,R,start,end,time0, time1, verb = True):
    Q = Q[time0:time1, start:end]
    R = R[:,start:end]
    if verb:
        print('dim R: ',R.shape,'\ndim Q: ', Q.shape)
    return Q,R

#This function allows us to create the LinkCount matrix and the Sampling matrix to get the Flow-level measurement matrix 
def preprocess(Q, R, verb = True):
    Y =  R @ Q.T
    T = Q.shape[0]; F = Q.shape[1]; N = F*T
    perc = 0.1
    m = round(N*perc)
    P = np.zeros([F,T])
    P = P.flatten()
    samp = np.random.randint(1 , N , size = m)
    Q  = Q.T.flatten()
    P[samp] = 1
    Z = cp.multiply(P, Q)
    Z = Z.value.reshape(F,T) ; Q = Q.reshape(F,T); P = P.reshape(F,T)
    if verb:
        print('Link counts matrix has shape: ', Y.shape)
        print('Total flow volumes: N = FxT = ', F, 'x', T, '=', N, '\nI am going to measure directly only', m, 'samples from the original dataset')
        print('sampling matrix P = ', P.shape,'\noriginal traffic matrix Q = ', Q.shape, '\nflow-level-measures matrix Z = ', Z.shape) 
    return F,T,P,Q,R,Z,Y

#This functions perform the Principal Component Pursuit taking as arguments the shape of interest for the components, matrices to work out
#the optimization and minimum & maximum for the penalty lambda (default between 0.1-0.3)
def PCP(F,T,P,Q,R,Z,Y, min_lin = 0.1, max_lin = 0.3):
    anom = []; nomin = []
    A = cp.Variable((F,T)); X = cp.Variable((F,T)); G = X+A
    constraints = [Z == cp.multiply(P,G),    Y == R@G, A >= 0, X >= 0]
    lamb =  cp.Parameter(nonneg=True)
    objective = cp.Minimize(cp.norm(X,'nuc') + (lamb*cp.norm1(A)))
    prob = cp.Problem(objective, constraints)

    aval = []; xval = []; res = []
    lamb_vals = np.linspace(min_lin,max_lin, num = 10) #CVXPY solves the problem with different Lambdas
    for val in lamb_vals:
        lamb.value = val                                   
        prob.solve(solver=cp.SCS, eps = 1e-7)                                       

        aval.append(A.value)
        xval.append(X.value)
        res.append(G.value)
    all_MSE = {}
    for pos in range(10):
        MSE =  cp.sum_squares(Q - res[pos])/cp.sum_squares(Q)  #we choose the best lambda based on MSE
        all_MSE[pos] = MSE.value
    top =  min(all_MSE, key=all_MSE.get)
    anom.append(aval[top])
    nomin.append(xval[top]) 
    print('List Updated! Top minimum: {:0.3e} , correspondent to lambda = '.format(all_MSE[top]), lamb_vals[top])
    anomalies = np.hstack(anom)
    nominal = np.hstack(nomin)
    tot = nominal + anomalies
    return anomalies, nominal, tot

