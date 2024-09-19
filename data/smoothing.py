import numpy as np

#Throughout this the assumption is that:
# u is an IxJ matrix with ij-th entry u^{(i)}(x_j)
# X is a Jx2 matrix with jth row (x_j,t_j) for the berger's eqn
# l is the timescale for the RBF kernel
# lambda is the nugget parameter
# the default value they take is the one they used in the paper

def kermatrix_berg_rbf(X,l = 1,lam = 10**-8):
    #this just gives you K(X,X) for the RBF kernel w/ timescale l
    J = X.shape[0]
    kermatrix = np.zeros([J,J])
    for i in range(J):
        for j in range(J):
            if j<i:
                kermatrix[i,j] = kermatrix[j,i]
            else:
                kermatrix[i,j] = np.exp((-1/(2*(l**2)))*np.inner(X[i,:]-X[j,:],X[i,:]-X[j,:]))
    return kermatrix

def kermatrix_berg_poly(X,c = 1,lam = 10**-8, a=5):
    J = X.shape[0]
    kermatrix = np.zeros([J,J])
    for i in range(J):
        for j in range(J):
            if j<i:
                kermatrix[i,j] = kermatrix[j,i]
            else:
                kermatrix[i,j] = (c + np.inner(X[i,:],X[j,:]))**a
    return kermatrix

def smooth_berg(X,u,l = 1,lam = 10**-8, ker='rbf'):
    #Returns the smoothed data
    #X should be J rows and have # of variables (2 for bergers) columns
    #u should have I rows and J columns with ijth entry of u^{(i)}(x_j)
    if ker == 'rbf':
        return smooth_berg_rbf(X, u, l, lam)
    elif ker=='poly':
        return smooth_berg_poly(X, u, l, lam)
    else:
        raise ValueError("Error: unexpected kernel type")

def smooth_berg_rbf(X, u, l = 1,lam = 10**-8):
    I = u.shape[0]
    J = u.shape[1]
    S = []
    K = kermatrix_berg_rbf(X,l,lam)
    for j in range(J):
        v = K[:,j]
        v_x = -((X[j,0] - X[:,0])/(l**2))*v
        v_t = -((X[j,1] - X[:,1])/(l**2))*v
        v_xt = ((X[j,0] - X[:,0])/(l**2))*((X[j,1] - X[:,1])/(l**2))*v
        v_xx = (-1/(l**2) + ((X[j,0] - X[:,0])/(l**2))**2)*v
        v_tt = (-1/(l**2) + ((X[j,1] - X[:,1])/(l**2))**2)*v
        T = np.zeros([I,8])
        for i in range(I):
            regvec = np.linalg.solve(K + lam*np.eye(J,J),u[i,:])
            T[i,0] = np.inner(v,regvec)
            T[i,1] = np.inner(v_x,regvec)
            T[i,2] = np.inner(v_t,regvec)
            T[i,3] = np.inner(v_xx,regvec)
            T[i,4] = np.inner(v_xt,regvec)
            T[i,5] = np.inner(v_tt,regvec)
            T[i,6] = X[j,0]
            T[i,7] = X[j,1]
        S.append(np.copy(T))
    #because S has 3 indicies, S should have J entries, each entry is a I by 6 matrix with ith row s_j^{(i)} 
    #ie s_j^{i} should be stored as S[j][i,:]
    return S

def smooth_berg_poly(X,u,l = 1,lam = 10**-8):
#Returns the smoothed data
    #X should be J rows and have # of variables (2 for bergers) columns
    #u should have I rows and J columns with ijth entry of u^{(i)}(x_j)
    I = u.shape[0]
    J = u.shape[1]
    S = []
    K = kermatrix_berg_poly(X,c,a)
    for j in range(J):
        v = K[:,j]
        v_x = a*X[:,0]*(v**((a-1)/a))
        v_t = a*X[:,1]*(v**((a-1)/a))
        v_xt = a*(a-1)*X[:,0]*X[:,1]*(v**((a-2)/a))
        v_xx = a*(a-1)*(X[:,0]**2)*(v**((a-2)/a))
        v_tt = a*(a-1)*(X[:,1]**2)*(v**((a-2)/a))
        T = np.zeros([I,8])
        for i in range(I):
            regvec = np.linalg.solve(K + lam*np.eye(J,J),u[i,:])
            T[i,0] = X[j,0]
            T[i,1] = X[j,1]
            T[i,2] = np.inner(v,regvec)
            T[i,3] = np.inner(v_x,regvec)
            T[i,4] = np.inner(v_t,regvec)
            T[i,5] = np.inner(v_xx,regvec)
            T[i,6] = np.inner(v_xt,regvec)
            T[i,7] = np.inner(v_tt,regvec)
        S.append(np.copy(T))
    #because S has 3 indicies, S should have J entries, each entry is a I by 6 matrix with ith row s_j^{(i)} 
    #ie s_j^{i} should be stored as S[j][i,:]
    return S

#Throughout this the assumption is that:
# u_1 is an IxJ matrix with ij-th entry u1^{(i)}(x_j)
# u_2 is an IxJ matrix with ij-th entry u2^{(i)}(x_j)
# X is a J-vector with jth entry t_j
# l is the timescale for the RBF kernel
# lambda is the nugget parameter
# the default value they take is the one they used in the paper

def kermatrix_pend_rbf(X,l = 1):
    #this just gives you K(X,X) for the RBF kernel w/ timescale l
    J = X.shape[0]
    kermatrix = np.zeros([J,J])
    for i in range(J):
        for j in range(J):
            if j<i:
                kermatrix[i,j] = kermatrix[j,i]
            else:
                kermatrix[i,j] = np.exp((-1/(2*(l**2)))*np.inner(X[i]-X[j],X[i]-X[j]))
    return kermatrix

def kermatrix_pend_poly(X, c = 1, a = 5):
    #this just gives you K(X,X) for the RBF kernel w/ timescale l
    J = X.shape[0]
    kermatrix = np.zeros([J,J])
    for i in range(J):
        for j in range(J):
            if j<i:
                kermatrix[i,j] = kermatrix[j,i]
            else:
                kermatrix[i,j] = (c + X[i]*X[j])**a
    return kermatrix

def smooth_pend(X, u1, u2, l = .1, lam = .0001, a = 5, ker='rbf'):
    #Returns the smoothed data
    #X should be J rows and have # of variables (2 for bergers) columns
    #u should have I rows and J columns with ijth entry of u^{(i)}(x_j)
    if ker == 'rbf':
        return smooth_pend_rbf(X, u1, u2, l=l, lam=lam)
    elif ker == 'poly':
        return smooth_pend_poly(X, u1, u2, c=l, lam=lam, a=a)
    else:
        raise ValueError("Error: unexpected kernel type")
    
def smooth_pend_rbf(X, u1, u2, l = .1, lam = .0001):
    #Returns the smoothed data
    #X should be J rows and have # of variables (1 for pendulum) columns
    #u should have I rows and J columns with ijth entry of u^{(i)}(x_j)
    I = u1.shape[0]
    J = u1.shape[1]
    S1 = []
    S2 = []
    K = kermatrix_pend_rbf(X,l)
    for j in range(J):
        v = K[j,:]
        v_t = -((X[j] - X)/(l**2))*v
        v_tt = (-1/(l**2) + ((X[j] - X)/(l**2))**2)*v
        T1 = np.zeros([I, 5]) #5])
        T2 = np.zeros([I, 5]) #5])
        for i in range(I):
            regvec1 = np.linalg.solve(K + lam*np.eye(J,J),u1[i,:])
            regvec2 = np.linalg.solve(K + lam*np.eye(J,J),u2[i,:])
            T1[i,0] = np.inner(v,regvec1)
            T1[i,1] = np.inner(v_t,regvec1)
            T1[i,2] = np.inner(v_tt,regvec1)
            T1[i,3] = X[j]
            T1[i,4] = np.sin(T1[i,0])
            T2[i,0] = np.inner(v,regvec2)
            T2[i,1] = np.inner(v_t,regvec2)
            T2[i,2] = np.inner(v_tt,regvec2)
            T2[i,3] = X[j]
            T2[i,4] = np.sin(T2[i,0])
        S1.append(np.copy(T1))
        S2.append(np.copy(T2))
    #because S has 3 indicies, S should have J entries, each entry is a I by 3 matrix with ith row s_j^{(i)} 
    #ie s_j^{i} should be stored as S[j][i,:]
    return S1,S2

def smooth_pend_poly(X, u1, u2, c = .3, lam = 1, a = 5):
    #Returns the smoothed data
    #X should be J rows and have # of variables (1 for pendulum) columns
    #u should have I rows and J columns with ijth entry of u^{(i)}(x_j)
    I = u1.shape[0]
    J = u1.shape[1]
    S1 = []
    S2 = []
    K = kermatrix_pend_poly(X, c, a)
    for j in range(J):
        v = K[j,:]
        v_t = a*X*(v**((a-1)/a))
        v_tt = a*(a-1)*(X**2)*(v**((a-2)/a))
        T1 = np.zeros([I,5])
        T2 = np.zeros([I,5])
        for i in range(I):
            regvec1 = np.linalg.solve(K + lam*np.eye(J,J),u1[i,:])
            regvec2 = np.linalg.solve(K + lam*np.eye(J,J),u2[i,:])
            T1[i,0] = np.inner(v,regvec1)
            T1[i,1] = np.inner(v_t,regvec1)
            T1[i,2] = np.inner(v_tt,regvec1)
            T1[i,3] = X[j]
            T1[i,4] = np.sin(T1[i,0])
            T2[i,0] = np.inner(v,regvec2)
            T2[i,1] = np.inner(v_t,regvec2)
            T2[i,2] = np.inner(v_tt,regvec2)
            T2[i,3] = X[j]
            T2[i,4] = np.sin(T2[i,0])
        S1.append(np.copy(T1))
        S2.append(np.copy(T2))
    #because S has 3 indicies, S should have J entries, each entry is a I by 4 matrix with ith row s_j^{(i)} 
    #ie s_j^{i} should be stored as S[j][i,:]
    return S1, S2
