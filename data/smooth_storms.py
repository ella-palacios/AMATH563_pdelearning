import sys
sys.path.append('..')
import numpy as np

def kermatrix_storm(X,l = 1):
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

def smooth_one_storm_rbf(t,u,l=1,lam=.1):
    
    K = kermatrix_storm(t,l)
    regvec1 = np.linalg.solve(K + lam*np.eye(K.shape[0]),u[:,0].astype('float'))
    regvec2 = np.linalg.solve(K + lam*np.eye(K.shape[0]),u[:,1].astype('float'))
    
    LAT = np.zeros([len(t),4])
    LON = np.zeros([len(t),4])
    
    for j in range(len(t)):
        v = K[j,:]
        v_t = -((t[j] - t)/(l**2))*v
        v_tt = (-1/(l**2) + ((t[j] - t)/(l**2))**2)*v
        
        LAT[j,0] = t[j]
        LAT[j,1] = np.inner(v,regvec1)
        LAT[j,2] = np.inner(v_t,regvec1)
        LAT[j,3] = np.inner(v_tt,regvec1)
        LON[j,0] = t[j]
        LON[j,1] = np.inner(v,regvec2)
        LON[j,2] = np.inner(v_t,regvec2)
        LON[j,3] = np.inner(v_tt,regvec2)
    
    return LAT,LON

def polymatrix_storm(t,c = 15,a = 3):
    J = t.shape[0]
    kermatrix = np.zeros([J,J])
    for i in range(J):
        for j in range(J):
            if j<i:
                kermatrix[i,j] = kermatrix[j,i]
            else:
                kermatrix[i,j] = (c + np.inner(t[i],t[j]))**a
    return kermatrix

def smooth_one_storm_poly(t,u,c=15,a=3,lam=10**-4):
    
    K = polymatrix_storm(t,c,a)
    regvec1 = np.linalg.solve(K + lam*np.eye(K.shape[0]),u[:,0].astype('float'))
    regvec2 = np.linalg.solve(K + lam*np.eye(K.shape[0]),u[:,1].astype('float'))
    
    LAT = np.zeros([len(t),4])
    LON = np.zeros([len(t),4])
    
    for j in range(len(t)):
        v = K[j,:]
        v_t = a*t*(v**((a-1)/a))
        v_tt = a*(a-1)*(t**2)*(v**((a-2)/a))
        
        LAT[j,0] = t[j]
        LAT[j,1] = np.inner(v,regvec1)
        LAT[j,2] = np.inner(v_t,regvec1)
        LAT[j,3] = np.inner(v_tt,regvec1)
        LON[j,0] = t[j]
        LON[j,1] = np.inner(v,regvec2)
        LON[j,2] = np.inner(v_t,regvec2)
        LON[j,3] = np.inner(v_tt,regvec2)
    
    return LAT,LON

def smooth_one_storm_rbf_withinv(t,u,l=1,lam=10**-4):
    
    K = kermatrix_storm(t,l)
    regvec1 = np.linalg.solve(K + lam*np.eye(K.shape[0]),u[:,0].astype('float'))
    regvec2 = np.linalg.solve(K + lam*np.eye(K.shape[0]),u[:,1].astype('float'))
    
    LAT = np.zeros([len(t),4])
    LON = np.zeros([len(t),4])
    t_lon = np.zeros([len(t),1])
    t_lat = np.zeros([len(t),1])
    
    for j in range(len(t)):
        v = K[j,:]
        v_t = -((t[j] - t)/(l**2))*v
        v_tt = (-1/(l**2) + ((t[j] - t)/(l**2))**2)*v
        
        LAT[j,0] = t[j]
        LAT[j,1] = np.inner(v,regvec1)
        LAT[j,2] = np.inner(v_t,regvec1)
        LAT[j,3] = np.inner(v_tt,regvec1)
        LON[j,0] = t[j]
        LON[j,1] = np.inner(v,regvec2)
        LON[j,2] = np.inner(v_t,regvec2)
        LON[j,3] = np.inner(v_tt,regvec2)
        t_lon[j] = 1/LAT[j,2] #this is because the others are flipped, this is the correct labels
        t_lat[j] = 1/LON[j,2] #same here
    
    return LAT,LON,t_lon,t_lat


# #code for to generate figures
# S_LAT = []
# S_LON = []

# for j in range(len(u_ids)):
#     t = ts[ids == u_ids[j]].astype('float')
#     u = us[ids == u_ids[j],:].astype('float')
#     LAT,LON = smooth_one_storm_poly(t,u)
    
#     S_LAT.append(np.copy(LAT))
#     S_LON.append(np.copy(LON))

# ## WANT TO FORMAT LAT AND LON (#STORMS, MAX # TIMES)
# ##valid storm ids is just u_ids

# m = 0
# for j in range(len(u_ids)):
#     m = max(m,np.sum(ids == u_ids[j]))
    
# LAT_plot = np.ones([len(u_ids),m])*np.nan
# LON_plot = np.ones([len(u_ids),m])*np.nan

# for j in range(len(u_ids)):
#     LAT_plot[j,range(len(S_LON[j][:,1]))] = S_LON[j][:,1]
#     LON_plot[j,range(len(S_LAT[j][:,1]))] = S_LAT[j][:,1]
    
# plot_trajectories(u_ids, LON_plot, LAT_plot)
