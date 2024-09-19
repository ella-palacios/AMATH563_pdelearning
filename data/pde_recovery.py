import numpy as np
import sys
sys.path.append('..')
from data.smoothing import smooth_pend, smooth_berg
from data.pendulum import pendulum_train_data
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import KFold
from utils.kernel import identity_like_matrix

# Old version
# def rbf_vector_kernel(x1,x2,l):
#     # Input vectors x1 and x2 are (num_mesh_samples, 3, num_time_samples)
#     print(x1.shape)
#     J1 = x1.shape[0]
#     J2 = x2.shape[0]
#     kermatrix = np.zeros([J1, J2])
#     for i in range(J1):
#         for j in range(J2):
#             if j<i and j < J1 and i < J2:
#                 kermatrix[i,j] = kermatrix[j,i]
#             else:
#                 for m in range(len(x1[-1])):
#                     for n in range(len(x2[-1])):
#                     ip = np.linalg.norm(x1[i, :, m] - x2[j, :, n], 2) ** 2
#                     kermatrix[i,j] += np.exp((-1/(2*(l**2)))*np.sum(ip))
#     return kermatrix

def rbf_vector_kernel(x1, x2, l):
    """
    Parameters:
    x1 (numpy.ndarray): First input data array with shape (num_mesh_samples, 3, num_time_samples).
    x2 (numpy.ndarray): Second input data array with shape (num_mesh_samples, 3, num_time_samples).
    l (float): Length scale parameter for the RBF kernel.

    Returns:
    numpy.ndarray: RBF kernel matrix.
    """
    J1 = x1.shape[2]
    J2 = x2.shape[2]
    kermatrix = np.zeros([J1, J2])
    
    for i in range(J1):
        for j in range(J2):
            # Reshape the (3, num_mesh_samples) matrices into (3*num_mesh_samples,)
            x1_flat = x1[:, :, i].reshape(-1)
            x2_flat = x2[:, :, j].reshape(-1)
            sq_dist = np.sum((x1_flat - x2_flat) ** 2)
            # Compute the RBF kernel value
            kermatrix[i, j] = np.exp(-sq_dist / (2 * (l)** 2))

    return kermatrix

def polynomial_vector_kernel(x1, x2, c, a):
    J1 = x1.shape[2]
    J2 = x2.shape[2]
    kermatrix = np.zeros([J1, J2])
    for i in range(J1):
        for j in range(J2):
            # Reshape the (3, num_time_samples) matrices into (3*num_time_samples,)
            x1_flat = x1[:, :, i].reshape(-1)
            x2_flat = x2[:, :, j].reshape(-1)
            kermatrix[i,j] = (c + np.dot(x1_flat, x2_flat))**a
    return kermatrix


# Kernel interpolation
def optimal_recovery(X_train, y_train, X_test, lengthscale=0.1, lam=0.1, a = 5, ker='rbf'):
    if ker == 'rbf':
        K_train = rbf_vector_kernel(X_train, X_train, lengthscale) + lam**2 * np.eye(X_train.shape[2])
        K_test = rbf_vector_kernel(X_test, X_train, lengthscale)
    elif ker == 'poly':
        K_train = polynomial_vector_kernel(X_train, X_train, lengthscale, a) + lam**2 * np.eye(X_train.shape[2])
        K_test = polynomial_vector_kernel(X_test, X_train, lengthscale, a) 
    else:
        raise ValueError("Error: unexpected kernel type")

    cho_factor_train = cho_factor(K_train)
    alpha = cho_solve(cho_factor_train, y_train)

    # Prediction
    y_pred = K_test.dot(alpha)
    return y_pred

def mse_loss(y_actual, y_pred, N_test):
    return (1/N_test**2)*np.linalg.norm(y_actual - y_pred, ord=2)**2

def rhs(u1, f, k=-9.81, l=.2):
    return -(k/l)*np.sin(u1)+f

# Cross-validation to choose lengthscale
def cross_validation_lengthscale(X, y, values, num_folds=10, ker='rbf'):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    mse_values = []
    for val in values:
        fold_mse = []
        for train_index, val_index in kf.split(X.T):
            X_train, X_val = X[:, :, train_index], X[:, :, val_index]
            y_train, y_val = y[train_index], y[val_index]
            y_pred = optimal_recovery(X_train, y_train, X_val, val, ker='rbf')
            N_test = len(y_pred)
            mse = mse_loss(y_val, y_pred, N_test)
            fold_mse.append(mse)
        mse_values.append(np.mean(fold_mse))
    return mse_values

# Cross-validation to choose lengthscale
def cross_validation_nugget(X, y, values, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    mse_values = []
    for val in values:
        fold_mse = []
        for train_index, val_index in kf.split(X.T):
            X_train, X_val = X[:, :, train_index], X[:, :, val_index]
            y_train, y_val = y[train_index], y[val_index]
            y_pred = optimal_recovery(X_train, y_train, X_val, lam=val)
            N_test = len(y_pred)
            mse = mse_loss(y_val, y_pred, N_test)
            fold_mse.append(mse)
        mse_values.append(np.mean(fold_mse))
    return mse_values

# Cross-validation to choose lengthscale
def cross_validation(X, y, lengthscale_values, nugget_values, a_values=[5], num_folds=5, kernel='rbf'):
    if kernel == 'poly':
        return cross_validation_poly(X, y, lengthscale_values, nugget_values, a_values, num_folds)
    elif kernel == 'rbf':
        return cross_validation_rbf(X, y, lengthscale_values, nugget_values, num_folds)
    else: 
        raise ValueError("Error: unexpected kernel type")

def cross_validation_rbf(X, y, lengthscale_values, nugget_values, num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    mse_values = np.zeros((len(lengthscale_values), len(nugget_values)))
    for i, lengthscale in enumerate(lengthscale_values):
        for j, nugget in enumerate(nugget_values):
            fold_mse = []
            for train_index, val_index in kf.split(X.T):
                X_train, X_val = X[:, :, train_index], X[:, :, val_index]
                y_train, y_val = y[train_index], y[val_index]
                y_pred = optimal_recovery(X_train, y_train, X_val, lengthscale, nugget, ker='rbf')
                N_test = len(y_pred)
                mse = mse_loss(y_val, y_pred, N_test)
                fold_mse.append(mse)
            mse_values[i, j] = np.mean(fold_mse)
    return mse_values

def cross_validation_poly(X, y, lengthscale_values, nugget_values, a_values, num_folds):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    mse_values = np.zeros((len(lengthscale_values), len(nugget_values), len(a_values)))
    for i, lengthscale in enumerate(lengthscale_values):
        for j, nugget in enumerate(nugget_values):
            for k, a in enumerate(a_values):
                fold_mse = []
                for train_index, val_index in kf.split(X.T):
                    X_train, X_val = X[:, :, train_index], X[:, :, val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    y_pred = optimal_recovery(X_train, y_train, X_val, lengthscale, nugget, a, ker='poly')
                    N_test = len(y_pred)
                    mse = mse_loss(y_val, y_pred, N_test)
                    fold_mse.append(mse)
                mse_values[i, j, k] = np.mean(fold_mse)
    return mse_values

# Sample some data for train and test
def sample_idxs(n = 1000):
    some_values = np.floor(np.random.rand(n)*10)
    sample = np.array(np.where(some_values==0)).flatten()
    return sample

def load_and_shift_pend_data(results, rhs, sample, l, lam, kernel='rbf'):
    u1 = results[:,0,:]
    u2 = results[:,1,:]
    u = np.array([u1, u2])
    t = np.linspace(0,1,1000)
    
    # Build the train and test sets from the data
    t = t[sample]
    I = u1.shape[0]
    u1 = np.reshape(u1[:,sample],[I,len(sample)])
    u2 = np.reshape(u2[:,sample],[I,len(sample)])

    S1, S2 = smooth_pend(t, u1, u2, l=l, lam=lam, ker=kernel)
    
    S = np.transpose(np.array([S1, S2]), (0, 1, 3, 2)) #(2, 95, 3, 20)
    f = rhs[:, sample]
    return S, f, t