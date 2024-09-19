from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from scipy.integrate import solve_ivp


def pendulum_train_data(I=20, k=9.81/0.2, noise = 0, random_state = 0):
    """ Generates pendulum data
    Using the following system:
        (u_1)_t = u_2
        (u_2)_t = -k * sin(u_1) + f(t)
    f(t) is the source term generated from a GP with RBF kernel with length scale 0.2
    Solves the system using solve_ivp for time [0,1] with 1000 points

    Parameters: 
    I: Int
        Decides the number of samples
    k:  float, default = 9.81/0.2
        k value in ODE
    noise: float, default = 0
        Decides the noise-signal ratio for added gaussian noise.

    Returns:
    u, source_term
        u: ndarray, shape (I, 2, 1000)
            u contains both u_1 and u_2 of the system
        source_term: ndarray, shape (I, 1000)
    """

    def f(t, u, single_prior, time):
        u_1 = u[0]
        u_2 = u[1]
        idx  = min(time, key=lambda x:abs(x-t))
        return np.array([u_2, -k * np.sin(u_1) + single_prior[time == idx][0]])

    kernel = 1.0 * RBF(length_scale=0.2, length_scale_bounds=(1e-1, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    time = np.linspace(0, 1, 1000)
    init_u = [0, 0]


    t = time.reshape(-1,1)
    y_samples = gpr.sample_y(t, I)
    results = []
    for idx, single_prior in enumerate(y_samples.T):
        results.append(solve_ivp(f, [0, 1], init_u, method="RK45", t_eval=time, rtol=1e-8, args=(single_prior,time)).y)
        if noise != 0:
            results[-1][0] += noise * np.random.normal(0,np.max(np.abs(results[-1][0])),1000)
            results[-1][1] += noise * np.random.normal(0,np.max(np.abs(results[-1][1])),1000)
    results = np.array(results)
    return results, y_samples.T


def pendulum_test_data(I=20, k=9.81/0.2, noise = 0, B = 0.5, random_state = 0):
    """ Generates pendulum test data
    Using the following system:
        (u_1)_t = u_2
        (u_2)_t = -k * sin(u_1) + f(t) + B*sin(5*pi*t)
    f(t) is the source term generated from a GP with RBF kernel with length scale 0.2
    Solves the system using solve_ivp for time [0,1] with 1000 points

    Parameters: 
    I: Int
        Decides the number of samples
    k:  float, default = 9.81/0.2
        k value in ODE
    noise: float, default = 0
        Decides the noise-signal ratio for added gaussian noise.
    B: float, default = 0.5
        Decides the amount that we perturb the source terms.

    Returns:
    u, source_term
        u: ndarray, shape (I, 2, 1000)
            u contains both u_1 and u_2 of the system
        source_term: ndarray, shape (I, 1000)
    """

    def f(t, u, single_prior, time):
        u_1 = u[0]
        u_2 = u[1]
        idx  = min(time, key=lambda x:abs(x-t))
        return np.array([u_2, -k * np.sin(u_1) + single_prior[time == idx][0]])

    kernel = 1.0 * RBF(length_scale=0.2, length_scale_bounds=(1e-1, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    time = np.linspace(0, 1, 1000)
    init_u = [0, 0]


    t = time.reshape(-1,1)
    y_samples = gpr.sample_y(t, I)
    results = []
    for idx, single_prior in enumerate(y_samples.T):
        results.append(solve_ivp(f, [0, 1], init_u, method="RK45", t_eval=time, rtol=1e-8, args=(single_prior + B * np.sin(5*np.pi * time),time)).y)
        if noise != 0:
            results[-1][0] += noise * np.random.normal(0,np.max(np.abs(results[-1][0])),1000)
            results[-1][1] += noise * np.random.normal(0,np.max(np.abs(results[-1][1])),1000)
    results = np.array(results)
    return results, y_samples.T + np.tile(B * np.sin(5*np.pi * time), (I, 1))
