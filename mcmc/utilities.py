from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.special import comb

import jax.numpy as jnp
from jax import jit, vmap

from mcmc.data import Data
from mcmc.parameter import Parameter

def convert_to_nparray(values) -> np.ndarray:
    """
    Converts input to a numpy array.

    Args:
    -----
        input: np.ndarray, list of floats or float
            to be converted
    Returns:
    --------
        numpy array with values from input
    """
    if not isinstance(values, np.ndarray):
        if not isinstance(values, list):
            values = [values]
        return np.array(values)
    return values


def dist_matrix1(data: Data, theta: Parameter) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Calculates and returns the upper triangle of the distance matrix as a flattened array. Each
    row corresponds to a single location and each column to a different variable.
    """
    D = np.zeros((
        int((data.n + data.m) * (data.n + data.m - 1)/2),
        data.p + data.q
    ))
    indices = np.triu_indices(n=data.n+data.m, k=1)

    #the variables a and b are used to iterate through the rows of the matrix.
    #The [a,b] range of integers references the start and end block of each
        # observation x observation
        # observation x simulation
        # simulation x simulation
    #in an interative manor thus making it much easier to build the matrix.

    b = -1
    for i in range(data.n-1):
        # observation x observation
        a = b+1
        b = a+data.n-2-i

        #control variables
        D[a:(b+1), :data.p] = (data.x_f[i, :] - data.x_f[(i+1):, :])**2
        #calibration variables
            #are all 0 so nothing needed here.


        # observation x simulation
        a = b+1
        b = a+data.m-1

        #control variables
        D[a:(b+1), :data.p] = (data.x_f[i, :] - data.x_c[:, :])**2
        #calibration variables
        D[a:(b+1), data.p:] = (theta.values - data.t[:, :])**2



    # observation x simulation
    #fill in the final row
    a = b+1
    b = a+data.m-1
    #control variables
    D[a:(b+1), :data.p] = (data.x_f[data.n-1, :] - data.x_c[:, :])**2
    #calibration variables
    D[a:(b+1), data.p:] = (theta.values - data.t[:, :])**2



    # simulation x simulation
    for i in range(data.m-1):
        a = b+1
        b = a+data.m-2-i

        #control variables
        D[a:(b+1), :data.p] = (data.x_c[i, :] - data.x_c[(i+1):, :])**2
        #calibration variables
        D[a:(b+1), data.p:] = (data.t[i, :] - data.t[(i+1):, :])**2


    return D, indices




def dist_matrix2(data: Data, theta: Parameter) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates and returns the upper triangle of the distance matrix as a flattened array. Each
    row corresponds to a single location and each column to a different variable.
    """
    D = np.zeros((
        int((data.n + data.m) * (data.n + data.m - 1)/2),
        data.p + data.q
    ))
    D_delta = np.zeros((
        int((data.n) * (data.n - 1)/2),
        data.p
    ))

    #the variables a and b are used to iterate through the rows of the matrix.
    #The [a,b] range of integers references the start and end block of each
        # observation x observation
        # observation x simulation
        # simulation x simulation
    #in an interative manner thus making it much easier to build the matrix.

    b = -1
    for i in range(data.n-1):
        # observation x observation
        a = b+1
        b = a+data.n-2-i

        #control variables
        D[a:(b+1), :data.p] = (data.x_f[i, :] - data.x_f[(i+1):, :])**2
        D_delta[(a-i*data.m):(b+1-i*data.m), :] = D[a:(b+1), :data.p].copy()
        #calibration variables
            #are all 0 so nothing needed here.


        # observation x simulation
        a = b+1
        b = a+data.m-1

        #control variables
        D[a:(b+1), :data.p] = (data.x_f[i, :] - data.x_c[:, :])**2
        #calibration variables
        D[a:(b+1), data.p:] = (theta.values - data.t[:, :])**2



    # observation x simulation
    #fill in the final row
    a = b+1
    b = a+data.m-1
    #control variables
    D[a:(b+1), :data.p] = (data.x_f[data.n-1, :] - data.x_c[:, :])**2
    #calibration variables
    D[a:(b+1), data.p:] = (theta.values - data.t[:, :])**2



    # simulation x simulation
    for i in range(data.m-1):
        a = b+1
        b = a+data.m-2-i

        #control variables
        D[a:(b+1), :data.p] = (data.x_c[i, :] - data.x_c[(i+1):, :])**2
        #calibration variables
        D[a:(b+1), data.p:] = (data.t[i, :] - data.t[(i+1):, :])**2


    return D, D_delta





def dist_matrix3(data: Data, theta: Parameter) -> Tuple[np.ndarray, ...]:
    """Calculates and returns the upper triangle of the distance matrix as a flattened array. Each
    row corresponds to a single location and each column to a different variable.
    """
    D = np.empty((
        int((data.n + data.m) * (data.n + data.m - 1)/2),
        data.p + data.q
    ))
    D_delta = np.empty((
        int((data.n) * (data.n - 1)/2),
        data.p
    ))

    for i in range(data.p):
        D[:, i] = pdist(
            np.concatenate((data.x_f[:, i], data.x_c[:, i])).reshape(-1,1),
            'sqeuclidean'
        )
        D_delta[:, i] = pdist(
            data.x_f[:, i].reshape(-1,1),
            'sqeuclidean'
        )

    for i in range(data.p, data.p+data.q):
        D[:, i] = pdist(
            np.concatenate((np.tile(theta.values[i-data.p], data.n).reshape(-1,1), data.t)), 
            'sqeuclidean'
        )

    def triangle_diff(n, i):
        return comb(n,2) - comb(n-1-i,2)

    D_B_I = np.array(
        [[triangle_diff(data.n, i) + data.m*i + np.arange(0, data.m)] for i in range(data.n)], 
        dtype=int
    ).flatten()

    return D, D_delta, D_B_I


# def obs_sim_dist(data: Data, theta: float) -> np.ndarray:
#     return cdist(np.tile(theta, data.n).reshape(-1,1), data.t, 'sqeuclidean')



########## The numpy implementation using cdist is just as fast when n=10,m=255
def sqeuclidean(x, y):
    return (x - y)**2

sqeuclidean_vmapped = vmap(vmap(sqeuclidean, in_axes=(None, 0)), in_axes=(0, None))

def obs_sim_dist(data_t, theta_vec):
    return sqeuclidean_vmapped(theta_vec, data_t).flatten()

obs_sim_dist_jitted = jit(obs_sim_dist)