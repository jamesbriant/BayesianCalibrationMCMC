from typing import Tuple

import numpy as np

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


# def dist_matrix(data: Data, theta: Parameter) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#     """Calculates and returns the upper triangle of the distance matrix as a flattened array. Each
#     row corresponds to a single location and each column to a different variable.
#     """
#     D = np.zeros((
#         int((data.n + data.m) * (data.n + data.m - 1)/2),
#         data.p + data.q
#     ))
#     indices = np.triu_indices(n=data.n+data.m, k=1)

#     #the variables a and b are used to iterate through the rows of the matrix.
#     #The [a,b] range of integers references the start and end block of each
#         # observation x observation
#         # observation x simulation
#         # simulation x simulation
#     #in an interative manor thus making it much easier to build the matrix.

#     b = -1
#     for i in range(data.n-1):
#         # observation x observation
#         a = b+1
#         b = a+data.n-2-i

#         #control variables
#         D[a:(b+1), :data.p] = (data.x_f[i, :] - data.x_f[(i+1):, :])**2
#         #calibration variables
#             #are all 0 so nothing needed here.


#         # observation x simulation
#         a = b+1
#         b = a+data.m-1

#         #control variables
#         D[a:(b+1), :data.p] = (data.x_f[i, :] - data.x_c[:, :])**2
#         #calibration variables
#         D[a:(b+1), data.p:] = (theta.values - data.t[:, :])**2



#     # observation x simulation
#     #fill in the final row
#     a = b+1
#     b = a+data.m-1
#     #control variables
#     D[a:(b+1), :data.p] = (data.x_f[data.n-1, :] - data.x_c[:, :])**2
#     #calibration variables
#     D[a:(b+1), data.p:] = (theta.values - data.t[:, :])**2



#     # simulation x simulation
#     for i in range(data.m-1):
#         a = b+1
#         b = a+data.m-2-i

#         #control variables
#         D[a:(b+1), :data.p] = (data.x_c[i, :] - data.x_c[(i+1):, :])**2
#         #calibration variables
#         D[a:(b+1), data.p:] = (data.t[i, :] - data.t[(i+1):, :])**2


#     return D, indices


def dist_matrix(data: Data, theta: Parameter) -> Tuple[np.ndarray, np.ndarray]:
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
    #in an interative manor thus making it much easier to build the matrix.

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