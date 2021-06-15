import numpy as np
from numpy.random import choice


def generate_data(D, K, N):
    '''
    This function generates random data that are mix-Gaussian distributed but with noise randomly generated with unidentical distribution. In this function the Gaussian mixture model equally weights its components, and particularly each component gives equal number of data.

    Parameters
    ----------
    D : int
        Number of dimensions.
    K : int
        Number of Gaussian components.
    N : int
        The length of the data.
    Returns
    -------
    data : tuple, 
        The noisy data and noise covariance. This includes 4 components: training data, shape (N, K, D), training data noise covariance, shape (N, K, D, D) test data, shape (N, K, D) test data covariance, shape (N, K, D, D).
    params : tuple,
        The parameters of the real Gaussian mixture distribution. This includes 2 components: means, shape (K, D), covars, shape (K, D, D).
    '''
    means = (np.random.rand(K, D) * 2000) - 1000 # Random array that has shape (K,D), uniformly distributed in (-1000, 1000)
    q = (2 * np.random.randn(K, D, D)) # Random array that has shape (K, D, D), normal distribution.
    covars = np.matmul(q.swapaxes(1, 2), q) # the random covariance matrix, shape (K, D, D)

    qn = (0.5 * np.random.randn(2 * N, K, D, D))
    noise_covars = np.matmul(qn.swapaxes(2, 3), qn) # the noise covariance matrix (unidentical noise)

    X = np.empty((2 * N, K, D)) # the dataset, in which the first N are training set while the latter N are test set

    for i in range(K):
        # the real data
        X[:, i, :] = np.random.multivariate_normal(
            mean=means[i, :],
            cov=covars[i, :, :],
            size=2 * N
        )
        # the noise
        for j in range(2 * N):
            X[j, i, :] += np.random.multivariate_normal(
                mean=np.zeros(D),
                cov=noise_covars[j, i, :, :]
            )

    # shuffle the data
    p = np.random.permutation(2 * N)

    # split the data
    X_train = X[p, :][:N]
    X_test = X[p, :][N:]

    # split the noise covariance matrix
    nc_train = noise_covars[p, :, :][:N]
    nc_test = noise_covars[p, :, :][N:]

    # return the noisy data and the parameters of the real data
    data = (X_train, nc_train, X_test, nc_test)
    params = (means, covars)

    return data, params




def generate_data2(D, K, N, weights=None):
    '''
    This function generates random data that are mix-Gaussian distributed but with noise randomly generated with unidentical distribution.

    Parameters
    ----------
    D : int
        Number of dimensions.
    K : int
        Number of Gaussian components.
    N : int
        The length of the data.
    weights: array, default=None
        The weight of each Gaussian component.
    Returns
    -------
    data : tuple, 
        The noisy data and noise covariance. This includes 4 components: training data, shape (N, D), training data noise covariance, shape (N, D, D) test data, shape (N, D) test data covariance, shape (N, D, D).
    params : tuple,
        The parameters of the real Gaussian mixture distribution. This includes 4 components: means, shape (K, D), covars, shape (K, D, D), weights, length K, and the index of component each data is from, length N.
    '''
    means = (np.random.rand(K, D) * 2000) - 1000 # Random array that has shape (K,D), uniformly distributed in (-1000, 1000)
    q = (2 * np.random.randn(K, D, D)) # Random array that has shape (K, D, D), normal distribution.
    covars = np.matmul(q.swapaxes(1, 2), q) # the random covariance matrix, shape (K, D, D)

    qn = (0.5 * np.random.randn(2 * N, D, D))
    noise_covars = np.matmul(qn.swapaxes(1, 2), qn) # the noise covariance matrix (unidentical noise)

    # draw the data according to the weights
    if weights==None:
        weights = np.random.uniform(size=K)
        weights/= weights.sum()
    draw = choice(np.arange(K), 2*N, p=weights) # which component each data is from
        

    X = np.empty((2 * N, D)) # the dataset, in which the first N are training set while the latter N are test set

    #for i in range(K):
    for j in range(2 * N):
        # the real data
        X[j, :] = np.random.multivariate_normal(
            mean=means[draw[j], :],
            cov=covars[draw[j], :, :]
        )
        # the noise
        
        X[j, :] += np.random.multivariate_normal(
            mean=np.zeros(D),
            cov=noise_covars[j, :, :]
        )

    # shuffle index
    p = np.random.permutation(2 * N)

    # shuffle the component selecting array
    draw = draw[p][:N]

    # split and shuffle the data
    X_train = X[p][:N]
    X_test  = X[p][N:]

    # split the noise covariance matrix
    nc_train = noise_covars[p][:N]
    nc_test  = noise_covars[p][N:]

    # return the noisy data and the parameters of the real data
    data = (X_train, nc_train, X_test, nc_test)
    params = (means, covars, weights, draw)

    return data, params