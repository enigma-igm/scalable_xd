import numpy as np


# the assumed functions
# all functions use simple linear transformations to a given cond vector. The transformations are fixed by given random seeds.
def weight_func(param_cond, K):
    """To generate the weights of the Gaussian components given the conditiontial parameters.

    Args:
        param_cond (array): conditional parameter that determines the noise function. 1D array.
        K (int): Number of Gaussian components (how many Gaussians).

    Returns:
        array: The weight of each Gaussian component, add up to unity. 1D array, lenght K. 
    """
    D_cond = len(param_cond)
    np.random.seed(9)
    matr = np.random.permutation(K*D_cond*10)[:D_cond].reshape(D_cond) / (K*D_cond*5)
    weights0 = np.zeros(K)
    for i in range(K):
        weights0[i] = np.dot(matr**(1-i/10), param_cond**(1+i/10))

    weights0 = weights0 / weights0.sum()

    return weights0


def means_func(param_cond, K, D):
    """To generate the means of the Gaussian components given the conditiontial parameters.

    Args:
        param_cond (array): conditional parameter that determines the noise function. 1D array.
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.

    Returns:
        narray: The means of each Gaussian component. KxD array.
    """
    D_cond = len(param_cond)
    np.random.seed(13)
    matr = np.random.permutation(K*D*D_cond*10)[:K*D*D_cond].reshape(K, D, D_cond) / (K*D)
    matr -= matr.mean()
    means0 = np.dot(matr, param_cond**1.2)
    
    return means0

def covar_func(param_cond, K, D):
    """To generate the covariance of the Gaussian components given the conditiontial parameters.

    Args:
        param_cond (array): conditional parameter that determines the noise function. 1D array.
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.

    Returns:
        narray: Covariance matrices of the Gaussian mixture model. KxDxD array.
    """

    D_cond = len(param_cond)
    
    np.random.seed(21)
    matr1 = np.random.permutation(K*D*D_cond*10)[:K*D*D_cond].reshape(K, D, D_cond) / (K*D*D_cond*50)
    l_diag = np.dot(matr1, param_cond**0.5)
    
    np.random.seed(35)
    matr2 = np.random.permutation(K*D*(D-1)//2*D_cond*10)[:K*D*(D-1)//2*D_cond].reshape(K, D*(D-1)//2, D_cond) / (K*D*(D-1)//2*D_cond*50)
    l_lower = np.dot(matr2, param_cond**0.5)

    d_idx = np.eye(D).astype('bool')
    l_idx = np.tril_indices(D, -1)

    matrL = np.zeros((K, D, D))

    matrL[:, d_idx] = l_diag + (0.1)**0.5
    matrL[:, l_idx[0], l_idx[1]] = l_lower
    
    covars0 =(np.matmul(matrL, np.swapaxes(matrL, -2, -1)))

    return covars0


def noise_func(param_cond, D):
    """To generate the noise matrix. Diagonal and identical so far.

    Args:
        param_cond (array): conditional parameter that determines the noise function. 1D array..
        D (int): Dimension of data.

    Returns:
        narray: DxD noise array.
    """
    return (np.eye(D)*0.3)


def sample_func(weights0, means0, covars0, noise=None, N=1):
    """Random sampling from a Gaussian mixture distribution.

    Args:
        weights0 (array): The weight of each Gaussian component, length K.
        means0 (narray): The means of each Gaussian component, shape KxD.
        covars0 (narray): Covariance matrices of the Gaussian mixture model, shape KxDxD.
        noise (narray, optional): The noise matrix, shape DxD. Defaults to None.
        N (int, optional): Size of samples. Defaults to 1.

    Returns:
        (narray, array): Samples and the index of which Gausssian component the sample belongs to. The sample has shape NxD, and the index has length N.
    """

    K = len(weights0)
    D = len(means0[0])
    X = np.empty((N, D))
    np.random.seed()
    draw = np.random.choice(np.arange(K), N, p=weights0) # which component each data is from
    
    if noise is None:
        noise = np.zeros_like(covars0[0, :, :])

    for j in range(N):
        # the real data
        X[j, :] = np.random.multivariate_normal(
            mean=means0[draw[j], :],
            cov=covars0[draw[j], :, :] + noise
        )
        
    # shuffle the data
    p = np.random.permutation(N)
    X_train = X[p]
    
    # shuffle the draw array
    draw = draw[p]
    
    return X_train, draw


def data_load(N, K, D, D_cond, noisy=False):
    """General data loader.

    Args:
        N (int): Size of dataset.
        K (int): Number of Gaussian components (how many Gaussians).
        D (int): Dimension of data.
        D_cond (int): Dimension of conditional parameter.
        noisy (bool, optional): Whether add noise or not. Defaults to False.

    Returns:
        (narray, narray, narray, narray, narray, narray, array): Conditional parameters, shape NxD. Weights, shape NxK. Means, shape KxD. Covariance matrices, shape NxKxDxD. Noise matrices, shape NxDxD (is zeros if noisy=False). Index indicating which component the data belong to, length N.
    """
    param_cond = np.zeros((N, D_cond))
    weights = np.zeros((N, K))
    means   = np.zeros((N, K, D))
    covars  = np.zeros((N, K, D, D))
    noise   = np.zeros((N, D, D))
    draw    = np.zeros(N)
    data    = np.zeros((N, D))


    # conditional paramesters
    for i in range(N):
        np.random.seed()
        param_cond[i] = np.random.rand(D_cond)
        weights[i]    = weight_func(param_cond[i], K)
        means[i]      = means_func(param_cond[i], K, D)
        covars[i]     = covar_func(param_cond[i], K, D)

        if noisy is True:
            noise[i]      = noise_func(param_cond[i], D)
            data[i], draw[i] = sample_func(weights[i], means[i], covars[i], noise=noise[i])

        if noisy is False:
            data[i], draw[i] = sample_func(weights[i], means[i], covars[i])

    return (param_cond, weights, means, covars, data, noise, draw)