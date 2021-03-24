import numpy as np


def generate_data(D, K, N):
    '''
    This function returns random data that are mix-Gaussian distributed but with noise randomly generated with unidentical distribution. The first output is data, including 4 components: training data, training data noise covariance, test data, test data covariance. The data have shape (N, K, D), and the noise covariance hanve shape (N, K, D, D).
    D: int, the length of the first dimension of an individual datum.
    K: int, number of components, the length of the second dimension of an individual datum.
    N: int, number of dimensions, length of an data set.
    '''
    means = (np.random.rand(K, D) * 2000) - 1000 # Random array that has shape (K,D), uniformly distributed in (-1000, 1000)
    q = (2 * np.random.randn(K, D, D)) # Random array that has shape (K, D, D), normal distribution.
    covars = np.matmul(q.swapaxes(1, 2), q) # the random covariance matrix

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
