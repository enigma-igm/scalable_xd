import numpy as np


# the assumed functions
# all functions use simple linear transformations to a given cond vector. The transformations are fixed by given random seeds.
def weight_func(param_cond, K):
    '''
    To generate the weights of the components given the conditiontial parameters.
    '''
    D_cond = len(param_cond)
    np.random.seed(9)
    matr = np.random.permutation(K*D_cond*10)[:D_cond].reshape(D_cond) / (K*D_cond*5)
    weights0 = np.zeros(K)
    for i in range(K):
        weights0[i] = np.dot(matr**(1-i/10), param_cond**(1+i/10))

    weights0 = weights0 / weights0.sum()

    return weights0


def means_func(param_cond, K, D):
    '''
    To generate the means of the components given the conditiontial parameters.
    '''
    D_cond = len(param_cond)
    np.random.seed(13)
    matr = np.random.permutation(K*D*D_cond*10)[:K*D*D_cond].reshape(K, D, D_cond) / (K*D)
    matr -= matr.mean()
    means0 = np.dot(matr, (param_cond)**1.2)
    
    return means0

def covar_func(param_cond, K, D):
    '''
    To generate the covariance of the components given the conditiontial parameters.
    '''

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

def sample_func(weights0, means0, covars0, N=1):
    '''
    Random sampling given weights, means and covariance.
    '''

    K = len(weights0)
    D = len(means0[0])
    X = np.empty((N, D))
    np.random.seed()
    draw = np.random.choice(np.arange(K), N, p=weights0) # which component each data is from
    
    for j in range(N):
        # the real data
        X[j, :] = np.random.multivariate_normal(
            mean=means0[draw[j], :],
            cov=covars0[draw[j], :, :]
        )
        
    # shuffle the data
    p = np.random.permutation(N)
    X_train = X[p]
    
    # shuffle the draw array
    draw = draw[p]
    
    return X_train, draw


def data_load(N, K, D, D_cond):

    param_cond = np.zeros((N, D_cond))
    weights = np.zeros((N, K))
    means   = np.zeros((N, K, D))
    covars  = np.zeros((N, K, D, D))
    draw    = np.zeros(N)
    data    = np.zeros((N, D))


    # conditional paramesters
    for i in range(N):
        np.random.seed()
        param_cond[i] = np.random.rand(D_cond)
        weights[i]    = weight_func(param_cond[i], K)
        means[i]      = means_func(param_cond[i], K, D)
        covars[i]     = covar_func(param_cond[i], K, D)
        data[i], draw[i] = sample_func(weights[i], means[i], covars[i])

    return (param_cond, weights, means, covars, data, draw)