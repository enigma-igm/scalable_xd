import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from deconv.gmm.plotting import plot_covariance
from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.data import DeconvDataset

from data import generate_data

def check_sgd_deconv_gmm(D, K, N, plot=False, verbose=False, device=None):
    '''
    The body of a SGD Deconvolution Gaussian mixture model question.

    Parameters
    ----------
    D : int
        Number of dimensions.
    K : int
        Number of Gaussian components.
    N : int
        The length of the data.
    plot : bool, default=False
        Whether to plot the training and test loss curve, and real/learned means and covariances.
    verbose :  bool, default=False
        Whether to print out the training process.
    device : torch.device, default=None
        The device used to train. Default is CPU.
    '''

    if not device:
        device = torch.device('cpu')

    # load random noisy training and test data, each with shape (N, K, D), and noise covariances with shape (N, K, D, D). 
    # also load the real data parameters: mean with shape (K, D) and covariances matrix with shape (K, D, D).
    data, params = generate_data(D, K, N) 
    X_train, nc_train, X_test, nc_test = data
    means, covars = params

    # load training data
    train_data = DeconvDataset(
        torch.Tensor(X_train.reshape(-1, D).astype(np.float32)),
        torch.Tensor(
            nc_train.reshape(-1, D, D).astype(np.float32)
        )
    )

    # load test data
    test_data = DeconvDataset(
        torch.Tensor(X_test.reshape(-1, D).astype(np.float32)),
        torch.Tensor(
            nc_test.reshape(-1, D, D).astype(np.float32)
        )
    )

    # initializing the training parameters
    gmm = SGDDeconvGMM(
        K, # number of components
        D, # number of dimensions
        device=device,
        batch_size=250,
        epochs=200,
        restarts=1,
        lr=1e-1
    )
    # training
    gmm.fit(train_data, val_data=test_data, verbose=verbose)
    train_score = gmm.score_batch(train_data)
    test_score = gmm.score_batch(test_data)

    print('Training score: {}'.format(train_score))
    print('Test score: {}'.format(test_score))

    # plotting the results
    if plot:
        fig, ax = plt.subplots()

        # plotting the training and test loss curve
        ax.plot(gmm.train_loss_curve, label='Training Loss')
        ax.plot(gmm.val_loss_curve, label='Validation Loss')

        fig, ax = plt.subplots()

        for i in range(K):
            # plot training data points
            sc = ax.scatter(
                X_train[:, i, 0],
                X_train[:, i, 1],
                alpha=0.2,
                marker='x',
                label='Cluster {}'.format(i)
            )

            # plot real means and covarianvces
            plot_covariance(
                means[i, :], # means shape (K, D)
                covars[i, :, :], # covars shape (K, D, D)
                ax,
                color=sc.get_facecolor()[0]
            )
        # plot learned means
        sc = ax.scatter(
            gmm.means[:, 0],
            gmm.means[:, 1],
            marker='+',
            label='Fitted Gaussians'
        )

        # plot learned covarianvces
        for i in range(K):
            plot_covariance(
                gmm.means[i, :],
                gmm.covars[i, :, :],
                ax,
                color=sc.get_facecolor()[0]
            )

        ax.legend()
        #plt.show()

        #embed()
        # plot the learned weight
        fig, ax = plt.subplots()
        width = 0.35

        ax.bar(np.arange(K) - width/2, 
            np.ones(K)/K, 
            width=width,
            color='C0',
            label='Data')  

        ax.bar(np.arange(K) + width/2,
            gmm.soft_weights.softmax(0),
            width=width,
            color=sc.get_facecolor()[0],
            label='Fitted') 
            
        ax.set_ylabel('Weights')
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([('Cluster ' + str(j)) for j in range(K)])
        plt.legend()
        plt.show()

from IPython import embed
if __name__ == '__main__':
    sns.set()
    D = 2
    K = 3
    N = 500
    check_sgd_deconv_gmm(D, K, N, verbose=True, plot=True)
