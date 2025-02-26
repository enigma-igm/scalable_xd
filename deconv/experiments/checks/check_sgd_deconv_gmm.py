import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from deconv.gmm.plotting import plot_covariance
from deconv.gmm.sgd_deconv_gmm import SGDDeconvGMM
from deconv.gmm.data import DeconvDataset

from data import generate_data

def check_sgd_deconv_gmm(D, K, N, plot=False, verbose=False, device=None):

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

        ax.plot(gmm.train_loss_curve, label='Training Loss')
        ax.plot(gmm.val_loss_curve, label='Validation Loss')

        fig, ax = plt.subplots()

        for i in range(K):
            sc = ax.scatter(
                X_train[:, i, 0],
                X_train[:, i, 1],
                alpha=0.2,
                marker='x',
                label='Cluster {}'.format(i)
            )
            plot_covariance(
                means[i, :],
                covars[i, :, :],
                ax,
                color=sc.get_facecolor()[0]
            )

        sc = ax.scatter(
            gmm.means[:, 0],
            gmm.means[:, 1],
            marker='+',
            label='Fitted Gaussians'
        )

        for i in range(K):
            plot_covariance(
                gmm.means[i, :],
                gmm.covars[i, :, :],
                ax,
                color=sc.get_facecolor()[0]
            )

        ax.legend()
        plt.show()


if __name__ == '__main__':
    sns.set()
    D = 2
    K = 3
    N = 500
    check_sgd_deconv_gmm(D, K, N, verbose=True, plot=True)
