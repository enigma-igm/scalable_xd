from abc import ABC
import copy

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.utils.data as data_utils

from .util import minibatch_k_means

mvn = dist.multivariate_normal.MultivariateNormal


class SGDGMMModule(nn.Module):
    """Implementation of a standard GMM as a PyTorch nn module."""

    def __init__(self, components, dimensions, w, device=None):
        super().__init__()

        self.k = components
        self.d = dimensions
        self.device = device

        self.soft_weights = nn.Parameter(torch.zeros(self.k))
        self.soft_max = torch.nn.Softmax(dim=0)

        self.means = nn.Parameter(torch.rand(self.k, self.d))
        self.l_diag = nn.Parameter(torch.zeros(self.k, self.d))

        self.l_lower = nn.Parameter(
            torch.zeros(self.k, self.d * (self.d - 1) // 2)
        )

        self.d_idx = torch.eye(self.d, device=self.device).to(torch.bool)
        self.l_idx = torch.tril_indices(self.d, self.d, -1, device=self.device)

        self.w = w * torch.eye(self.d, device=device)

    @property
    def L(self):
        '''The covariance of the data'''
        # d_idx is the diagonal index of the last 2 dimensions, and l_idx is the lower left triangle without diagonal index of the last 2 dimensions. 
        L = torch.zeros(self.k, self.d, self.d, device=self.device)
        L[:, self.d_idx] = torch.exp(self.l_diag)
        L[:, self.l_idx[0], self.l_idx[1]] = self.l_lower
        return L

    @property
    def covars(self):
        # covariance matrix, shape (k, d, d)
        return torch.matmul(self.L, torch.transpose(self.L, -2, -1))

    def forward(self, data):

        x = data[0] # get the data

        weights = self.soft_max(self.soft_weights) # weights for the data

        log_resp = mvn(loc=self.means, scale_tril=self.L).log_prob(
            x[:, None, :] # the Multi-Gaussian distribution template
        )
        log_resp += torch.log(weights)

        log_prob = torch.logsumexp(log_resp, dim=1)

        return -1 * torch.sum(log_prob)


class BaseSGDGMM(ABC):
    """ABC for fitting a PyTorch nn-based GMM.
    Parameters
    ----------
    components : int
        The number of mixture components.

    dimensions : int
        The number of Gaussian dimensions.

    epochs : int, default=10000
        The number of training iterations in each restart. As max_no_improvement exists, the training of a restart does not need to go thorugh all the epochs.

    lr : float, default=1e-3
        learning rate. 

    batch_size : int
        The size of a batch.

    tol : float, default=1e-6
        The convergence criterion, of the training loss change between every 2 epoch.

    restarts : int, default=5
        The number of times of training. In each restart the final training or validation (if set) is restored and after all restarts the best model with the lowest loss is saved.

    max_no_improvement : int, default=20
        The maximum allowed number of epochs with no improvement. Only valid when validation data is provided.

    k_means_factor : int, default=100
        ???

    w : float, default=1e-6
        ???

    k_means_iters : int, default=10
        ???

    lr_step : int, default=5
        The range of the milestone of the scheduler is [lr_step, lr_step + 5]. Milestones: List of epoch indices.

    lr_gamma : float, default=0.1
        The gamma of the milestone of the scheduler. Multiplicative factor of learning rate decay.

    device : torch.device
        The device used to train the model.
    """

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, restarts=5, max_no_improvement=20,
                 k_means_factor=100, w=1e-6, k_means_iters=10, lr_step=5,
                 lr_gamma=0.1, device=None):
        self.k = components
        self.d = dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = 1e-6 # the convergence criterion, of the training loss change between every 2 epoch
        self.lr = lr    # learing rate
        self.w = w      # what factor?
        self.restarts = restarts                     # number of times to training the model (outside of epochs)
        self.k_means_factor = k_means_factor         # ??
        self.k_means_iters = k_means_iters           # 
        self.max_no_improvement = max_no_improvement # the max allowed number of epochs with no improvement 

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.module.to(device)

        # Adam optimizer
        self.optimiser = torch.optim.Adam(
            params=self.module.parameters(),
            lr=self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimiser,
            milestones=[lr_step, lr_step + 5],
            gamma=lr_gamma
        )

    @property
    def means(self):
        return self.module.means.detach()

    @property
    def covars(self):
        return self.module.covars.detach()

    @property
    def soft_weights(self):
        return self.module.soft_weights.detach()

    def reg_loss(self, n, n_total):
        '''
        regression loss.
        '''
        l = (n / n_total) * self.w / torch.diagonal(self.module.covars, dim1=-1, dim2=-2)
        return l.sum()

    def fit(self, data, val_data=None, verbose=False, interval=1):
        """Fit the GMM to data."""
        n_total = len(data)

        # ???
        init_loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size * self.k_means_factor,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )

        loader = data_utils.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )

        best_loss = float('inf')

        for q in range(self.restarts):

            self.init_params(init_loader)

            train_loss_curve = []

            if val_data:
                val_loss_curve = []

            prev_loss = float('inf') # the training loss of the last epoch
            if val_data:
                best_val_loss = float('inf')
                no_improvement_epochs = 0

            for i in range(self.epochs):
                train_loss = 0.0
                for j, d in enumerate(loader):
                    # j is the order of the minibatch, d is the minibatch
                    d = [a.to(self.device) for a in d]
                    # initializing the gradient
                    self.optimiser.zero_grad()
                    
                    # getting the loss of the current minibatch
                    loss = self.module(d)
                    train_loss += loss.item()

                    # adding reg_loss to the current training loss
                    n = d[0].shape[0] # length of the minibatch
                    loss += self.reg_loss(n, n_total)

                    # backward and update parameters
                    loss.backward()
                    self.optimiser.step()

                # recording the current training loss
                train_loss_curve.append(train_loss)

                # getting the validation loss
                if val_data:
                    val_loss = self.score_batch(val_data)
                    val_loss_curve.append(val_loss)

                self.scheduler.step()

                # print out the training (and validation) loss
                if verbose and i % interval == 0:
                    if val_data:
                        print('Epoch {}, Train Loss: {}, Val Loss :{}'.format(
                            i,
                            train_loss,
                            val_loss
                        ))
                    else:
                        print('Epoch {}, Loss: {}'.format(i, train_loss))

                if val_data:
                    # if the validation loss is decreasing then continuing the training
                    if val_loss < best_val_loss:
                        no_improvement_epochs = 0
                        best_val_loss = val_loss
                    else:
                        no_improvement_epochs += 1
                    # if the validation loss did not decrease for epochs more than max_no_improvement, stop training
                    if no_improvement_epochs > self.max_no_improvement:
                        print('No improvement in val loss for {} epochs. Early Stopping at {}'.format(
                            self.max_no_improvement,
                            val_loss
                        ))
                        break
                
                # if the change of the training loss is less than self.tol, stop training
                if abs(train_loss - prev_loss) < self.tol:
                    print('Training loss converged within tolerance at {}'.format(
                        train_loss
                    ))
                    break
                
                # update the training loss of the 'previous' epoch
                prev_loss = train_loss
            
            # the loss of the current training, after all epochs
            if val_data:
                score = val_loss
            else:
                score = train_loss

            # if the loss is the minimum so far, update the model
            if score < best_loss:
                best_model = copy.deepcopy(self.module)
                best_loss = score
                best_train_loss_curve = train_loss_curve
                if val_data:
                    best_val_loss_curve = val_loss_curve

        # save the best model
        self.module = best_model
        self.train_loss_curve = best_train_loss_curve
        if val_data:
            self.val_loss_curve = best_val_loss_curve

    def score(self, data):
        with torch.no_grad():
            return self.module(data)

    def score_batch(self, dataset):
        '''
        Loss of a batch of data.
        '''
        loader = data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

        log_prob = 0

        for j, d in enumerate(loader):
            d = [a.to(self.device) for a in d]
            log_prob += self.score(d).item()

        return log_prob

    def init_params(self, loader):
        '''
        To initialize the parameters, which is to get the weights, data, and zero the covariance.?
        '''
        counts, centroids = minibatch_k_means(loader, self.k, max_iters=self.k_means_iters, device=self.device)
        self.module.soft_weights.data = torch.log(counts / counts.sum())
        self.module.means.data = centroids
        self.module.l_diag.data = nn.Parameter(torch.zeros(self.k, self.d, device=self.device))
        self.module.l_lower.data = torch.zeros(self.k, self.d * (self.d - 1) // 2, device=self.device)


class SGDGMM(BaseSGDGMM):
    """Concrete implementation of class to fit a standard GMM with SGD."""

    def __init__(self, components, dimensions, epochs=10000, lr=1e-3,
                 batch_size=64, tol=1e-6, w=1e-3, device=None):
        self.module = SGDGMMModule(components, dimensions, w, device)
        super().__init__(
            components, dimensions, epochs=epochs, lr=lr,
            batch_size=batch_size, w=w, tol=tol, device=device
        )
