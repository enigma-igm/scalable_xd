import torch
import torch.utils.data as data_utils


class DeconvDataset(data_utils.Dataset):
    """
    Dataset with values and noise covariances.  
    Attributes:
    - __len__(): length of the data.
    - __getitem__(int(i)): to get the ith item.
    """

    def __init__(self, X, noise_covars):
        self.X = X
        self.noise_covars = noise_covars

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i, :], self.noise_covars[i, :, :])
