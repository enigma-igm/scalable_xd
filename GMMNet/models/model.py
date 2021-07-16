import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import WeightedRandomSampler
mvn = dist.multivariate_normal.MultivariateNormal



class GMMNet(nn.Module):
    """Neural network for Gaussian Mixture Model"""
    def __init__(self, 
                 n_components, 
                 data_dim, 
                 conditional_dim, 
                 means0, 
                 vec_dim=128,
                 ):
        super().__init__()
        self.n_components = n_components
        self.data_dim = data_dim
        self.conditional_dim = conditional_dim
        self.means0 = means0
        self.vec_dim = vec_dim
        
        self.embedding_network = nn.Sequential(*[nn.Linear(self.conditional_dim, self.vec_dim),
                                                 nn.PReLU(),
                                                 nn.Linear(self.vec_dim, self.vec_dim)])
       
        self.weights_network = nn.Sequential(*[nn.Linear(self.vec_dim, self.n_components),
                                               nn.Softmax(-1)])
        
        self.means_network = nn.Sequential(*[nn.Linear(self.vec_dim, self.n_components*self.data_dim)])
        
        self.covar_network = nn.Sequential(*[nn.Linear(self.vec_dim, self.n_components*self.data_dim*(self.data_dim+1)//2)])
        
    def forward(self, conditional):
        '''
        data: shape(batch_size, D)
        '''
        
        B = conditional.shape[0] # batch size
        
        # embed conditional info
        embedding = self.embedding_network(conditional)
        
        # calculate weights
        weights = self.weights_network(embedding)
        
        # calculate means
        means = self.means_network(embedding)
        means = means.reshape(-1, self.n_components, self.data_dim) + self.means0
        
        # calculate cholesky matrix
        covars_ele = self.covar_network(embedding)
        d_idx = torch.eye(self.data_dim).to(torch.bool)
        l_idx = torch.tril_indices(self.data_dim, self.data_dim, -1)
        
        scale_tril = torch.zeros((B, self.n_components, self.data_dim, self.data_dim))
        log_diagonal = covars_ele[:, :self.n_components*self.data_dim].reshape(B, self.n_components, self.data_dim)
        scale_tril[:, :, d_idx] = torch.exp(log_diagonal)
        lower_tri = covars_ele[:, self.n_components*self.data_dim:].reshape(B, self.n_components, self.data_dim*(self.data_dim-1)//2)
        scale_tril[:, :, l_idx[0], l_idx[1]] = lower_tri
        
        # calculate covariance matrix
        covars = torch.matmul(scale_tril, scale_tril.transpose(-2, -1))
        return weights, means, covars
    
    
    def log_prob_b(self, data, conditional, noise=None):
        weights, means, covars = self.forward(conditional)
        if noise is None:
            noise = torch.zeros_like(covars)
        
        noisy_covars = covars + noise
        
        log_resp = mvn(loc=means, covariance_matrix=noisy_covars).log_prob(data[:, None, :])
        
        log_resp += torch.log(weights)

        log_prob_b = torch.logsumexp(log_resp, dim=1)

        return log_prob_b


    def sample(self, conditional, n_per_conditional=1):
        
        weights, means, covars = self.forward(conditional)

        draw = list(WeightedRandomSampler(weights, n_per_conditional))
        
        batchsize = conditional.shape[0]
        means  = means[:, draw][torch.eye(batchsize).to(torch.bool)]
        covars = covars[:, draw][torch.eye(batchsize).to(torch.bool)]
        data   = mvn(loc=means, covariance_matrix=covars).sample()

        return data
        




#github copilot
        