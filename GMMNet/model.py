import torch
import torch.nn as nn
import torch.distributions as dist
mvn = dist.multivariate_normal.MultivariateNormal



class GMMNet(nn.Module):
    def __init__(self, K, D, D_cond, means0):
        super().__init__()
        self.K = K
        self.D = D
        self.D_cond = D_cond
        self.means0 = means0
        
        self.vec_dim = 128
        
        self.soft_max = torch.nn.Softmax(dim=0)
        
        self.embedding_network = nn.Sequential(*[nn.Linear(self.D_cond, self.vec_dim),
                                               nn.PReLU(),
                                                nn.Linear(self.vec_dim, self.vec_dim)])
       
        self.weights_network = nn.Sequential(*[nn.Linear(self.vec_dim, self.K),
                                            nn.Softmax(-1)])
        
        self.means_network = nn.Sequential(*[nn.Linear(self.vec_dim, self.K*self.D)])
        
        self.covar_network = nn.Sequential(*[nn.Linear(self.vec_dim, self.K*self.D*(self.D+1)//2)])
        
    def forward(self, data):
        '''
        data: shape(batch_size, D)
        '''
        
        B = data.shape[0] # batch size
        
        embedding = self.embedding_network(data)
        
        weights   = self.weights_network(embedding)
        #weights   = weights / weights.sum()
        
        means     = self.means_network(embedding)
        means     = means.reshape(-1, self.K, self.D) + self.means0
        
        covars_ele = self.covar_network(embedding)
        d_idx = torch.eye(self.D).to(torch.bool)
        l_idx = torch.tril_indices(self.D, self.D, -1)
        covars = torch.zeros((B, self.K, self.D, self.D))
        covars[:, :, d_idx] = torch.exp(covars_ele[:, :self.K*self.D].reshape(B, self.K, self.D))
        covars[:, :, l_idx[0], l_idx[1]] = covars_ele[:, self.K*self.D:].reshape(B, self.K, self.D*(self.D-1)//2)
        
        return weights, means, covars
    
    
    def log_prob_b(self, data, conditional):
        
        weights, means, covars = self.forward(conditional)
        
        log_resp = mvn(loc=means, scale_tril=covars).log_prob(data[:, None, :])
        
        log_resp += torch.log(weights)

        log_prob_b = torch.logsumexp(log_resp, dim=1)

        return log_prob_b