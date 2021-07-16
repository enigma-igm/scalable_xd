from data.toy import *
from models.model import *
from diagnostics.toy import all_figures

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from more_itertools import chunked

from sklearn.cluster import KMeans

import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

K, D, D_cond = 3, 2, 1

N_t = 10000
N_v = N_t//5

param_cond_t, weights_t, means_t, covars_t, data_t, draw_t = data_load(N_t, K, D, D_cond)
param_cond_v, weights_v, means_v, covars_v, data_v, draw_v = data_load(N_v, K, D, D_cond)

# kmeans to classify each data point
kmeans_t = KMeans(n_clusters=K, random_state=0).fit(data_t)
means0_t = kmeans_t.cluster_centers_

kmeans_v = KMeans(n_clusters=K, random_state=0).fit(data_v)
means0_v = kmeans_v.cluster_centers_

# initialization
gmm = GMMNet(K, D, D_cond, torch.FloatTensor(means0_t))

learning_rate = 1e-3
optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)

log_true = 0
for i, data_i in enumerate(data_t):
    
    log_resp = mvn(loc=torch.from_numpy(means_t[i][None, :]),
                   covariance_matrix=torch.from_numpy(covars_t[i][None, :])
                   ).log_prob(torch.from_numpy(data_i[None, None, :]))

    log_resp += torch.log(torch.ones(1, K)/K)

    log_prob = torch.logsumexp(log_resp, dim=1)
    
    log_true += log_prob
    
log_true = log_true / data_t.shape[0]


batch_size = 250
def chunck(array, bsize):

    array = list(chunked(array, bsize)) 
    array = torch.FloatTensor(array)

    return array


data_t = chunck(data_t, batch_size)
data_v = chunck(data_v, batch_size)


param_cond_t = chunck(param_cond_t, batch_size)
param_cond_v = chunck(param_cond_v, batch_size)




epoch = 250

for n in range(epoch):
    try:
        train_loss = 0
        for i, data_i in enumerate(data_t):
            optimizer.zero_grad()
            loss = -gmm.log_prob_b(data_i, param_cond_t[i]).mean()
            train_loss += loss.item()

            # backward and update parameters
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / data_t.shape[0]
        print((n+1),'Epoch', train_loss)
        scheduler.step(train_loss)

    except KeyboardInterrupt:
        break
    

# global check
# conditional parameter covering the training range
param_cond_tes = np.arange(0, 1, 0.01) + 0.01
param_cond_tes = torch.FloatTensor(param_cond_tes.reshape(-1,1))

# get the trained parameters
weight_tes, means_tes, covars_tes = gmm(param_cond_tes)
weight_tes = weight_tes.detach().numpy()
means_tes  = means_tes.detach().numpy()
covars_tes = covars_tes.detach().numpy()

# get the true paramters
weight_r   = np.zeros((len(param_cond_tes), K))
means_r    = np.zeros((len(param_cond_tes), K, D))
covars_r   = np.zeros((len(param_cond_tes), K, D, D))
for i in range(len(param_cond_tes)):
    weight_r[i]   = weight_func(param_cond_tes[i], K)
    means_r[i]    = means_func(param_cond_tes[i], K, D)
    covars_r[i]   = covar_func(param_cond_tes[i], K, D)

param_cond_tes = param_cond_tes.numpy()

all_figures(K, gmm, sample_func, data_t, means0_t, weight_tes, covars_tes, means_tes, param_cond_tes, means_r, covars_r, weight_r)

print(f'KL divergense = {train_loss + log_true.numpy()}')