from data import *
from model import *

import matplotlib.pyplot as plt

from more_itertools import chunked

from sklearn.cluster import KMeans

import os

from IPython import embed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

K, D, D_cond = 3, 2, 9

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
                   covariance_matrix=torch.from_numpy(covars_t[i][None, :])).log_prob(torch.from_numpy(data_i[None, None, :]))

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




epoch = 2000

for n in range(epoch):
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
    

weight_t, means_t, covars_t = gmm(param_cond_v[4,:1])#weight_t,

weight_r   = weight_func(param_cond_v[4,0], K)
means_r    = means_func(param_cond_v[4,0], K, D)
covars_r   = covar_func(param_cond_v[4,0], K, D)

Nr = 100
data_r = sample_func(weight_r, means_r, covars_r, Nr)


fig, ax = plt.subplots()
ax.bar(x=np.arange(K)+1, height=weight_r, alpha=0.5)
ax.bar(x=np.arange(K)+1, height=weight_t.detach().numpy()[0], alpha=0.5)
ax.set_title('weights of each component')


fig, ax = plt.subplots()
ax.scatter(*means_r.transpose())
ax.scatter(*data_r[0].transpose(), marker='.', color='grey', alpha=0.5)
ax.scatter(*means_t.detach().numpy()[0].transpose())
ax.set_title('centroid of 3 Components')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
#plt.xlim([100, 160])
plt.show()

print(train_loss + log_true.numpy())