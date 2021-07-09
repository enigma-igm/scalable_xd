from data import *
from model import *

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




epoch = 500

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
    

embed()

# global check
# conditional parameter covering the training range
param_cond_tes = np.arange(0, 1, 0.01) + 0.01
param_cond_tes = torch.FloatTensor(param_cond_tes.reshape(-1,1))

# get the trained parameters
weight_tes, means_tes, covars_tes = gmm(param_cond_tes)
weight_tes = weight_tes.detach().numpy()
means_tes  = means_tes.detach().numpy()
covarL_tes = covars_tes.detach().numpy()
covars_tes = np.matmul(covarL_tes, np.swapaxes(covarL_tes, -2, -1))

# get the true paramters
weight_r   = np.zeros((len(param_cond_tes), K))
means_r    = np.zeros((len(param_cond_tes), K, D))
covars_r   = np.zeros((len(param_cond_tes), K, D, D))
for i in range(len(param_cond_tes)):
    weight_r[i]   = weight_func(param_cond_tes[i], K)
    means_r[i]    = means_func(param_cond_tes[i], K, D)
    covars_r[i]   = covar_func(param_cond_tes[i], K, D)

param_cond_tes = param_cond_tes.numpy()


# figure. plot means vs conditional parameter
fig, ax = plt.subplots()
norm = plt.Normalize(param_cond_tes.min(), param_cond_tes.max())
for i in range(K):
    points = means_r[:,i,:].reshape(-1, 1, 2)#np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='Blues', norm=norm)
    # Set the values used for colormapping
    lc.set_array(param_cond_tes.flatten())
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
cbar = plt.colorbar(line, ax=ax, aspect=15)
cbar.set_label('True vs Conditional Parameter z', fontsize=10)
for i in range(K):
    points = means_tes[:,i,:].reshape(-1, 1, 2)#np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='Oranges', norm=norm)
    # Set the values used for colormapping
    lc.set_array(param_cond_tes.flatten())
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
cbar = plt.colorbar(line, ax=ax, aspect=15)
cbar.set_label('Predicted vs Conditional Parameter z', fontsize=10)
ax.set_xlim(means_tes[:,:,0].min()-0.5, means_tes[:,:,0].max()+0.5)
ax.set_ylim(means_tes[:,:,1].min()-0.5, means_tes[:,:,1].max()+0.5)
ax.set_title('Means of Each Component', fontsize=14)


# figure. plot weights vs conditional paramters
fig, ax = plt.subplots()
pw_r = ax.plot(param_cond_tes, weight_r, color='tab:blue', label='True')
pw_tes = ax.plot(param_cond_tes, weight_tes, color='tab:orange', label='Predicted')
ax.set_xlabel('Conditional Parameter z', fontsize=14)
ax.set_ylabel('weight', fontsize=14)
ax.set_title(f'Weight of {K} Components', fontsize=14)
customs = [pw_r[0], pw_tes[0]]
ax.legend(customs, [pw_r[0].get_label(), pw_tes[0].get_label()], fontsize=10)


# figure. all samples and initial guess of the means
data_t = data_t.numpy().reshape(-1, 2)
fig, ax = plt.subplots()
pd_t = ax.scatter(*data_t.transpose(), marker='.', color='grey', alpha=0.5, label='Training Set')
pd_k = ax.scatter(*means0_t.transpose(), s=80, marker='.', color='tab:orange', label='kmeans centroids')
ax.set_title('Training Set', fontsize=14)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.legend(fontsize=14)

# figure. Diagonal Element
fig, ax = plt.subplots(2)
for i in range(2):
    pde_r = ax[i].plot(param_cond_tes, covars_r[:,:,i,i], color='tab:blue', label='True')
    pde_tes = ax[i].plot(param_cond_tes, covars_tes[:,:,i,i], color='tab:orange', label='Predicted')
    ax[i].set_title(f'{i+1, i+1} Element of the Covariance Matrix', fontsize=12)
    ax[i].set_xlabel('Conditional Parameter z')
    customs = [pde_r[0], pde_tes[0]]
    ax[i].legend(customs, [pde_r[0].get_label(), pde_tes[0].get_label()], fontsize=10)
plt.tight_layout() 


# specific check
# GMM parameters at a certain conditional parameter
weight_t, means_t, covars_t = gmm(param_cond_v[4,:1])#weight_t,

weight_r   = weight_func(param_cond_v[4,0], K)
means_r    = means_func(param_cond_v[4,0], K, D)
covars_r   = covar_func(param_cond_v[4,0], K, D)

# figure. weights of each component at certain condtional parameter
fig, ax = plt.subplots()
ax.bar(x=np.arange(K)+1, height=weight_r, alpha=0.5)
ax.bar(x=np.arange(K)+1, height=weight_t.detach().numpy()[0], alpha=0.5)
ax.set_title('weights of each component')
ax.set_xticks(np.arange(3)+1)
customs = [Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='k', markersize=5)]
ax.legend(customs, [f'Conditional z={(param_cond_v[4,0].numpy()[0]):.2f}'])



# figure. means, and some samples, and learned means at certain condtional parameter
Nr = 1000
cond_array = [9, 49, 89]
param_cond_tes = torch.from_numpy(param_cond_tes)
embed()
for i in cond_array:
    data_r, _ = sample_func(weight_tes[i], means_tes[i], covars_tes[i], Nr)
    data_t    = gmm.sample(param_cond_tes[i].unsqueeze(0), Nr).squeeze().detach().numpy()
    weight_t, means_t, covars_t = gmm(param_cond_tes[i].unsqueeze(0))

    fig, ax = plt.subplots()
    pm_r = ax.scatter(*means_tes[i].transpose(), label='True')
    pd_r = ax.scatter(*data_r.transpose(), marker='.', color='tab:blue', label='Samples (on true)', alpha=0.1)
    sns.kdeplot(x=data_r[:,0], y=data_r[:,1], color='tab:blue', alpha=0.5)
    pm_t = ax.scatter(*means_t.detach().numpy()[0].transpose(), label='Predicted')
    sns.kdeplot(x=data_t[:,0], y=data_t[:,1], color='tab:orange', alpha=0.5)
    pd_t = ax.scatter(*data_t.transpose(), marker='.', color='tab:orange', label='Samples (on model)', alpha=0.1)
    ax.set_title('Centroid of 3 Components')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    customs = [pm_r, pd_r, pm_t, pd_t,
                Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='k', markersize=5)]
    ax.legend(customs, [pm_r.get_label(), pd_r.get_label(), pm_t.get_label(), pd_r.get_label(),
                    f'Conditional z={(param_cond_tes[i].numpy()[0]):.2f}'], fontsize=10)

plt.show()


print(f'KL divergense = {train_loss + log_true.numpy()}')