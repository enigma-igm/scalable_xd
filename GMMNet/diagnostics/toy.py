import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
import torch


def all_figures(K, 
                gmm, 
                sample_func,                
                data_t, 
                means0_t, 
                weight_tes, 
                covars_tes, 
                means_tes, 
                param_cond_tes, 
                means_r, 
                covars_r, 
                weight_r, 
                ):
    # figure. all samples and initial guess of the means
    data_t = data_t.numpy().reshape(-1, 2)
    fig, ax = plt.subplots()
    pd_t = ax.scatter(*data_t.transpose(), marker='.', color='grey', alpha=0.5, label='Training Set')
    pd_k = ax.scatter(*means0_t.transpose(), s=80, marker='.', color='tab:orange', label='kmeans centroids')
    ax.set_title('Training Set', fontsize=14)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(fontsize=14)

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

    # figure. means, and some samples, and learned means at certain condtional parameter
    Nr = 1000
    cond_array = [9, 49, 89]
    param_cond_tes = torch.from_numpy(param_cond_tes)

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

