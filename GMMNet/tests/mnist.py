import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans



import sys
# TODO: remove this once we make a package
sys.path.append("..") # Adds higher directory to python modules path.
from models.model import GMMNet

# load the data
digits = load_digits().data
labels = load_digits().target

n_clusters = 20    # number of gaussian components 
n_pca_components = 15  # dimension of the PCA subspace

# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=n_pca_components, whiten=True)
data = pca.fit_transform(digits)
# add homoscedastic noise to the data
# data = np.random.randn(*data.shape)

data = torch.tensor(data).float()
labels = torch.tensor(labels).unsqueeze(1).float()
# normalize labels
labels_mean = labels.mean()
labels_std = labels.std()
labels = (labels - labels_mean)/labels_std
# synthesize homoscedastic noise
data = data + torch.randn_like(data)
noise_covar = torch.diag(torch.ones(n_pca_components))
noise_covars = torch.stack([noise_covar for _ in range(data.shape[0])])

# kmeans to classify each data point
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

# initialization
xd_gmm = GMMNet(n_components=n_clusters, 
             data_dim=n_pca_components, 
             conditional_dim=1, 
             cluster_centroids=torch.tensor(kmeans.cluster_centers_).float(), 
             vec_dim=128,
             num_embedding_layers=3,
             num_weights_layers=1,
             num_means_layers=1,
             num_covar_layers=1)
# initialization
basic_gmm = GMMNet(n_components=n_clusters, 
             data_dim=n_pca_components, 
             conditional_dim=1, 
             cluster_centroids=torch.tensor(kmeans.cluster_centers_).float(), 
             vec_dim=128,
             num_embedding_layers=3,
             num_weights_layers=1,
             num_means_layers=1,
             num_covar_layers=1)


learning_rate = 1e-3
xd_optimizer = torch.optim.Adam(xd_gmm.parameters(), lr=learning_rate, weight_decay=0.001)
xd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(xd_optimizer, factor=0.4, patience=2)
basic_optimizer = torch.optim.Adam(basic_gmm.parameters(), lr=learning_rate, weight_decay=0.001)
basic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(basic_optimizer, factor=0.4, patience=2)

batch_size = 128
train_loader = DataLoader(TensorDataset(data, labels, noise_covars), batch_size=batch_size, shuffle=True)

epoch = 150
try:
    for n in range(epoch):
        xd_train_loss = 0
        basic_train_loss = 0
        for i, (dig, lab, noise) in enumerate(train_loader):
            xd_optimizer.zero_grad()
            basic_optimizer.zero_grad()
            
            xd_loss = -xd_gmm.log_prob_b(dig, lab, noise=noise).mean()
            xd_train_loss += xd_loss.item()
            
            basic_loss = -basic_gmm.log_prob_b(dig, lab).mean()
            basic_train_loss += basic_loss.item()

            # backward and update parameters
            xd_loss.backward()
            xd_optimizer.step()
            
            basic_loss.backward()
            basic_optimizer.step()
        
        xd_train_loss = xd_train_loss / dig.shape[0]
        basic_train_loss = basic_train_loss / dig.shape[0]
        print(f'Epoch {n+1}: Basic loss: {basic_train_loss:.3f}   XD loss: {xd_train_loss:.3f}')
        xd_scheduler.step(xd_train_loss)
        basic_scheduler.step(basic_train_loss)

except KeyboardInterrupt:
    pass


def plot_digits(sample_digit=0):
    # sample 44 new points from the data
    with torch.no_grad():
        context = sample_digit * torch.ones(44, 1)
        context = (context - labels_mean)/labels_std  # normalize context
        xd_data = xd_gmm.sample(context)
        basic_data = basic_gmm.sample(context)
        
    noisy_data = pca.inverse_transform(data.numpy())
    xd_data = pca.inverse_transform(xd_data.numpy())
    basic_data = pca.inverse_transform(basic_data.numpy())

    # turn data into a 4x11 grid
    xd_data = xd_data.reshape((4, 11, -1))
    basic_data = basic_data.reshape((4, 11, -1))
    # real_data = digits[:44].reshape((4, 11, -1))
    real_data = noisy_data[:44].reshape((4, 11, -1))


    # plot real digits and resampled digits
    fig, ax = plt.subplots(14, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
        ax[4, j].set_visible(False)
        ax[9, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(basic_data[i, j].reshape((8, 8)),
                                    cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 10, j].imshow(xd_data[i, j].reshape((8, 8)),
                                    cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title(f'Conditional samples drawn from basic model (context={int(sample_digit)})')
    ax[10, 5].set_title(f'Conditional samples drawn from xd model (context={int(sample_digit)})')

    plt.show()

plot_digits(sample_digit=0)