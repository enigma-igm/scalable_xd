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

# kmeans to classify each data point
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

# initialization
gmm = GMMNet(n_components=n_clusters, 
             data_dim=n_pca_components, 
             conditional_dim=1, 
             cluster_centroids=torch.tensor(kmeans.cluster_centers_).float(), 
             vec_dim=128,
             num_embedding_layers=3,
             num_weights_layers=1,
             num_means_layers=1,
             num_covar_layers=1)

learning_rate = 1e-3
optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)

batch_size = 128
train_loader = DataLoader(TensorDataset(data, labels), batch_size=batch_size, shuffle=True)

epoch = 150
try:
    for n in range(epoch):
        train_loss = 0
        for i, (dig, lab) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -gmm.log_prob_b(dig, lab).mean()
            train_loss += loss.item()

            # backward and update parameters
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / dig.shape[0]
        print((n+1),'Epoch', train_loss)
        scheduler.step(train_loss)

except KeyboardInterrupt:
    pass


def plot_digits(samp_digit=0):
    # sample 44 new points from the data
    with torch.no_grad():
        context = samp_digit * torch.ones(44, 1)
        context = (context - labels_mean)/labels_std
        new_data = gmm.sample(context)
    new_data = pca.inverse_transform(new_data.numpy())

    # turn data into a 4x11 grid
    new_data = new_data.reshape((4, 11, -1))
    real_data = digits[:44].reshape((4, 11, -1))


    # plot real digits and resampled digits
    fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(11):
        ax[4, j].set_visible(False)
        for i in range(4):
            im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)
            im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                    cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('Selection from the input data')
    ax[5, 5].set_title('"New" digits drawn from the kernel density model')

    plt.show()

plot_digits(samp_digit=0)