import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

from ..models.model import GMMNet

# load the data
digits = load_digits().data
labels = load_digits().target

n_clusters = 10
n_components = 15

# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=n_components, whiten=True)
data = pca.fit_transform(digits)
# add homoscedastic noise to the data
data = np.random.randn(*data.shape)


# kmeans to classify each data point
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

# initialization
gmm = GMMNet(n_clusters=n_clusters, 
             data_dim=n_components, 
             conditional_dim=1, 
             cluster_centroids=kmeans.cluster_centers_)

learning_rate = 1e-3
optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=2)

epoch = 5
try:
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

except KeyboardInterrupt:
    pass

















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
