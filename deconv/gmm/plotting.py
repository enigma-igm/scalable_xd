import numpy as np
import matplotlib as mpl


def plot_covariance(mean, cov, ax, alpha=0.5, color=None):
    """
    Plot a Gaussian convariance projected on the 1st and 2nd dimension on a Matplotlib axis.

    Adapted from https://scikit-learn.org/stable/auto_examples/
    mixture/plot_gmm_covariances.html

    Parameters
    ----------
    mean : 1D array, length D
        Mean value of a component of GMM.
    cov : 2D array, shape (D, D)
        Covariance of a component of GMM.
    ax : .axes.Axes
        The plot axes.
    alpha : float, default=0.5.
        Transparency, from 0 to 1, in which 0 is completely transparent.
    color : string, default=None
        Color of the ellipse.
    """
    v, w = np.linalg.eigh(cov) # v is the eigenvalues (D) and w is the eigenvectors (D,D)
    u = w[0] / np.linalg.norm(w[0]) # normalize the 1st eigenvector
    angle = np.arctan2(u[1], u[0]) * 180 / np.pi # the position angle
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mean, # xy
        v[0], # width
        v[1], # height
        180 + angle, # angle
        color=color  # color
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ax.add_artist(ell)
