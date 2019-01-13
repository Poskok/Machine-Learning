#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 14:53:51 2018

@author: mguina
"""

# The purpose of this example is to show when the Gaussian Mixture Models
# performs better than the K-Means clustering algorithm

# Create first in 1D a simple 3 set of clusters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

mu1 = 12
sigma1 = 2
s1 = np.random.normal(mu1, sigma1, 1000)

mu2 = 5
sigma2 = 1
s2 = np.random.normal(mu2, sigma2, 1000)

s = np.concatenate([s1, s2])

count, bins, ignored = plt.hist(s, 30, normed=True)
plt.axis([0, 20, 0, 0.1])
plt.show()


X = s.reshape(-1, 1)


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

kmeans_labels = kmeans.labels_

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6,
                       random_state=0)


kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

y_corr = y_true
for i in range(1, len(y_true)):
    if (y_true[i-1] == 3):
        y_corr[i-1] = 1
    elif (y_true[i-1] == 2):
        y_corr[i-1] = 0
    elif (y_true[i-1] == 0):
        y_corr[i-1] = 3
    else:
        y_corr[i-1] = 2

np.sum(y_corr == labels)


def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # plot the representation of the k-means model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i,
             center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5,
                                zorder=1))


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if (covariance.shape == (2, 2)):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2*np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle,
                             **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis',
                   zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w*w_factor)


kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)


gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)
