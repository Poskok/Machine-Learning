#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 19:28:38 2018

@author: mguina
"""

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# importing the digits dataset => have a set of 1797 images
# containing each a 8 X 8 matrix of ints, with a value depending on
# the level of grey in this image
from sklearn.datasets import load_digits
digits = load_digits()

# function to plot the images in a format 10 X 10
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8,8), 
                           subplot_kw=dict(xticks=[],yticks=[]))
    fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8,8), cmap='binary')
        im.set_clim(0 , 16)

"""
 ************ PRINCIPAL COMPONENT ANALYSIS ************************
 we will train our GMM algorithm on a set of 1797 images. Each image
 contains 64 variables that take an non-negative integer as value. 
 We will reduce the number of variables using a Principal Component
 Analysis method with an information loss of 1%, filtering out all the 
 variables that do not intervene a lot in the digit identification
"""
from sklearn.decomposition import PCA
pca = PCA(0.99, whiten = True)
data = pca.fit_transform(digits.data)


"""
************ AIC Method ************************
We will now check the number of necessary clusters
to give as input to the GMM method by using the Aikeke
information criterion (AIC) based on the estimation of
the maximum likelihood 
"""

n_component = np.arange(50,210 ,10) #range of number of clusters
models = [GaussianMixture(n, covariance_type = 'full', random_state = 0) for n in n_component] # we define a model per number of clusters
aics = [model.fit(data).aic(data) for model in models]


"""
************ GAUSSIAN MIXTURE MODEL ************************
"""
 
          
plt.plot(n_components, aics)
plot_digits(digits.data)
gmm = GaussianMixture(n_component[np.argmin(aics)], covariance_type='full', random_state = 0)
gmm.fit(data)

data_new = gmm.sample(100)[0]
data_new.shape

digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)

