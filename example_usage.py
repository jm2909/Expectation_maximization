# How to apply ####
import numpy as np

from selfEM import __MultivariateGaussianpdf,singleseriesGassianMixture

component = 4
x = np.hstack((np.random.normal(0,1,100),np.random.normal(30,6,100)))
y = np.hstack((np.random.normal(0,13,100),np.random.normal(11,2,100)))

z = np.hstack((np.random.normal(0,5,100),np.random.normal(13,2,100)))
test_array = np.transpose(np.vstack((x,y,z)))

np.random.shuffle(test_array)
em =  __MultivariateGaussianpdf(matrix=test_array,components = 5)
mu_array, sigma_array, sum_array, count, sumsq,lables = em.__EM__()
print("Mu:",mu_array, "Sigma:",sigma_array)

em1 = singleseriesGassianMixture(x,n_components=3)
mu_array, sigma_array, sum_array, count, sumsq,lables = em1.__EM__()
print("Mu:",mu_array, "Sigma:",sigma_array)