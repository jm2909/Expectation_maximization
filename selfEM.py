import scipy
import  numpy as np
import pandas as pd
from scipy import stats
class singleseriesGassianMixture():
    def __init__(self,series,n_components):
        self.series = series
        self.n_components  =n_components
    def __normaldensity__(self,value,mean,sigma):
        return stats.norm.pdf(value,mean,sigma)
    def __initializer(self):
        mu = np.zeros(self.n_components)
        sd = np.ones(self.n_components)
        sum = np.zeros(self.n_components)
        count = np.zeros(self.n_components)
        sumsq = np.zeros(self.n_components)
        return mu,sd,sum,count,sumsq

    def __Expectation__(self,value,mu_array,sigma_array):
        ncompo = mu_array.shape[0]
        prob = np.array([self.__normaldensity__(value,mu_array[j],sigma_array[j]) for j in range(0,ncompo)])
        label = np.argmax(prob)
        return label

    def __maximization__(self,value,mu_array,sigma_array,sum_array,count,sumsq,labellist):
        cluster = self.__Expectation__(value,mu_array,sigma_array)
        labellist.append(cluster)
        sum_array[cluster] = sum_array[cluster] + value
        count[cluster] = count[cluster] + 1
        mu_array[cluster] = sum_array[cluster] / float(count[cluster])
        sumsq[cluster] = sumsq[cluster]+(value**2)
        if count[cluster] == 1:
            sigma_array[cluster]  = 1
        else:
            sigma_array[cluster] = np.sqrt((sumsq[cluster]/float(count[cluster]) - (mu_array[cluster]**2)))
        return mu_array,sigma_array,sum_array,count,sumsq,labellist

    def __EM__(self):
        lists = []
        mu, sd, sum, count, sumsq = self.__initializer()
        for xx in self.series:
            mu_array, sigma_array, sum_array, count, sumsq, labellist = self.__maximization__(xx, mu_array=mu,
                                                                                         sigma_array=sd, sum_array=sum,
                                                                                         count=count, sumsq=sumsq,
                                                                                         labellist=lists)
        return mu_array, sigma_array, sum_array, count, sumsq,pd.DataFrame(np.array(labellist), columns=['Labels'])

## How to apply ####
# component = 3
# x = np.hstack((np.random.normal(0,1,100),np.random.normal(30,6,100),np.random.normal(8,2,100)))
# np.random.shuffle(x)
# em =  singleseriesGassianMixture(series=x,n_components = component)
# mu_array, sigma_array, sum_array, count, sumsq,lables = em.__EM__()


class __MultivariateGaussianpdf():
    def __init__(self,matrix,components):
        self.series = matrix
        self.n_components = components
    def __multivariatenormaldensity__(self,row,mean,sigma):
        return stats.multivariate_normal.pdf(row,mean,sigma)

    def __initializer(self):
        mu = np.zeros((self.n_components,self.series.shape[1]))
        sd = list(np.random.uniform(3,0.05,(self.n_components*self.series.shape[1]*self.series.shape[1])))
        sum = np.zeros((self.n_components,self.series.shape[1]))
        count =np.zeros((self.n_components))
        sumsq = np.zeros((self.n_components,self.series.shape[1]))
        return mu,sd,sum,count,sumsq
    def __Expectation__(self,row,mu_array,sigma_array):
        ncompo = mu_array.shape[0]
        prob = np.array([self.__multivariatenormaldensity__(row,mu_array[j,:],sigma_array[j]) for j in range(0,ncompo)])
        label = np.argmax(prob)
        return label
    def __maximization__(self,row,mu_array,sigma_array,sum_array,count,sumsq,labellist):
        cluster = self.__Expectation__(row,mu_array,sigma_array)
        labellist.append(cluster)
        sum_array[cluster,:] = sum_array[cluster] + row
        count[cluster] = count[cluster] + 1
        mu_array[cluster,:] = sum_array[cluster,:] / float(count[cluster])
        sumsq[cluster] = sumsq[cluster,:]+(row**2)
        if count[cluster] == 1:
            sigma_array[cluster]  = sigma_array[cluster] + 1
        else:
            sigma_array[cluster] = np.sqrt((sumsq[cluster,:]/float(count[cluster]) - (mu_array[cluster,:]**2)))
        return mu_array,sigma_array,sum_array,count,sumsq,labellist

    def __EM__(self):
        lists = []
        mu, sd, sum, count, sumsq = self.__initializer()
        for x in range(0,self.series.shape[0]):
            xx = self.series[x,:]
            mu_array, sigma_array, sum_array, count, sumsq, labellist = self.__maximization__(xx, mu_array=mu,
                                                                                                  sigma_array=sd,
                                                                                                  sum_array=sum,
                                                                                                  count=count,
                                                                                                  sumsq=sumsq,
                                                                                                  labellist=lists)
        return mu_array, sigma_array, sum_array, count, sumsq, pd.DataFrame(np.array(labellist), columns=['Labels'])


