from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import random 

class GibbsModel(ABC):
    def __init__(self, prior):
        self.fitted = False 
        self.rng = np.random.default_rng(42)
        self.prior = prior
        self.K = prior.K
        self.α_0 = prior.α_0

    @abstractmethod
    def fit(self, X, num_iters=4000, burn=1000):
        """
            To be implemented by subclass 
        """
        pass

    def sampleZ(self, p):
        ''' 
            Samples cluster assignments z for n datapoints 

            @param (p) : softmax categorical probabilities over clusters (N, K)
        '''
        
        # Inverse sample from categorical distribution
        cdf = np.cumsum(p, axis=1) # compute CDF for each row (each categorical distribution)
        u   = self.rng.random(size=(p.shape[0], 1))
        return (cdf > u).argmax(axis=1)  # return first index where cdf is greater than random u
    
    def sample_π(self, z):
        ''' 
            Samples mixing weights from Dirichlet distribution parameterized by pseudocounts of components
            
            @param (zs)  : cluster assignments (n)
            @param (α_0) : Dirichlet prior list (K)
            @param (K)   : the number of components
        '''

        z_counts = np.bincount(z.astype(np.int64), minlength=self.K)
        return self.rng.dirichlet(self.α_0 + z_counts)
    


    def get_map_params(self):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.") 
        
        sample_map = max(self.samples, key=lambda p : p['posterior'])
        return sample_map
    
    def get_summarizing_results(self, y):
        """ 
            Produces summarizing results : mean of ARI, loglikelihood, posterior likelihood
            along with standard deviation for each 
        """
        if not self.fitted:
            raise Exception("Model has not been fitted yet.") 
        
        avg_ari = 0.0
        avg_ll = 0.0
        avg_pl = 0.0

        for sample in self.samples:
            avg_ll += sample['loglike']
            avg_pl += sample['posterior']
            avg_ari += adjusted_rand_score(y,sample['z'])

        N = len(self.samples)

        return {
            'avg_ari' : avg_ari / N,
            'avg_ll'  : avg_ll / N,
            'avg_pl'  : avg_pl / N
        }