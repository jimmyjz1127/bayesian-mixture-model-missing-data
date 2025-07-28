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
        self.z = (cdf > u).argmax(axis=1)  # return first index where cdf is greater than random u
    
    def sample_π(self):
        ''' 
            Samples mixing weights from Dirichlet distribution parameterized by pseudocounts of components
            
            @param (zs)  : cluster assignments (n)
            @param (α_0) : Dirichlet prior list (K)
            @param (K)   : the number of components
        '''

        z_counts = np.bincount((self.z).astype(np.int64), minlength=self.K)
        self.π = self.rng.dirichlet(self.α_0 + z_counts)   


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
    
    def posterior_predict(self, X_new, sample):
        N, D = X_new.shape
        K = self.K
        missing_mask = np.isnan(X_new)

        if not self.fitted:
            raise Exception("Model has not been fitted yet.")
        if X_new.shape != self.X.shape:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new

        
        π = sample['π'] 
        X_imputed = X_new.copy()

        for i in range(N):
            mask = missing_mask[i]
            obs_mask = ~mask

            # Sample cluster assignment
            z = np.random.choice(K, p=π)

            # For BMM:
            if self.model_type == 'bernoulli':
                θ = sample['θ']
                probs = θ[z]
                X_imputed[i, mask] = np.random.binomial(1, probs[mask])

            # For GMM:
            elif self.model_type == 'gaussian':
                μ = sample['μ'][z]
                Σ = sample['Σ'][z]

                μ_o = μ[obs_mask]
                μ_m = μ[mask]
                Σ_oo = Σ[np.ix_(obs_mask, obs_mask)]
                Σ_om = Σ[np.ix_(obs_mask, mask)]
                Σ_mo = Σ[np.ix_(mask, obs_mask)]
                Σ_mm = Σ[np.ix_(mask, mask)]

                x_obs = X_new[i, obs_mask]
                μ_cond = μ_m + Σ_mo @ np.linalg.inv(Σ_oo) @ (x_obs - μ_o)
                Σ_cond = Σ_mm - Σ_mo @ np.linalg.inv(Σ_oo) @ Σ_om

                X_imputed[i, mask] = np.random.multivariate_normal(μ_cond, Σ_cond)

        return X_imputed