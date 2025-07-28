import numpy as np
import pandas as pd

from scipy.special import gammaln, psi
from scipy.special import logsumexp

from abc import ABC, abstractmethod
import random
from sklearn.metrics import adjusted_rand_score

class VBEMModel(ABC):   
    def __init__(self, prior):
        self.fitted = False 
        self.rng = np.random.default_rng(42)
        self.prior = prior
        self.K = prior.K
        self.α_0 = prior.α_0

    @abstractmethod
    def fit(self, X, **kwargs):
        """ 
            To be implemented by subclass
        """
        pass 

    
    def update_z(self,dataLogProb):
        N,D = self.X.shape

        R = np.zeros((N,self.K))

        α_sum = psi(np.sum(self.α))

        for k in range(self.K):
            logprior = psi(self.α[k]) - α_sum

            R[:,k] = dataLogProb[:,k].copy()

            if self.mode == 0:
                R[:,k] += logprior
            elif self.mode == 1:
                R[:,k] += np.log(self.π[k])

        log_norm = logsumexp(R,axis=1,keepdims=True)

        R = np.exp(R - log_norm)

        loglik = np.sum(log_norm) / N

        return R, loglik
    
    def update_π(self):
        self.α = self.α_0 + np.sum(self.R, axis=0) 
    
    def b_func(self, α):
        return np.sum(gammaln(α), axis=0) - gammaln(α.sum())
    
    def kl_pi(self, α):
        return self.b_func(α) - self.b_func(self.α_0) + np.sum((self.α_0 - α) * (psi(α) - psi(α.sum())))
    
    def kl_z(self, R, α):
        return np.sum(R * (psi(α) - psi(α.sum()))[None,:] - (R * np.log(R + 1e-12)))