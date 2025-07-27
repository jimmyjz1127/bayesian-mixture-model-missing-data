import numpy as np
# from abc import ABC, abstractmethod
import random
from sklearn.metrics import adjusted_rand_score
from scipy.special import logsumexp

class BMMEM:
    def __init__(self, K):
        self.rng = np.random.default_rng(5099)
        self.K = K

    
    def e_step(self,eps=1e-14):
        N,D = self.X.shape
        K = self.K

        self.R = np.zeros((N,K))

        if not self.missing:
            self.R = np.log(self.π + eps) + (self.X @ np.log(self.θ + eps).T + (1 - self.X) @ np.log(1 - self.θ + eps).T)
        else:
            for i in range(N):
                for k in range(K):
                    mask = ~self.missing_mask[i]

                    self.R[i,k] = (np.log(self.π[k] + eps) + 
                                    np.sum((self.X[i] * np.log(self.θ[k] + eps))[mask]) + 
                                    np.sum(((1 - self.X[i]) * np.log(1 - self.θ[k] + eps))[mask]))

        log_norm = logsumexp(self.R, axis=1, keepdims=True)
        self.R = np.exp(self.R - log_norm)

        loglik = np.sum(log_norm) / N

        return loglik
    
    def m_step(self):
        K = self.K

        N,D = self.X.shape
        obs_mask = ~self.missing_mask

        nk = self.R.sum(axis=0)

        x = self.X
        if self.missing:
            x = self.X.copy()
            x[self.missing_mask] = (self.R@self.θ)[obs_mask]

        self.θ = (self.R.T @ x)/nk[:,None]
        self.π = nk/N

    
    def fit(self, X, max_iters=100, tol=1e-4):
        N,D = X.shape

        self.X = X

        K = self.K

        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)

        self.R = np.random.dirichlet(alpha=np.full(K, 1/K), size=N)
        self.θ = np.random.uniform(0,1,size=(K,D))

        loglikes = []

        self.m_step()

        for _ in range(0,max_iters):
            ll = self.e_step()

            self.m_step()

            self.z = np.argmax(self.R, axis=1)
            loglikes.append(ll)

            if len(loglikes) > 1 and np.abs(loglikes[-1] - loglikes[-2]) < tol : break

        return self.z,self.π,self.θ,loglikes
    
    def posterior_predictive(self, X_new):
        """
            Parameters:
                X_new : (N,D) array with missing values (np.nan)
                θ     : (K,D) learned Bernoulli means
                π     : (K,)  learned mixing weights

        """
        N, D = X_new.shape
        missing_mask = np.isnan(X_new)
        eps = 1e-14

        R = np.zeros((N, self.K))

        for i in range(N):
            for k in range(self.K):
                mask = ~missing_mask[i]
                logp = np.log(self.π[k] + eps)
                logp += np.sum((X_new[i] * np.log(self.θ[k] + eps))[mask])
                logp += np.sum(((1 - X_new[i]) * np.log(1 - self.θ[k] + eps))[mask])
                R[i, k] = logp

        log_norm = logsumexp(R, axis=1, keepdims=True)
        R = np.exp(R - log_norm)

        X_filled = X_new.copy()
        for i in range(N):
            for d in range(D):
                if missing_mask[i, d]:
                    X_filled[i, d] = np.sum(R[i] * self.θ[:, d])

        return X_filled