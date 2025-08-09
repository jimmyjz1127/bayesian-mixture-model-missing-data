import numpy as np
# from abc import ABC, abstractmethod
import random
from sklearn.metrics import adjusted_rand_score
from scipy.special import logsumexp

class BMMEM:
    def __init__(self, K, complete_case=False):
        """
            Parameters 
                K             : number of components 
                complete_case : whether to delete (ignore) incomplete datapoints
        """

        self.rng = np.random.default_rng(5099)
        self.K = K
        self.fitted = False
        self.complete_case = complete_case

    def e_step(self,eps=1e-14):
        N,D = self.X.shape
        K = self.K
        self.R = np.zeros((N,K))

        if not self.missing:
            self.R = np.log(self.π + eps) + (self.X @ np.log(self.θ + eps).T + (1 - self.X) @ np.log(1 - self.θ + eps).T)
        else:
            for i in range(N):
                mask = ~self.missing_mask[i]
            
                if not np.any(mask) or (self.complete_case and np.any(self.missing_mask[i])):
                    self.R[i,:] = np.log(self.π + eps)
                    continue

                for k in range(K):
                    x_obs = self.X[i][mask]
                    θ_obs = self.θ[k][mask]

                    self.R[i,k] = (
                        np.log(self.π[k] + eps) + 
                        np.sum(x_obs * np.log(θ_obs + eps)) + 
                        np.sum((1 - x_obs) * np.log(1 - θ_obs + eps))
                    )

        log_norm = logsumexp(self.R, axis=1, keepdims=True)
        self.R = np.exp(self.R - log_norm)

        loglik = np.sum(log_norm) / N
        # if self.complete_case:
        #     valid_rows = ~np.any(self.missing_mask, axis=1)
        #     loglik = np.sum(log_norm[valid_rows]) / np.sum(valid_rows)
        # else:
        #     loglik = np.sum(log_norm) / N

        return loglik
    

    def m_step(self, eps=1e-10):
        N,D = self.X.shape
        if self.complete_case:
            # Filter to only complete rows
            complete_rows = ~np.any(self.missing_mask, axis=1)
            x = self.X[complete_rows]
            R = self.R[complete_rows]
        else:
            x = self.X.copy()
            R = self.R.copy()
            if self.missing:
                x[self.missing_mask] = (self.R @ self.θ)[self.missing_mask]

        nk = R.sum(axis=0)
        nk_safe = np.maximum(nk, eps)

        self.θ = (R.T @ x) / nk_safe[:, None]
        self.π = nk_safe / nk_safe.sum()

    
    def fit(self, X, max_iters=200, tol=1e-4):
        N,D = X.shape

        self.X = X

        K = self.K

        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)
        
        z = np.random.choice(K, size=N)
        self.R = np.zeros((N, K))
        self.R[np.arange(N), z] = 1
        self.θ = np.random.uniform(0.1,0.9,size=(K,D))

        loglikes = []

        for _ in range(0,max_iters):
            self.m_step()
            ll = self.e_step()

            self.z = np.argmax(self.R, axis=1)
            loglikes.append(ll)

            if len(loglikes) > 1 and np.abs(loglikes[-1] - loglikes[-2]) < tol : break

        self.fitted = True
        return {
            'z' : self.z,
            'π' : self.π,
            'θ' : self.θ,
            'loglike' : loglikes
        }
    
    def compute_responsibility(self, X_new, eps=1e-14):
        """
            Parameters:
                X_new : (N,D) array with missing values (np.nan)
        """
        N, D = X_new.shape

        missing_mask = np.isnan(X_new)
        missing = np.any(missing_mask)

        R = np.zeros((N, self.K))

        if not missing : 
            R = np.log(self.π + eps) + (X_new @ np.log(self.θ + eps).T + (1 - X_new) @ np.log(1 - self.θ + eps).T)
        else:
            for i in range(N):
                mask = ~missing_mask[i]
                x_obs = X_new[i][mask]

                if not np.any(mask)or (self.complete_case and np.any(~self.missing_mask[i])):
                    R[i,:] = np.log(self.π + eps)
                    continue
                
                for k in range(self.K):
                    θ_obs = self.θ[k][mask]
                    
                    logp = np.log(self.π[k] + eps)
                    logp += np.sum(x_obs * np.log(θ_obs + eps))
                    logp += np.sum((1 - x_obs) * np.log(1 -θ_obs + eps))
                    R[i, k] = logp

        log_norm = logsumexp(R, axis=1, keepdims=True)
        R = np.exp(R - log_norm)

        loglike = np.sum(log_norm)/N

        return R,loglike
    
    def impute(self, X_new, eps=1e-14):
        N,D = X_new.shape
        missing_mask = np.isnan(X_new)

        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape[1] != self.X.shape[1]:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new

        R,_ = self.compute_responsibility(X_new)

        X_filled = X_new.copy()
        for i in range(N):
            for d in range(D):
                if missing_mask[i, d]:
                    X_filled[i, d] = np.sum(R[i] * self.θ[:, d])

        return X_filled
    

    def log_likelihood(self, X_new):
        return self.compute_responsibility(X_new)[1]
    
    def predict(self, X_new):
        R,_ = self.compute_responsibility(X_new)
        return np.argmax(R, axis=1)


