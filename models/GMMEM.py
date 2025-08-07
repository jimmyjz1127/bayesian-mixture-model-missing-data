from scipy.stats import invwishart 
from scipy.stats import multivariate_normal 
from scipy.special import logsumexp
import numpy as np
import random 
from sklearn.cluster import KMeans
from utils.ArbitraryImputer import mean_impute


class GMMEM:
    def __init__(self, K, complete_case=False):
        self.rng = np.random.default_rng(5099)
        self.K = K
        self.fitted = False
        self.complete_case = complete_case

    def e_step(self, eps=1e-14):
        N,D = self.X.shape
        K = self.K
        self.R = np.zeros((N,K))

        if not self.missing:
            for k in range(K):
                Σ = 0.5 * (self.Σ[k] + self.Σ[k].T) + (1e-6 * np.eye(self.Σ[k].shape[0]))
                self.R[:,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(self.X, self.μ[k], Σ,allow_singular=True)
        else:
            for i in range(N):
                obs_mask = ~self.missing_mask[i]

                if not np.any(obs_mask) or (self.complete_case and np.any(self.missing_mask[i])):
                    self.R[i,:] = np.log(self.π + eps)
                    continue

                for k in range(K):
                    μ_o = self.μ[k][obs_mask]
                    Σ_oo = self.Σ[k][np.ix_(obs_mask, obs_mask)]
                    Σ_oo = 0.5 * (Σ_oo + Σ_oo.T) + (1e-6 * np.eye(Σ_oo.shape[0]))

                    self.R[i,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(self.X[i,obs_mask],μ_o,Σ_oo,allow_singular=True)

        log_norm = logsumexp(self.R, axis=1, keepdims=True)
        self.R = np.exp(self.R - log_norm)
        self.z = np.argmax(self.R, axis=1)

        # loglik = np.sum(log_norm) / N
        if self.complete_case:
            valid_rows = ~np.any(self.missing_mask, axis=1)
            loglik = np.sum(log_norm[valid_rows]) / np.sum(valid_rows)
        else:
            loglik = np.sum(log_norm) / N

        return loglik
    
    def m_step(self, eps=1e-10):
        N,D = self.X.shape
        K = self.K

        if self.complete_case:
            valid_rows = ~np.any(self.missing_mask, axis=1)
            X = self.X[valid_rows]
            R = self.R[valid_rows]
            nk = R.sum(axis=0)
            nk_safe = np.maximum(nk, eps)

            self.μ = (R.T @ X) / nk_safe[:, None]
            
            diff = X[:, None, :] - self.μ[None, :, :]
            outer = diff[:, :, :, None] * diff[:, :, None, :]
            weighted_outer = R[:, :, None, None] * outer
            self.Σ = weighted_outer.sum(axis=0) / nk_safe[:, None, None]
            self.π = nk_safe / nk_safe.sum()
            return

        nk = self.R.sum(axis=0)
        nk_safe = np.maximum(nk, eps) # regularize for safety

        # self.π = nk / N
        self.π = nk_safe / nk_safe.sum()

        if not self.missing: # if complete data
            self.μ = (self.R.T @ self.X)/nk_safe[:,None]
            diff = self.X[:, None, :] - self.μ[None, :, :]
            outer = diff[:, :, :, None] * diff[:, :, None, :]
            weighted_outer = self.R[:, :, None, None] * outer  # (N, K, D, D)
            self.Σ = weighted_outer.sum(axis=0) / nk_safe[:, None, None]
        else: # missing data
            new_μs = np.zeros((K,D))
            new_Σs = np.zeros((K,D,D))

            for i in range(N):
                miss_mask = self.missing_mask[i]
                obs_mask = ~miss_mask

                if not np.any(obs_mask):
                    continue

                for k in range(K):
                    μ_h = self.μ[k][miss_mask]
                    μ_o = self.μ[k][obs_mask]
                    Σ_oh = self.Σ[k][np.ix_(obs_mask, miss_mask)]
                    Σ_ho = self.Σ[k][np.ix_(miss_mask, obs_mask)]
                    Σ_oo = self.Σ[k][np.ix_(obs_mask, obs_mask)]
                    Σ_hh = self.Σ[k][np.ix_(miss_mask, miss_mask)]

                    Σ_oo = 0.5 * (Σ_oo + Σ_oo.T) + (1e-6 * np.eye(Σ_oo.shape[0])) # regularization
                    Σ_oo_inv = np.linalg.inv(Σ_oo)

                    m_i = μ_h + Σ_ho @ Σ_oo_inv @ (self.X[i,obs_mask] - μ_o)
                    V_i = Σ_hh - Σ_ho @ Σ_oo_inv @ Σ_oh

                    x_hat = self.X[i].copy()
                    x_hat[miss_mask] = m_i
                    new_μs[k] += self.R[i,k] * x_hat 

                    x_hats_outer = np.outer(x_hat, x_hat)
                    if np.any(miss_mask):
                        x_hats_outer[np.ix_(miss_mask, miss_mask)] += V_i 

                    new_Σs[k] += self.R[i, k] * x_hats_outer

            new_μs /= nk_safe[:, None]
            for k in range(K):
                new_Σs[k] /=  nk_safe[k] 
                new_Σs[k] -= new_μs[k][:, None] @ new_μs[k][:, None].T
            
            self.μ = new_μs
            self.Σ = new_Σs

    def init_params(self, X, eps=1e-10):
        N,D = X.shape
        K = self.K
        nk = self.R.sum(axis=0)
        nk_safe = np.maximum(nk, eps)

        self.μ = (self.R.T @ X)/nk_safe[:,None]
        diff = X[:, None, :] - self.μ[None, :, :]
        outer = diff[:, :, :, None] * diff[:, :, None, :]
        weighted_outer = self.R[:, :, None, None] * outer  # (N, K, D, D)
        self.Σ = weighted_outer.sum(axis=0) / nk_safe[:, None, None]

        self.π = nk_safe / N
    

    def fit(self,X,max_iters=200,tol=1e-4):
        self.X = X 
        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)

        N,D = X.shape
        K = self.K

        z = np.random.choice(K, size=N)
        self.R = np.zeros((N, K))
        self.R[np.arange(N), z] = 1
        self.init_params(mean_impute(self.X))

        loglikes = []

        for t in range(0,max_iters):  
            self.m_step()  
            ll = self.e_step()        
            loglikes.append(ll)
            if t > 1 and np.abs(loglikes[-1] - loglikes[-2]) < tol : break

        self.fitted = True
        return {
            'z':self.z,
            'R':self.R,
            'μ':self.μ,
            'Σ':self.Σ,
            'π':self.π,
            'loglike' :loglikes
        } 
    
    def compute_responsibility(self, X_new, eps=1e-10):
        N,D = X_new.shape
        K = self.K

        missing_mask = np.isnan(X_new)
        missing = np.any(missing_mask)

        cond_means = np.array([[None]*K for _ in range(N)])
        cond_covs = np.array([[None]*K for _ in range(N)])
        R = np.zeros((N,K))

        if not missing:
            for k in range(K):
                R[:,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(X_new, self.μ[k], self.Σ[k], allow_singular=True)
        else:
            for i in range(N):
                miss_mask = missing_mask[i]
                obs_mask = ~miss_mask

                for k in range(K):
                    if not np.any(obs_mask)or (self.complete_case and np.any(~self.missing_mask[i])):
                        cond_means[i, k] = self.μ[k]
                        cond_covs[i, k] = self.Σ[k]
                        R[i,k] = np.log(self.π[k] + eps)
                    else:
                        μ_h = self.μ[k][miss_mask]
                        μ_o = self.μ[k][obs_mask]
                        Σ_oh = self.Σ[k][np.ix_(obs_mask, miss_mask)]
                        Σ_ho = self.Σ[k][np.ix_(miss_mask, obs_mask)]
                        Σ_oo = self.Σ[k][np.ix_(obs_mask, obs_mask)]
                        Σ_hh = self.Σ[k][np.ix_(miss_mask, miss_mask)]

                        Σ_oo = 0.5 * (Σ_oo + Σ_oo.T) + (1e-6 * np.eye(Σ_oo.shape[0]))
                        Σ_oo_inv = np.linalg.inv(Σ_oo)

                        m_i = μ_h + Σ_ho @ Σ_oo_inv @ (X_new[i,obs_mask] - μ_o)
                        V_i = Σ_hh - Σ_ho @ Σ_oo_inv @ Σ_oh
                        
                        cond_means[i,k] = m_i
                        cond_covs[i,k] = V_i

                        R[i,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(X_new[i,obs_mask],μ_o,Σ_oo,allow_singular=True)

        invalid_rows = ~np.any(np.isfinite(R), axis=1)
        if np.any(invalid_rows):
            R[invalid_rows] = np.log(np.ones(K) / K)

        log_norm = logsumexp(R, axis=1, keepdims=True)
        R = np.exp(R - log_norm)

        loglike = np.sum(log_norm) / N
        return R, cond_means, cond_covs, loglike
    
    def posterior_predict(self, X_new, eps=1e-14):
        ''' 
            Imputes missing entries using expectation 
        '''
        N,D = X_new.shape
        K = self.K

        missing_mask = np.isnan(X_new)

        if not np.any(missing_mask):
            return X_new
        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape[1] != self.X.shape[1]:
            raise Exception("Dimensions do not match fit.")

        R, cond_means, cond_covs, _ = self.compute_responsibility(X_new)

        X_filled = X_new.copy()
        for i in range(N):
            miss_mask = missing_mask[i]
            X_filled[i][miss_mask] = np.sum(R[i] * cond_means[i])

        return X_filled
        
    def predict(self, X_new):
        R, _, _, _ = self.compute_responsibility(X_new)
        return np.argmax(R, axis=1)
    
    def log_likelihood(self, X_new):
        R, _, _, ll = self.compute_responsibility(X_new)
        return ll



