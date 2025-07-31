from scipy.stats import invwishart 
from scipy.stats import multivariate_normal 
from scipy.special import logsumexp
import numpy as np
import random 
from sklearn.cluster import KMeans


class GMMEM:
    def __init__(self, K):
        self.rng = np.random.default_rng(5099)
        self.K = K
        self.fitted = False

    def e_step(self, eps=1e-14):
        N,D = self.X.shape
        K = self.K
        self.R = np.zeros((N,K))

        if not self.missing:
            for k in range(K):
                self.R[:,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(self.X, self.μ[k], self.Σ[k],allow_singular=True)
        else:
            for i in range(N):
                obs_mask = ~self.missing_mask[i]

                for k in range(K):
                    μ_o = self.μ[k][obs_mask]
                    Σ_oo = self.Σ[k][np.ix_(obs_mask, obs_mask)]
                    Σ_oo = 0.5 * (Σ_oo + Σ_oo.T) + (1e-6 * np.eye(Σ_oo.shape[0]))

                    self.R[i,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(self.X[i,obs_mask],μ_o,Σ_oo,allow_singular=True)

        log_norm = logsumexp(self.R, axis=1, keepdims=True)
        self.R = np.exp(self.R - log_norm)
        self.z = np.argmax(self.R, axis=1)

        loglik = np.sum(log_norm) / N
        return loglik
    
    def m_step(self, eps=1e-10):
        N,D = self.X.shape
        K = self.K
        nk = self.R.sum(axis=0)
        nk_safe = np.maximum(nk, eps)

        self.π = nk / N

        if not self.missing:
            self.μ = (self.R.T @ self.X)/nk_safe[:,None]
            diff = self.X[:, None, :] - self.μ[None, :, :]
            outer = diff[:, :, :, None] * diff[:, :, None, :]
            weighted_outer = self.R[:, :, None, None] * outer  # (N, K, D, D)
            self.Σ = weighted_outer.sum(axis=0) / nk_safe[:, None, None]
        else:
            new_μs = np.zeros((K,D))
            new_Σs = np.zeros((K,D,D))

            for i in range(N):
                miss_mask = self.missing_mask[i]
                obs_mask = ~miss_mask

                for k in range(K):
                    μ_h = self.μ[k][miss_mask]
                    μ_o = self.μ[k][obs_mask]
                    Σ_oh = self.Σ[k][np.ix_(obs_mask, miss_mask)]
                    Σ_ho = self.Σ[k][np.ix_(miss_mask, obs_mask)]
                    Σ_oo = self.Σ[k][np.ix_(obs_mask, obs_mask)]
                    Σ_hh = self.Σ[k][np.ix_(miss_mask, miss_mask)]

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
                self.Σ[k] += eps * np.eye(D)
            
            self.μ = new_μs
            self.Σ = new_Σs

    
    def mean_impute(self, X, missing_mask):
        X_0 = X.copy()
        means = np.nanmean(np.where(missing_mask, np.nan, X), axis=0)
        X_0[missing_mask] = np.take(means, np.where(missing_mask)[1])
        return X_0
    
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

        # self.μ = np.zeros((K,D)) + np.random.gamma(1.0, 0.1, size=(K, D))
        # self.Σ = np.array([1e-6 * np.eye(D) for _ in range(K)])
        z = np.random.choice(K, size=N)
        self.R = np.zeros((N, K))
        self.R[np.arange(N), z] = 1
        self.init_params(self.mean_impute(self.X, self.missing_mask))

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
                    μ_h = self.μ[k][miss_mask]
                    μ_o = self.μ[k][obs_mask]
                    Σ_oh = self.Σ[k][np.ix_(obs_mask, miss_mask)]
                    Σ_ho = self.Σ[k][np.ix_(miss_mask, obs_mask)]
                    Σ_oo = self.Σ[k][np.ix_(obs_mask, obs_mask)]
                    Σ_hh = self.Σ[k][np.ix_(miss_mask, miss_mask)]

                    Σ_oo_inv = np.linalg.inv(Σ_oo)

                    m_i = μ_h + Σ_ho @ Σ_oo_inv @ (X_new[i,obs_mask] - μ_o)
                    V_i = Σ_hh - Σ_ho @ Σ_oo_inv @ Σ_oh
                    
                    cond_means[i,k] = m_i
                    cond_covs[i,k] = V_i

                    R[i,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(X_new[i,obs_mask],μ_o,Σ_oo,allow_singular=True)

        log_norm = logsumexp(R, axis=1, keepdims=True)
        R = np.exp(R - log_norm)

        return R, cond_means, cond_covs
    
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

        R, cond_means, cond_covs = self.compute_responsibility(X_new)

        X_filled = X_new.copy()
        for i in range(N):
            miss_mask = missing_mask[i]
            X_filled[i][miss_mask] = np.sum(R[i] * cond_means[i])

        return X_filled
        
    
    def predict(self, X_new):
        R, _, _ = self.compute_responsibility(X_new)
        return np.argmax(R, axis=1)



