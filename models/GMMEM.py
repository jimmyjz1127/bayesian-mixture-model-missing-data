from scipy.stats import invwishart 
from scipy.stats import multivariate_normal 
from scipy.special import logsumexp
import numpy as np
import random 


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

                    self.R[i,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(self.X[i,obs_mask],μ_o,Σ_oo,allow_singular=True)

        log_norm = logsumexp(self.R, axis=1, keepdims=True)
        self.R = np.exp(self.R - log_norm)
        self.z = np.argmax(self.R, axis=1)

        loglik = np.sum(log_norm) / N
        return loglik
    
    def m_step(self):
        N,D = self.X.shape
        K = self.K
        nk = self.R.sum(axis=0)

        self.π = nk / N

        if not self.missing:
            self.μ = (self.R.T @ self.X)/nk[:,None]
            diff = self.X[:, None, :] - self.μ[None, :, :]
            outer = diff[:, :, :, None] * diff[:, :, None, :]
            weighted_outer = self.R[:, :, None, None] * outer  # (N, K, D, D)
            self.Σ = weighted_outer.sum(axis=0) / nk[:, None, None]
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

                    m_i = μ_h + Σ_ho @ np.linalg.inv(Σ_oo) @ (self.X[i,obs_mask] - μ_o)
                    V_i = Σ_hh - Σ_ho @ np.linalg.inv(Σ_oo) @ Σ_oh

                    self.x_hat = self.X[i].copy()
                    self.x_hat[miss_mask] = m_i
                    new_μs[k] += self.R[i,k] * self.x_hat 

                    self.x_hats_outer = np.outer(self.x_hat, self.x_hat)
                    if np.any(miss_mask):
                        self.x_hats_outer[miss_mask][:,miss_mask] += V_i 

                    new_Σs[k] += self.R[i, k] * self.x_hats_outer

            new_μs /= nk[:, None]
            for k in range(K):
                new_Σs[k] /=  nk[k] 
                new_Σs[k] -= new_μs[k][:, None] @ new_μs[k][:, None].T
            
            self.μ = new_μs
            self.Σ = new_Σs

    def fit(self,X,max_iters=200,tol=1e-4):
        self.X = X 
        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)

        N,D = X.shape
        K = self.K

        self.μ = np.zeros((K,D)) + np.random.gamma(1.0, 0.1, size=(K, D))
        self.Σ = np.array([np.eye(D) for _ in range(K)])
        self.R = np.random.dirichlet(alpha=np.full(K, 1), size=N)

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
            'loglikes' :loglikes
        } 
    
    def posterior_predict(self, X_new, eps=1e-14):
        N,D = X_new.shape
        K = self.K

        missing_mask = np.isnan(X_new)

        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape != self.X.shape:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new

        cond_means = np.array([[None]*K for _ in range(N)])
        cond_covs = np.array([[None]*K for _ in range(N)])
        
        R = np.zeros((N,K))

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

                m_i = μ_h + Σ_ho @ np.linalg.inv(Σ_oo) @ (X_new[i,obs_mask] - μ_o)
                V_i = Σ_hh - Σ_ho @ np.linalg.inv(Σ_oo) @ Σ_oh
                
                cond_means[i,k] = m_i
                cond_covs[i,k] = V_i

                R[i,k] = np.log(self.π[k] + eps) + multivariate_normal.logpdf(X_new[i,obs_mask],μ_o,Σ_oo,allow_singular=True)

        log_norm = logsumexp(R, axis=1, keepdims=True)
        R = np.exp(R - log_norm)

        return R, cond_means, cond_covs
    
    def posterior_predict_impute(self, X_new, eps=1e-14):
        ''' 
            Imputes missing entries using expectation 
        '''

        N,D = X_new.shape
        K = self.K

        missing_mask = np.isnan(X_new)

        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape != self.X.shape:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new

        R, cond_means, cond_covs = self.posterior_predict(X_new)

        X_filled = X_new.copy()
        for i in range(N):
            miss_mask = missing_mask[i]
            X_filled[i][miss_mask] = np.sum(R[i] * cond_means[i])

        return X_filled
        
    
    def posterior_predict_sample(self,X_new,num_samples=1,eps=1e-14):
        ''' 
            Imputes using hierarchical sampling
        '''

        N,D = X_new.shape
        K = self.K

        missing_mask = np.isnan(X_new)

        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape != self.X.shape:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new

        R, cond_means, cond_covs = self.posterior_predict(X_new)

        imputed_Xs = []

        for n in range(num_samples):
            cdf = np.cumsum(R, axis=1) # compute CDF for each row (each categorical distribution)
            u   = self.rng.random(size=(R.shape[0], 1))
            z = (cdf > u).argmax(axis=1)

            X_filled = X_new.copy()
            for i in range(N):
                k = z[i]
                X_filled[i,missing_mask[i]] = self.rng.multivariate_normal(mean=cond_means[i,k], cov=cond_covs[i,k])

            imputed_Xs.append(X_filled.copy())

        return np.mean(imputed_Xs, axis=0)



