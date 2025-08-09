from scipy.special import logsumexp
from scipy.special import softmax
from scipy.stats import invwishart
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from utils.ArbitraryImputer import mean_impute

import numpy as np
import random 

from models.PriorParameters import GMMPriorParameters 
from models.GibbsModel import GibbsModel

class GMMGibbs(GibbsModel):
    """
        Implementation of Gaussian Mixture Model with missing data using Gibbs Sampling Inference 
    """

    def __init__(self, priorParameters : GMMPriorParameters):
        """
            Parameters 
        """
        super().__init__(priorParameters)

        self.m_0 =  priorParameters.m_0
        self.S_0 = priorParameters.S_0
        self.k_0 = priorParameters.k_0
        self.ν_0 = priorParameters.ν_0
        self.model_type = "gaussian"

    def likelihood(self, X, mnar=False, collapse=False):
        N,D = X.shape
        K = self.π.shape[0]
        
        logp = np.zeros((N,K))
        π = np.clip(self.π, 1e-14, 1.0)
        
        for k in range(K):
            if not self.missing or not collapse:
                logp[:,k] = np.log(π[k]) + multivariate_normal.logpdf(X, mean=self.μ[k],cov=self.Σ[k], allow_singular=True)
            else:
                μ = self.μ[k]
                Σ = self.Σ[k]
                for i in range(N):
                    miss_mask = self.missing_mask[i]
                    obs_mask = ~miss_mask

                    x_o = X[i][obs_mask]
                    if x_o.size == 0:
                        logp[i,k] = np.log(π[k])  # Just prior — no data likelihood
                        continue
                    μ_o  = μ[obs_mask]
                    Σ_oo = Σ[np.ix_(obs_mask, obs_mask)]
                    Σ_oo = 0.5 * (Σ_oo + Σ_oo.T) + (1e-6 * np.eye(Σ_oo.shape[0])) # regularization

                    logp[i,k] = np.log(π[k]) + multivariate_normal.logpdf(x_o, mean=μ_o, cov=Σ_oo, allow_singular=True)

        if mnar and self.missing:
            obs_mask = (~self.missing_mask).astype(float)
            logp += (obs_mask @ np.log(self.γ).T + (1 - obs_mask) @ np.log(1 - self.γ).T)

        log_norm = logsumexp(logp,axis=1,keepdims=True)

        loglik = np.sum(log_norm)/N

        p = np.exp(logp - log_norm)

        return p,loglik
    
    def sampleX(self):
        N, D = self.X.shape
        X_sample = self.X.copy()

        for i in range(N):
            k = self.z[i].astype(int)
            μ = self.μ[k]
            Σ = self.Σ[k]

            miss_mask = self.missing_mask[i]
            if not np.any(miss_mask):
                continue
            obs_mask = ~miss_mask
            obs_idx  = np.flatnonzero(obs_mask)
            miss_idx = np.flatnonzero(miss_mask)

            if not np.any(obs_mask):
                X_sample[i] = np.random.multivariate_normal(μ, Σ)
                continue

            μ_o  = μ[obs_idx]
            μ_h  = μ[miss_idx]
            Σ_oo = Σ[np.ix_(obs_idx,  obs_idx)]
            Σ_ho = Σ[np.ix_(miss_idx, obs_idx)]
            Σ_oh = Σ[np.ix_(obs_idx,  miss_idx)]
            Σ_hh = Σ[np.ix_(miss_idx, miss_idx)]

            Σ_oo = Σ_oo + 1e-6 * np.eye(Σ_oo.shape[0]) # jitter for stability
            try:
                v = np.linalg.solve(Σ_oo, (self.X[i, obs_idx] - μ_o))
            except np.linalg.LinAlgError:
                v = np.linalg.pinv(Σ_oo) @ (self.X[i, obs_idx] - μ_o)

            m_i = μ_h + Σ_ho @ v
            V_i = Σ_hh - Σ_ho @ (np.linalg.pinv(Σ_oo)) @ Σ_oh  

            # symmetry + jitter to ensure PSD
            V_i = 0.5 * (V_i + V_i.T) + 1e-8 * np.eye(V_i.shape[0])
            X_sample[i, miss_idx] = self.rng.multivariate_normal(m_i, V_i)

        return X_sample
    
    def sample_NIW(self, X):
        N,D = X.shape

        zs_hot = np.eye(self.K)[self.z.astype(int)]
        counts = np.bincount(self.z.astype(int), minlength=self.K)[:, None] # component cardinality 
        nonzero = counts.ravel() > 0 
        x_bar = np.zeros((self.K, D))
        np.divide(zs_hot.T @ X, counts, out=x_bar, where=nonzero[:, None])

        self.Σ = np.zeros((self.K,D,D))
        self.μ = np.zeros((self.K,D))

        for k in range(self.K):
            indices = (self.z == k)
            n_k = np.sum(indices) 
            k_n = self.k_0[k] + n_k

            X_k = X[indices]
            S = (X_k - x_bar[k]).T @ (X_k - x_bar[k])
            diff = (x_bar[k] - self.m_0[k]).reshape(-1,1)
            S_n = self.S_0[k] + S + ((self.k_0[k] * n_k)/(k_n)) * (diff @ diff.T)
            self.Σ[k] = invwishart.rvs(df=self.ν_0[k] + n_k, scale=S_n)
            self.Σ[k] = 0.5*(self.Σ[k] + self.Σ[k].T)

            μ_n = (self.k_0[k] * self.m_0[k] + n_k * x_bar[k])/k_n
            self.μ[k] = np.random.multivariate_normal(mean=μ_n, cov=self.Σ[k]/k_n)


    def fit(self, X, num_iters=6000, burn=2000, mnar=False, collapse=False):
        """ 
            Main Gibbs Sampling Loop
        """
        self.X =X 
        N,D = self.X.shape
        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)

        self.fitted = True
        self.samples = []

        self.π = self.rng.dirichlet(self.α_0)
        self.z = np.random.randint(0,self.K,size=N)
        x = self.X.copy() if not self.missing else mean_impute(self.X)
        self.sample_NIW(x)

        for t in range(0, num_iters + burn):
            self.sample_π()

            if (self.missing):
                x = self.sampleX()

            self.sample_NIW(x)
            if mnar : self.sample_γ()
            p, loglike = self.likelihood(x, mnar, collapse)
            self.sampleZ(p)

            if t > burn:
                self.samples.append({
                    'π': self.π.copy(),
                    'z': self.z.copy(),
                    'μ': self.μ.copy(),
                    'Σ': self.Σ.copy(),
                    'x': x.copy(),
                    'loglike': loglike,
                    'posterior': self.compute_posterior(x)
                })
        self.relabel_all_samples()

        self.aligned_means = self.get_aligned_param_means()
        self.map_params = self.get_map_params()

        return self.map_params
    
    def compute_posterior(self, X):
        N,D = X.shape

        comp_prior = dirichlet.logpdf(self.π, self.α_0)
        cat_prior = np.sum(np.log(self.π[self.z]))

        param_prior = 0.0
        for k in range(self.K):
            μ = self.μ[k]
            Σ = self.Σ[k]

            mean_prior   = multivariate_normal.logpdf(μ, mean=self.m_0[k], cov=(1/self.k_0[k])*Σ)
            cov_prior    = invwishart.logpdf(Σ, df=self.ν_0[k], scale=self.S_0[k])
            param_prior += mean_prior + cov_prior

        log_likelihood = 0.0
        for n in range(N):
            k = self.z[n]
            μ = self.μ[k]
            Σ = self.Σ[k]
            log_likelihood += multivariate_normal.logpdf(X[n], mean=μ, cov=Σ)

        posterior_prob = comp_prior + cat_prior + param_prior + log_likelihood
        return posterior_prob
    
    def compute_responsibility(self, X):
        ''' 
            Computes marginalized responsibility matrix for data with or without missing features 
            Used for computing log likelihood for holdout set
        '''
        N,D = X.shape
        missing_mask = np.isnan(X)
        missing = np.isnan(X).any()

        logprobs = np.zeros((N,self.K))

        for k in range(self.K):
            μ = self.μ[k]
            Σ = self.Σ[k]
            π = self.π[k]

            if (not missing) : 
                logprobs[:,k] = np.log(π) + multivariate_normal.logpdf(X, mean=μ, cov=Σ)
                continue

            for i in range(N):
                miss_mask = missing_mask[i]
                obs_mask = ~miss_mask
                if not np.any(obs_mask):
                    logprobs[i,k] = np.log(π)
                    continue

                x_o = X[i][obs_mask]
                μ_o  = μ[obs_mask]
                Σ_oo = Σ[obs_mask][:, obs_mask]

                logprobs[i,k] = np.log(π) + multivariate_normal.logpdf(x_o, mean=μ_o, cov=Σ_oo)

        return logprobs
    
    def log_likelihood(self, X):
        """
        
            X  : input data (N, D)
            μs : component means (K, D)
            Σs : component covariances (K, D, D)
            πs : component weights (,K)
        """
        N,D = X.shape

        R = self.compute_responsibility(X)
        log_norm = logsumexp(R, axis=1, keepdims=True)
        loglike = np.sum(log_norm)/N

        return loglike
    
    def predict(self, X, sample=None):
        """
            For making clustering predictions on a holdout set (test set)
        """ 
        if sample is None:
            sample = self.aligned_means

        R = self.compute_responsibility(X)
        zs = np.argmax(R, axis=1)
        return zs
    

    def impute(self, X_new, sample=None, eps=1e-14):
        N, D = X_new.shape
        K = self.K
        missing_mask = np.isnan(X_new)

        if not self.fitted:
            raise Exception("Model has not been fitted yet.")
        if X_new.shape[1] != self.X.shape[1]:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new
        if sample is None:
            sample = self.get_aligned_param_means()

        π = np.asarray(sample['π'])
        μ = np.asarray(sample['μ'])    
        Σ = np.asarray(sample['Σ'])     
        X_imputed = X_new.copy()

        for i in range(N):
            mask = missing_mask[i]
            obs_mask = ~mask
            x_obs = X_new[i, obs_mask]

            if not np.any(mask): continue
            if not np.any(obs_mask):
                X_imputed[i, :] = (π[:, None] * μ).sum(axis=0)
                continue            

            μ = sample['μ']
            Σ = sample['Σ']
            R = np.zeros((K))
            for k in range(K):
                μ_k = μ[k]
                Σ_k = Σ[k]
                μ_o = μ_k[obs_mask]
                Σ_oo = Σ_k[np.ix_(obs_mask, obs_mask)]
                x_obs = X_new[i, obs_mask]

                try:
                    logpdf = multivariate_normal.logpdf(x_obs, μ_o, Σ_oo)
                except:
                    logpdf = -np.inf  

                R[k] = np.log(π[k] + eps) + logpdf
            R = np.exp(R - logsumexp(R))

            x_impute = np.zeros(np.sum(mask))
            for k in range(K):
                μ_k = μ[k]
                Σ_k = Σ[k]
                μ_o = μ_k[obs_mask]
                μ_m = μ_k[mask]
                Σ_oo = Σ_k[np.ix_(obs_mask, obs_mask)]
                Σ_mo = Σ_k[np.ix_(mask, obs_mask)]

                try:
                    v = np.linalg.solve(Σ_oo, (x_obs - μ_o))
                except np.linalg.LinAlgError:
                    v = np.linalg.pinv(Σ_oo) @ (x_obs - μ_o)

                μ_cond = μ_m + Σ_mo @ v
                x_impute += R[k] * μ_cond

            X_imputed[i, mask] = x_impute
        return X_imputed
    







