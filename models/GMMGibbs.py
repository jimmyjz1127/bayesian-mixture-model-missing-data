from scipy.special import logsumexp
from scipy.special import softmax
from scipy.stats import invwishart
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

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

    def likelihood(self, X):
        N,D = X.shape
        K = self.π.shape[0]
        
        logp = np.zeros((N,K))

        for k in range(K):
            logp[:,k] = np.log(self.π[k]) + multivariate_normal.logpdf(X, mean=self.μ[k],cov=self.Σ[k])

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
            obs_mask = ~miss_mask

            μ_h = μ[miss_mask]
            μ_o = μ[obs_mask]
            Σ_oh = Σ[obs_mask][:, miss_mask]
            Σ_ho = Σ[miss_mask][:, obs_mask]
            Σ_oo = Σ[obs_mask][:, obs_mask]
            Σ_hh = Σ[miss_mask][:, miss_mask]

            m_i = μ_h + Σ_ho @ np.linalg.inv(Σ_oo) @ (self.X[i,obs_mask] - μ_o)
            V_i = Σ_hh - Σ_ho @ np.linalg.inv(Σ_oo) @ Σ_oh

            # V_i += 1e-6 * np.eye(V_i.shape[0])

            X_sample[i,miss_mask] = np.random.multivariate_normal(m_i,V_i)

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

            μ_n = (self.k_0[k] * self.m_0[k] + n_k * x_bar[k])/k_n
            self.μ[k] = np.random.multivariate_normal(mean=μ_n, cov=self.Σ[k]/k_n)

    
    # @staticmethod
    def mean_impute(self, X, missing_mask):
        X_0 = X.copy()
        means = np.nanmean(np.where(missing_mask, np.nan, X), axis=0)
        X_0[missing_mask] = np.take(means, np.where(missing_mask)[1])
        return X_0

    def fit(self, X, num_iters=4000, burn=1000):
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
        x = X.copy() if not self.missing else self.mean_impute(self.X,self.missing_mask)
        self.sample_NIW(x)


        for t in range(0, num_iters + burn):
            self.sample_π()

            if (self.missing):
                x = self.sampleX()

            self.sample_NIW(x)

            p, loglike = self.likelihood(x)
            
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

        return self.samples
    
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

        R = self.compute_responsibility(X, self.μ, self.Σ, self.π)
        log_norm = logsumexp(R, axis=1, keepdims=True)
        loglike = np.sum(log_norm)/N

        return loglike
    
    def predict(self, X):
        """
            For making clustering predictions on a holdout set (test set)
        """ 

        R = self.compute_responsibility(X, self.μ, self.Σ, self.π)
        zs = np.argmax(R, axis=1)
        return zs
    

        







