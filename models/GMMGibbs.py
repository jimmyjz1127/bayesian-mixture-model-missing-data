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

class GMMGibbs:
    """
        Implementation of Gaussian Mixture Model with missing data using Gibbs Sampling Inference 
    """

    def __init__(self, priorParameters : GMMPriorParameters):
        """
            Parameters 
        """

        self.K   = priorParameters.K
        self.α_0 = priorParameters.α_0
        self.m_0 =  priorParameters.m_0
        self.S_0 = priorParameters.S_0
        self.k_0 = priorParameters.k_0
        self.ν_0 = priorParameters.ν_0

        self.rng = np.random.default_rng(5099)

        self.fitted = False

    def likelihood(self, X, μ, Σ, π):
        N,D = X.shape
        K = π.shape[0]
        
        logp = np.zeros((N,K))

        for k in range(K):
            logp[:,k] = np.log(π[k]) + multivariate_normal.logpdf(X, mean=μ[k],cov=Σ[k])

        log_norm = logsumexp(logp,axis=1,keepdims=True)

        p = np.exp(logp - log_norm)

        loglik = np.sum(log_norm)/N

        return p,loglik
    
    def sampleZ(self, p):
        ''' 
            Samples cluster assignments z for n datapoints 

            @param (p) : softmax categorical probabilities over clusters (N, K)
        '''

        # Inverse sample from categorical distribution
        cdf = np.cumsum(p, axis=1) # compute CDF for each row (each categorical distribution)
        u   = self.rng.random(size=(p.shape[0], 1))
        return (cdf > u).argmax(axis=1)  # return first index where cdf is greater than random u   
    
    def sampleX(self, X, zs, μs, Σs, missing_mask):
        N, D = X.shape
        X_sample = X.copy()

        for i in range(N):
            k = zs[i].astype(int)
            μ = μs[k]
            Σ = Σs[k]

            miss_mask = missing_mask[i]
            obs_mask = ~miss_mask

            μ_h = μ[miss_mask]
            μ_o = μ[obs_mask]
            Σ_oh = Σ[obs_mask][:, miss_mask]
            Σ_ho = Σ[miss_mask][:, obs_mask]
            Σ_oo = Σ[obs_mask][:, obs_mask]
            Σ_hh = Σ[miss_mask][:, miss_mask]

            m_i = μ_h + Σ_ho @ np.linalg.inv(Σ_oo) @ (X[i,obs_mask] - μ_o)
            V_i = Σ_hh - Σ_ho @ np.linalg.inv(Σ_oo) @ Σ_oh

            X_sample[i,miss_mask] = np.random.multivariate_normal(m_i,V_i)

        return X_sample
    
    def sample_NIW(self, X, zs):
        N,D = X.shape

        zs_hot = np.eye(self.K)[zs.astype(int)]
        counts = np.bincount(zs.astype(int), minlength=self.K)[:, None] # component cardinality 
        nonzero = counts.ravel() > 0 
        x_bar = np.zeros((self.K, D))
        np.divide(zs_hot.T @ X, counts, out=x_bar, where=nonzero[:, None])

        Σs = np.zeros((self.K,D,D))
        μs = np.zeros((self.K,D))

        for k in range(self.K):
            indices = (zs == k)
            n_k = np.sum(indices)
            k_n = self.k_0[k] + n_k

            X_k = X[indices]
            S = (X_k - x_bar[k]).T @ (X_k - x_bar[k])
            diff = (x_bar[k] - self.m_0[k]).reshape(-1,1)
            S_n = self.S_0[k] + S + ((self.k_0[k] * n_k)/(k_n)) * (diff @ diff.T)
            Σs[k] = invwishart.rvs(df=self.ν_0[k] + n_k, scale=S_n)


            μ_n = (self.k_0[k] * self.m_0[k] + n_k * x_bar[k])/k_n
            μs[k] = np.random.multivariate_normal(mean=μ_n, cov=Σs[k]/k_n)
            
        return Σs,μs
    
    def sample_π(self, z):
        ''' 
            Samples mixing weights from Dirichlet distribution parameterized by pseudocounts of components
            
            @param (zs)  : cluster assignments (n)
            @param (α_0) : Dirichlet prior list (K)
            @param (K)   : the number of components
        '''

        z_counts = np.bincount(z.astype(np.int64), minlength=self.K)
        return self.rng.dirichlet(self.α_0 + z_counts)
    
    def mean_impute(X,missing_mask):
        X_0 = X.copy()
        means = np.nanmean(np.where(missing_mask, np.nan, X), axis=0)
        X_0[missing_mask] = np.take(means, np.where(missing_mask)[1])
        return X_0

    def fit(self, X, num_iters=4000, burn=1000):
        N,D = X.shape
        missing_mask = np.isnan(X)
        missing = np.isnan(missing_mask).any()

        self.fitted = True
        self.samples = []

        π = self.rng.dirichlet(self.α_0)
        z = np.random.randint(0,self.K,size=N)
        x = X if not missing else self.mean_impute(X,missing_mask)
        Σ,μ = self.sample_NIW(x,z)

        for t in range(0, num_iters + burn):
            π = self.sample_π(z)

            if (missing):
                x = self.sampleX(X,z,μ,Σ,missing_mask)

            Σ,μ = self.sample_NIW(x,z)

            p, loglike = self.likelihood(X,μ,Σ,π)
            
            z = self.sampleZ(p)

            if t > burn:
                self.samples.append({
                    'π': π.copy(),
                    'z': z.copy(),
                    'μ': μ.copy(),
                    'Σ': Σ.copy(),
                    'x': x.copy(),
                    'loglike': loglike,
                    'posterior': self.compute_posterior(x, z, μ, Σ, π)
                })

        return self.samples
    
    def compute_posterior(self, X, zs, μs, Σs, πs):
        N,D = X.shape
        K,_ = μs.shape

        comp_prior = dirichlet.logpdf(πs, self.α_0)
        cat_prior = np.sum(np.log(πs[zs]))

        param_prior = 0.0
        for k in range(K):
            μ = μs[k]
            Σ = Σs[k]

            mean_prior   = multivariate_normal.logpdf(μ, mean=self.m_0[k], cov=(1/self.k_0[k])*Σ)
            cov_prior    = invwishart.logpdf(Σ, df=self.ν_0[k], scale=self.S_0[k])

            param_prior += mean_prior + cov_prior

        log_likelihood = 0.0
        for n in range(N):
            k = zs[n]
            μ = μs[k]
            Σ = Σs[k]
            log_likelihood += multivariate_normal.logpdf(X[n], mean=μ, cov=Σ)

        posterior_prob = comp_prior + cat_prior + param_prior + log_likelihood

        return posterior_prob
    
    def compute_responsibility(self, X, μs, Σs, πs):
        ''' 
            Computes marginalized responsibility matrix for data with or without missing features 
            Used for computing log likelihood for holdout set
        '''
        N,D = X.shape
        missing_mask = np.isnan(X)
        missing = np.isnan(X).any()

        logprobs = np.zeros((N,self.K))

        for k in range(K):
            μ = μs[k]
            Σ = Σs[k]
            π = πs[k]

            if (not missing) : 
                logprobs[:,k] = np.log(π) + multivariate_normal.logpdf(X, mean=μ, cov=Σ)
                continue

            for i in range(N):
                miss_mask = missing_mask[i]
                obs_mask = ~miss_mask

                x_o = X[i][obs_mask]
                μ_o  = μ[obs_mask]
                Σ_oo = Σ[obs_mask][:, obs_mask]

                logprobs[i,k] = np.log(π) + multivariate_normal(x_o, mean=μ_o, cov=Σ_oo)

        return logprobs
    
    def loglikelihood(self, X, μs, Σs, πs):
        N,D = X.shape

        R = self.compute_responsibility(X, μs, Σs, πs)
        log_norm = logsumexp(R, axis=1, keepdims=True)
        loglike = np.sum(log_norm)/N

        return loglike
    
    def predict(self, X, μs, Σs, πs):
        """
            For making clustering predictions on a holdout set (test set)
        """ 

        R = self.compute_responsibility(X, μs, Σs, πs)
        zs = np.argmax(R, axis=1)
        return zs
    
    def get_map_params(self):
        '''
            Returns the best Gibb Sample based on MAP 
        '''
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

        







