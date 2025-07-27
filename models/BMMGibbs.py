from scipy.special import logsumexp
from scipy.special import softmax
from scipy.stats import dirichlet
from scipy.stats import beta
import numpy as np
import random 

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from models.GibbsModel import GibbsModel
from models.PriorParameters import BMMPriorParameters

class BMMGibbs(GibbsModel):
    """
        Implementation of Bernoulli Mixture Model with missing data using Gibbs Sampling Inference 
    """
    def __init__(self, priorParameters : BMMPriorParameters):
        """
            Parameters 
        """
        super().__init__(priorParameters)

        self.a_0 = priorParameters.a_0
        self.b_0 = priorParameters.b_0

    def likelihood(self, X, θ, π, missing_mask, missing ):
        N,D = X.shape
        K,_ = θ.shape 
        log_ps = np.zeros((N,K))
        obs_mask = ~np.isnan(X)

        if not missing:
            logp  = np.log(π) + (X @ np.log(θ).T + (1 - X) @ np.log(1 - θ).T) # log likelihood
            logp -= logp.max(axis=1, keepdims=True) # reduce logits for numerical stability (invariance property)
            p     = np.exp(logp)
            p    /= p.sum(axis=1, keepdims=True) # normalize
            return p, np.sum(logp)/N, logp
    
        for i in range(N):           
            for k in range(K):
                x_obs = X[i][obs_mask[i]]
                θ_obs = θ[k][obs_mask[i]]
                log_ps[i,k] = np.log(π[k]) + np.sum(x_obs * np.log(θ_obs)) + np.sum((1 - x_obs) * np.log(1 - θ_obs))

        log_norm = logsumexp(log_ps, axis=1, keepdims=True)

        loglik = np.sum(log_norm)/N

        p = np.exp(log_ps - log_norm)

        return p, loglik, log_ps
    
    def sample_θ(self, X, zs, missing_mask, missing) :
        N,D=X.shape
        zs_zerohot = np.eye(self.K)[zs.astype(np.int64)]
         
        if missing: # If there is missing data 
            obs_mask = ~missing_mask
            obs_k = zs_zerohot.T @ obs_mask
            nk = obs_k
            X_observed = np.nan_to_num(X * obs_mask, nan=0)
        else: # if there is not missing data 
            nk = np.sum(zs_zerohot, axis=0)[:,None]
            X_observed = X

        nkd1 = zs_zerohot.T @ X_observed

        # nkd0 = nk - nkd1
        nkd0 = np.clip(nk - nkd1, 1e-8, None)

        return self.rng.beta(self.a_0 + nkd1, self.b_0 + nkd0)
    
    def sample_X_missing(self, z, θ, X, missing_mask):
        N, D = X.shape
        X_sample = X.copy()

        θ_indexed = θ[z]  # shape (n, D)

        sampled = np.random.binomial(1, θ_indexed)

        X_sample[missing_mask] = sampled[missing_mask]

        return X_sample
    
    def fit(self, X, num_iters=4000, burn=1000):
        N,D = X.shape 

        assert np.nanmin(X) >= 0 and np.nanmax(X) <= 1, "X must be binary"

        missing_mask = np.isnan(X)
        missing = np.any(missing_mask)

        z = np.random.randint(0,self.K,size=N)

        self.fitted = True
        self.samples = []

        for t in range(0,num_iters+burn):
            π = self.sample_π(z)

            θ = self.sample_θ(X, z, missing_mask, missing)

            ps,loglike,R = self.likelihood(X, θ,π, missing_mask, missing)
            
            z = self.sampleZ(ps).astype(np.int16)

            if t > burn:
                self.samples.append({
                    'π': π.copy(),
                    'z': z.copy(),
                    'θ': θ.copy(),
                    'loglike' : loglike,
                    'posterior' : self.compute_posterior(X, z, θ, π, missing_mask, missing)
                })

        return self.samples

    def compute_posterior(self, X, zs, θs, πs, missing_mask, missing):
        N,D = X.shape

        obs_mask = ~missing_mask
        
        comp_prior = dirichlet.logpdf(πs, self.α_0)
        cat_prior = np.sum(np.log(πs[zs]))

        param_prior = 0.0
        for k in range(self.K):
            param_prior += np.sum(beta.logpdf(θs[k], self.a_0[k], self.b_0[k]))

        log_likelihood = 0.0
        for n in range(N):
            k = zs[n]
            x = X[n] if not missing else X[n][obs_mask[n]]
            θ = θs[k] if not missing else θs[k][obs_mask[n]]
            log_likelihood += np.sum(x * np.log(θ)) + np.sum((1 - x) * np.log(1 - θ))

        return comp_prior + cat_prior + param_prior + log_likelihood


    def compute_responsibility(self, X, θs, πs):
        ''' 
            Computes marginalized responsibility matrix for data with or without missing features 
            Used for computing log likelihood for holdout set
        '''
        N,D = X.shape
        missing_mask = np.isnan(X)
        missing = np.isnan(X).any()

        return self.likelihood(X,θs,πs,missing_mask,missing)[2]

    def log_likelihood(self, X, θs, πs):
        N,D = X.shape

        R = self.compute_responsibility(X, θs, πs)
        log_norm = logsumexp(R, axis=1, keepdims=True)
        loglike = np.sum(log_norm)/N

        return loglike
    
    def predict(self, X, θs, πs):
        R = self.compute_responsibility(X, θs, πs)
        zs = np.argmax(R, axis=1)
        return zs





