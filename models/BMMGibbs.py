from scipy.special import logsumexp
from scipy.special import softmax
from scipy.stats import dirichlet
from scipy.stats import beta
import numpy as np
import random 

from utils.ArbitraryImputer import mean_impute

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
        self.model_type = "bernoulli"

    def likelihood(self,X,missing_mask,missing, mnar=False, eps=0):
        N, D = X.shape
        self.K, _ = self.θ.shape

        if not missing:
            log_ps  = np.log(self.π) + (X @ np.log(self.θ).T + (1 - X) @ np.log(1 - self.θ).T) # log likelihood
        else:
            logθ = np.log(np.clip(self.θ,  eps, 1 - eps)) # (K, D)
            log1mθ = np.log(np.clip(1 - self.θ, eps, 1 - eps)) # (K, D)

            obs_mask = ~missing_mask # (N, D)  
            X_filled = np.nan_to_num(X, nan=0.0) 

            X_exp = X_filled[:, None, :] # (N, 1,D)
            obs_exp = obs_mask.astype(float)[:,None,:] # (N, 1, D)
            logθ_exp = logθ[None, :, :]  # (1,K, D)
            log1mθ_exp = log1mθ[None, :, :] # (1, K, D)

            log_px = np.sum(
                            obs_exp * (X_exp * logθ_exp + 
                            (1.0 - X_exp) * log1mθ_exp )
                    ,axis=2) # (N, K)
            
            if mnar:
                log_px += (obs_mask @ np.log(self.γ).T + (1 - obs_mask) @ np.log(1 - self.γ).T)

            log_ps = log_px + np.log(self.π)[None, :] # (N, K)

        log_norm = logsumexp(log_ps, axis=1, keepdims=True)
        loglik = np.sum(log_norm)/N
        p = np.exp(log_ps - log_norm)

        return p, loglik, log_ps

    
    def sample_θ(self) :
        zs_zerohot = np.eye(self.K)[self.z.astype(np.int64)]
         
        if self.missing: # If there is missing data 
            obs_mask = ~self.missing_mask
            obs_k = zs_zerohot.T @ obs_mask
            nk = obs_k
            X_observed = np.nan_to_num(self.X * obs_mask, nan=0)
        else: # if there is not missing data 
            nk = np.sum(zs_zerohot, axis=0)[:,None]
            X_observed = self.X

        nkd1 = zs_zerohot.T @ X_observed
        nkd0 = np.clip(nk - nkd1, 1e-8, None)

        self.θ = self.rng.beta(self.a_0 + nkd1, self.b_0 + nkd0)
    
    def sample_X_missing(self):
        N, D = self.X.shape
        X_sample = self.X_miss.copy()

        θ_indexed = self.θ[self.z]  # shape (n, D)

        sampled = np.random.binomial(1, θ_indexed)

        X_sample[self.missing_mask] = sampled[self.missing_mask]

        self.X = X_sample
    
    def fit(self, X, num_iters=6000, burn=2000, mnar=False, collapse=False):
        '''
            Performs Gibbs Sampling 
            By default returns mean of aligned samples (using Hungarian algorithm)
        '''
        N,D = X.shape 

        self.missing_mask = np.isnan(X)
        self.missing = np.any(self.missing_mask)

        if collapse:
            self.X = X.copy()
        else:
            self.X_miss = X.copy()
            self.X = mean_impute(X)
            self.missing=False

        self.z = np.random.randint(0,self.K,size=N)

        self.fitted = True
        self.samples = []

        for t in range(0,num_iters+burn):
            self.sample_π()
            self.sample_θ()
            if mnar : self.sample_γ()

            ps,loglike,R = self.likelihood(self.X, self.missing_mask, self.missing, mnar)
            
            self.sampleZ(ps)

            if not collapse:
                self.sample_X_missing()

            if t > burn:
                self.samples.append({
                    'π': self.π.copy(),
                    'z': self.z.copy(),
                    'θ': self.θ.copy(),
                    'loglike' : loglike,
                    'posterior' : self.compute_posterior()
                })
        self.relabel_all_samples() # Align samples using Hungarian

        self.aligned_means = self.get_aligned_param_means()

        self.map_params = self.get_map_params()

        return self.aligned_means

    def compute_posterior(self):
        N,D = self.X.shape

        obs_mask = ~self.missing_mask
        
        comp_prior = dirichlet.logpdf(self.π, self.α_0)
        cat_prior = np.sum(np.log(self.π[self.z]))

        param_prior = 0.0
        for k in range(self.K):
            param_prior += np.sum(beta.logpdf(self.θ[k], self.a_0[k], self.b_0[k]))

        log_likelihood = 0.0
        for n in range(N):
            k = self.z[n]
            x = self.X[n] if not self.missing else self.X[n][obs_mask[n]]
            θ = self.θ[k] if not self.missing else self.θ[k][obs_mask[n]]
            log_likelihood += np.sum(x * np.log(θ)) + np.sum((1 - x) * np.log(1 - θ))

        return comp_prior + cat_prior + param_prior + log_likelihood


    def compute_responsibility(self, X_new, mnar=False):
        ''' 
            Computes marginalized responsibility matrix for data with or without missing features 
            Used for computing log likelihood for holdout set
        '''
        N,D = X_new.shape
        missing_mask = np.isnan(X_new)
        missing = np.isnan(X_new).any()

        return self.likelihood(X_new,missing_mask,missing, mnar)[2]

    def log_likelihood(self, X_new, mnar=False):
        N,D = X_new.shape

        R = self.compute_responsibility(X_new, mnar)
        log_norm = logsumexp(R, axis=1, keepdims=True)
        loglike = np.sum(log_norm)/N

        return loglike
    
    def predict(self, X_new, mnar=False):
        R = self.compute_responsibility(X_new, mnar)
        zs = np.argmax(R, axis=1)
        return zs





