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
            Initializes BMM using Gibbs Sampling

            Parameters:
                priorParameters (BMMPriorParameters):  prior parameters for the Bernoulli Mixture Model
        """
        super().__init__(priorParameters)

        self.a_0 = priorParameters.a_0
        self.b_0 = priorParameters.b_0
        self.model_type = "bernoulli"

    def likelihood(self,X,missing_mask,missing, mnar=False, eps=0):
        """
            computes log-likelihood for BMM with missing data

            parameters:
                X :  input data matrix (N x D)
                missing_mask : boolean array indicating missing data (N x D)
                missing (bool): whether there is missing data
                mnar (bool): whether to model MNAR missingness
                eps (float): small constant added to avoid numerical issues

            Rrturns:
                tuple:
                    - p (np.narray): responsibility matrix (N x K)
                    - loglik : loglikelihood of the data
                    - log_ps (np.ndarray): log probabilities for each data point and component
        """
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
        """
            Samples the parameters of the BMM 
        """
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
        """
            Samples the missing values in the data matrix 
        """
        N, D = self.X.shape
        X_sample = self.X_miss.copy()

        θ_indexed = self.θ[self.z]  # shape (n, D)

        sampled = np.random.binomial(1, θ_indexed)

        X_sample[self.missing_mask] = sampled[self.missing_mask]

        self.X = X_sample
    
    def fit(self, X, num_iters=6000, burn=2000, mnar=False, collapse=False):
        """
            Performs Gibbs sampling for the BMM

            Parameters:
                X (np.ndarray):  input data matrix (N x D)
                num_iters (int):  number of Gibbs sampling iterations
                burn (int):  number of burn-in iterations to discard
                mnar (bool): whether to model MNAR
                collapse (bool): hether to collapse the model (marginalize missing entries)

            Returns:
                np.ndarray: aligned component means from the posterior samples
        """
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
        """
        computes the posterior probability for the current parameters

        returns:
            float: posterior probability
        """
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
        """
            Ccmputes the responsibility matrix for the new data with or without missingness

            parameters:
                X_new (np.ndarray): input data matrix with missing values (
                mnar (bool): Whether to model MNAR

            Returns:
                np.ndarray:  responsibility matrix for the new data
        """
        N,D = X_new.shape
        missing_mask = np.isnan(X_new)
        missing = np.isnan(X_new).any()

        return self.likelihood(X_new,missing_mask,missing, mnar)[2]

    def log_likelihood(self, X_new, mnar=False):
        """
            computes  loglike for new data under current parameters

            Parameters:
                X_new (np.ndarray): new data points for which to compute the loglike
                mnar (bool): whether to model MNAR

            Returns:
                float: loglikelihood of new data
        """
        N,D = X_new.shape

        R = self.compute_responsibility(X_new, mnar)
        log_norm = logsumexp(R, axis=1, keepdims=True)
        loglike = np.sum(log_norm)/N

        return loglike
    
    def predict(self, X_new, mnar=False):
        """
            predicts cluster assignments for new data

            Parameters:
                X_new (np.ndarray): new data points for which to predict cluster assignments
                mnar (bool): Whether to model MNAR

            Returns:
                np.ndarray: predicted cluster assignments for new data points
        """
        R = self.compute_responsibility(X_new, mnar)
        zs = np.argmax(R, axis=1)
        return zs
    
    def impute(self, X_new, sample=None, eps=1e-14):
        """
            imputes the missing entries in the new data matrix using current model parameters

            Parameters
                X_new (np.ndarray): new data points with missing values (NaN).
                sample (dict, optional): specific Gibbs sample to use for imputation - Defaults to None
                eps (float): small constant to avoid numerical issues

            Returns
                np.ndarray: imputed data matrix 
        """
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
        θ = np.asarray(sample['θ'])       
        θ = np.clip(θ, eps, 1.0 - eps)

        X_imputed = X_new.copy()

        for i in range(N):
            miss = missing_mask[i]
            if not np.any(miss):
                continue 

            obs = ~miss
            if not np.any(obs):
                X_imputed[i] = π @ θ
                continue

            x_obs = X_new[i, obs]
            logw = np.log(π + eps)
            logw += (x_obs * np.log(θ[:, obs]) + (1.0 - x_obs) * np.log(1.0 - θ[:, obs])).sum(axis=1)

            # normalize safely
            m = np.max(logw)
            w = np.exp(logw - m)
            R = w / w.sum()

            X_imputed[i, miss] = R @ θ[:, miss]

        return X_imputed





