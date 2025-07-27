import numpy as np
import pandas as pd


from scipy.special import logsumexp
from scipy.special import softmax
from scipy.special import betaln
from scipy.special import beta

from scipy.stats import multivariate_normal 
from scipy.stats import invwishart 

from sklearn import metrics
from scipy.special import psi
from scipy.special import digamma 

from scipy.special import gammaln, psi
from numpy.linalg import slogdet, inv

from models.PriorParameters import GMMPriorParameters 
from models.VBEMModel import VBEMModel

class GMMVBEM(VBEMModel):
    
    def __init__(self, priorParameters : GMMPriorParameters):
        """
            Parameters 
        """
        super().__init__(priorParameters)

        self.m_0 =  priorParameters.m_0
        self.S_0 = priorParameters.S_0
        self.k_0 = priorParameters.k_0
        self.ν_0 = priorParameters.ν_0

    def update_Θ(self, X, R, x_hats, x_hats_outer):
        N, D = X.shape

        n_k = R.sum(axis=0)  # (K,)

        # Computing x_bar 
        num   = (R.T[:, :, None] * x_hats).sum(axis=1)
        x_bar = num / n_k[:, None] # (K,D)

        x_bar_outer = np.einsum('kd,ke->kde', x_bar, x_bar)      # (K,D,D)
        centered    = x_hats_outer - x_bar_outer[:, None, :, :]  # (K,N,D,D)

        # Σ_n r_{nk} E[(x‑x̄)(x‑x̄)ᵀ]
        S_k = np.einsum('nk,knij->kij', R, centered) 

        Σ_map = (self.S_0 + S_k) / (n_k[:, None, None] + self.ν_0[:, None, None] + D + 1)
        Σ_inv = np.array([np.linalg.inv(Σ_map[k]) for k in range(self.K)])

        Λ_k   = self.k_0[:, None, None] * Σ_inv          # (K,D,D)
        V_k   = np.array([np.linalg.inv(n_k[k] * Σ_inv[k] + Λ_k[k]) for k in range(self.K)])

        rhs   = (Λ_k @ self.m_0[..., None]).squeeze(-1) + (n_k[:, None] * (Σ_inv @ x_bar[..., None]).squeeze(-1))
        m_k   = np.einsum('kij,kj->ki', V_k, rhs)

        return V_k, m_k, Σ_map,n_k


    def logprob(self,X,ms,Σs,nk,missing_mask):
        N,D = X.shape

        conditional_means = np.array([[None]*self.K for _ in range(N)])
        conditional_covs = np.array([[None]*self.K for _ in range(N)])
        logprobs = np.zeros((N,self.K))

        for i in range(N):
            miss_mask = missing_mask[i]
            obs_mask = ~miss_mask
            x_o = X[i][obs_mask]

            for k in range(self.K):
                m = ms[k]
                Σ = Σs[k]

                m_h = m[miss_mask]
                m_o = m[obs_mask]
                Σ_oh = Σ[obs_mask][:, miss_mask]
                Σ_ho = Σ[miss_mask][:, obs_mask]
                Σ_oo = Σ[obs_mask][:, obs_mask]
                Σ_hh = Σ[miss_mask][:, miss_mask]

                inv_Σ_oo = np.linalg.inv(Σ_oo)

                cond_mean = m_h + Σ_ho @ inv_Σ_oo @ (x_o - m_o)
                cond_cov = Σ_hh - Σ_ho @ inv_Σ_oo @ Σ_oh
                conditional_means[i, k] = cond_mean
                conditional_covs[i, k] = cond_cov

                try:
                    D_o = np.sum(obs_mask)
                    diff = x_o - m_o 
                    logprobs[i, k] = -0.5 * (np.linalg.slogdet(Σ_oo)[1] + (diff @ inv_Σ_oo @ diff) - D_o/(nk[k] + self.k_0[k]))
                except np.linalg.LinAlgError:
                    logprobs[i, k] = -np.inf  # if singular

        return conditional_means, conditional_covs, logprobs
    
    def compute_sufficient_stats(self, X,cond_mean, cond_cov, missing_mask):
        N,D = X.shape
        K = self.K

        x_hats = np.zeros((K,N,D))
        x_hats_outer = np.zeros((K,N,D,D))

        for i in range(N):
            miss_mask = missing_mask[i]
            obs_mask = ~miss_mask
            for k in range(K):
                x_hat = X[i].copy()
                x_hat[miss_mask] = cond_mean[i,k]

                outer = np.outer(x_hat, x_hat)
                if np.any(miss_mask):
                    outer[np.ix_(miss_mask, miss_mask)] += cond_cov[i,k] 

                x_hats[k,i,:] = x_hat.copy()
                x_hats_outer[k,i,:,:] = outer.copy()

        return x_hats, x_hats_outer  
    
    ''' 
        ############## ELBO ##############
    '''
    
    def kl_phi(self, m, V, Σ):
        K, D = m.shape

        kl_total = 0.0
        for k in range(K):

            Σ_inv = np.linalg.inv(Σ[k])

            tr_term = self.k_0[k] * np.trace(Σ_inv @ V[k])

            diff = m[k] - self.m_0[k]
            maha_term = self.k_0[k] * diff.T @ Σ_inv @ diff

            logdet_prior = np.linalg.slogdet(Σ[k]/self.k_0[k])[1]
            logdet_q     = np.linalg.slogdet(V[k])[1]

            Σ_niw = invwishart.logpdf(Σ[k], self.ν_0[k], self.S_0[k])

            # E_{q()} [p()] 
            exp_logjoint = D * np.log(2 * np.pi) + logdet_prior + tr_term + maha_term

            # E_{q()} [q()]
            entropy = D * np.log(2 * np.pi) + logdet_q + D

            kl_k = (-0.5 * exp_logjoint) + Σ_niw + (0.5 * entropy)

            kl_total += kl_k

        return kl_total
    
    def expected_data_loglikelihood(self, X,R,m,Σ, x_hats, x_hats_outer, nks, missing_mask):
        N, D = X.shape
        K = R.shape[1]

        total = 0.0

        for n in range(N):
            x_n = X[n]

            for k in range(K):
                nk = nks[k]
                r_nk = R[n, k]
                if r_nk == 0:
                    continue

                Σ_inv = np.linalg.inv(Σ[k])

                d_pi = D * np.log(2 * np.pi)
                logdet = np.linalg.slogdet(Σ[k])[1]
                trace_term = np.linalg.trace(Σ_inv @ x_hats_outer[k,n])

                term = (
                    d_pi + logdet + trace_term
                    - 2 * x_hats[k,n].T @ Σ_inv @ m[k]
                    + m[k].T @ Σ_inv @ m[k]
                    + D/(nk + self.k_0[k])
                )

                total += -0.5 * r_nk * term

        return total
    
    def entropy_q_XH(self, R, V_ho, missing_mask):
        N, K = R.shape
        log_2pi = np.log(2 * np.pi)

        total = 0.0
        for n in range(N):
            D_H_n = int(missing_mask[n].sum())

            for k in range(K):
                r_nk = R[n, k]
                if r_nk == 0 or D_H_n == 0:
                    continue

                V_nk = V_ho[n, k]

                # Compute log determinant
                if np.ndim(V_nk) == 0:  # scalar
                    logdet = np.log(V_nk)
                elif np.ndim(V_nk) == 1:  # diagonal
                    logdet = np.sum(np.log(V_nk))
                else:  # full matrix
                    logdet = np.linalg.slogdet(V_nk)[1]

                entropy = D_H_n * log_2pi + logdet + D_H_n
                total += -0.5 * r_nk * entropy 

        return total
    
    def compute_elbo(self, R, X, Σ, m, V, V_ho,
         α, x_hats, x_hats_outer,nks,missing_mask):
        return (
            self.kl_pi(α)    
            + self.kl_z(R, α)          
            + self.kl_phi(m, V, Σ)
            + self.expected_data_loglikelihood(X,R,m, Σ, x_hats, x_hats_outer, nks, missing_mask)
            - self.entropy_q_XH(R, V_ho, missing_mask)
        )
    
    ''' 
        -------------END ELBO------------
    '''
     
    
    def fit(self, X, max_iters=200, tol=1e-4, mode=0):
        """
            Parameters:
            X         : input data (N, D)
            max_iters : maximum iterations to perform
            tol       : convergence criteria for elbo
            mode      : 1 for MLE estimate of pi, 0 for variational update
        """

        N,D = X.shape

        missing_mask = np.isnan(X)
        missing = np.any(missing_mask)

        self.fitted = True
        loglikes = []
        elbos = []

        R = np.random.dirichlet(alpha=self.α_0, size=N)  # (N, K)

        Σs = self.S_0.copy()  # Covariances of q(μ_k)
        ms = self.m_0.copy()  # Means of q(μ_k)
        α = self.α_0.copy()
        ν = self.ν_0.copy()
        πs = self.rng.dirichlet(alpha=α)
        Vs = np.zeros((self.K,D,D))
        x_hats = np.zeros((self.K, N, D))
        x_hats_outer = np.zeros((self.K, N, D, D))

        # fill with conditional mean imputation 
        for n in range(N):
            obs_mask = ~missing_mask[n]
            miss_mask = missing_mask[n]
            for k in range(self.K):
                x_hat = X[n].copy()
                # Impute missing dimensions with column-wise mean or 0
                x_hat[miss_mask] = np.nanmean(X[:, miss_mask], axis=0)
                x_hats[k, n, :] = x_hat
                x_hats_outer[k, n, :, :] = np.outer(x_hat, x_hat)


        for t in range(max_iters):
            V,m,Σ,nk = self.update_Θ(X, R, x_hats, x_hats_outer)
            πs = np.sum(R,axis=0)/N
            α = self.update_π(R)

            m_ho, V_ho, logprobs = self.logprob(X,m,Σ,nk,missing_mask)

            x_hats, x_hats_outer = self.compute_sufficient_stats(X,m_ho,V_ho, missing_mask)

            R, loglike = self.update_z(X,logprobs,α,πs,mode)

            elbo = self.compute_elbo(R, X, Σ, m, V, V_ho, α, x_hats, x_hats_outer, nk, missing_mask)

            elbos.append(elbo)
            loglikes.append(loglike)

            if t > 1 and np.abs(elbos[t] - elbos[t-1]) < tol:
                break

        z = np.argmax(R, axis=1)
        self.result = {
            'R'            : R,             # Responsibility matrix  (N,K)
            'z'            : z,             # Cluster assignments    (N)
            'α'            : α,             # dirichlet prior        (K)
            'Vk'           : V,             # prior covariance on μ  (K,D,D)
            'm'            : m,             # prior mean on μ        (K,D)
            'Σ'            : Σ,             # MAP estimate of Σ      (K,D,D)
            'm_ho'         : m_ho,          # conditional means      (N,K,D)
            'V_ho'         : V_ho,          # conditional covariance (N,K,D,D)
            'x_hats'       : x_hats,        # sufficient states x    (K,N,D)
            'x_hats_outer' : x_hats_outer,  # sufficient states xx^T (K,N,D,D)
            'loglikes'     : loglikes,      # log likelihoods
            'elbos'        : elbos          # elbos 
        }
        return self.result

    
    
     