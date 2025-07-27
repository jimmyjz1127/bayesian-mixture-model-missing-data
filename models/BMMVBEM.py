import numpy as np
from scipy.special import logsumexp
from sklearn import metrics
from scipy.special import psi, gammaln
from scipy.special import digamma,expit

from models.PriorParameters import BMMPriorParameters 
from models.VBEMModel import VBEMModel

class BMMVBEM(VBEMModel):
    def __init__(self, priorParameters : BMMPriorParameters):
        """
            Parameters 
        """
        super().__init__(priorParameters)

        self.a_0 = priorParameters.a_0
        self.b_0 = priorParameters.b_0

    def update_Θ(self, X,R,a,b, missing_mask, missing):
        N,D = X.shape

        obs_mask = ~missing_mask

        a_new = np.zeros((self.K,D))
        b_new = np.zeros((self.K,D))

        exp_x = expit(digamma(a) - digamma(b))

        for k in range(self.K):

            if missing : 
                x = (exp_x[k] * missing_mask) + np.nan_to_num(X * obs_mask, nan=0)
            else:
                x = X

            a_new[k,:] = self.a_0[k,:] + (R[:,k].T @ x)
            b_new[k,:] = self.b_0[k,:] + (R[:,k].T @ (1 - x))

        return a_new,b_new
    
    def logprob(self, X, a, b):
        """
            Computes (marginalized) data log probability of observed data
        """
        psi_ab = digamma(a + b)
        elog_mu   = digamma(a) - psi_ab      
        elog_1mu  = digamma(b) - psi_ab     

        obs = ~np.isnan(X)
        X_obs   = np.where(obs, X, 0.0)       
        Xc_obs  = np.where(obs, 1.0 - X, 0.0) 

        hat_x = expit(digamma(a) - digamma(b))

        # (N,D) @ (D,K) -> (N,K)
        return X_obs @ elog_mu.T + Xc_obs @ elog_1mu.T, hat_x
    
    '''
        ############## ELBO ##############
    '''
    def log_B_beta(self, a, b):
        return gammaln(a) + gammaln(b) - gammaln(a + b)

    def kl_mu(self, a,b):
        Elog_mu  = psi(a) - psi(a + b)
        Elog_1mu = psi(b) - psi(a + b)
        return (np.sum(self.log_B_beta(a, b) - self.log_B_beta(self.a_0, self.b_0)) +
                np.sum((self.a_0 - a) * Elog_mu) +
                np.sum((self.b_0 - b) * Elog_1mu))
    

    def kl_XH(self, miss, R, tau, a, b):
        # expected logs under q(μ)
        elog_mu  = psi(a) - psi(a + b)        # (K,D)
        elog_1mu = psi(b) - psi(a + b)

        eps = 1e-12
        tau_safe = np.clip(tau, eps, 1 - eps)

        kl_elem = (
            tau_safe * (np.log(tau_safe) - elog_mu[None,:,:]) +
            (1 - tau_safe) * (np.log(1 - tau_safe) - elog_1mu[None,:,:])
        )                                         # (N,K,D)

        kl_total = np.sum(R[:,:,None] * kl_elem * miss[:,None,:])
        return kl_total
    
    def elbo_X_obs(self, X, R, a, b, missing_mask):
        obs = ~missing_mask
        X_obs  = np.where(obs, X, 0.0)
        Xc_obs = np.where(obs, 1.0 - X, 0.0)

        Elog_mu  = psi(a) - psi(a + b)    # (K,D)
        Elog_1mu = psi(b) - psi(a + b)    # (K,D)

        ll = X_obs @ Elog_mu.T + Xc_obs @ Elog_1mu.T  # (N,K)
        return np.sum(R * ll)
    
    def compute_elbo(self,X,R,a,b,α,hat_x, missing_mask):
        N,K = R.shape

        tau = np.where(missing_mask[:, None, :],     # (N,1,D)
                    hat_x[None, :, :],       # (1,K,D)
                    X[:, None, :])    

        elbo = (
            self.kl_pi(α) + 
            self.kl_mu(a,b) + 
            self.kl_z(R,α) + 
            self.elbo_X_obs(X, R, a, b, missing_mask) -
            self.kl_XH(missing_mask,R,tau,a,b)
        )

        return elbo
    
    '''
        ----------- ELBO ------------
    '''

    def fit(self, X, mode=0, max_iters=200, tol=1e-3):
        '''
            Parameters 
                X       : input data matrix (N x D)
                K       : number of components (K)
                mode : 0 update pi as RV, 1 to update MLE style
        '''

        N,D = X.shape
        K= self.K

        self.fitted = True

        missing_mask = np.isnan(X)
        missing = np.any(missing_mask)

        a = self.a_0.copy() + np.random.gamma(1.0, 0.1, size=(K, D))
        b = self.b_0.copy() + np.random.gamma(1.0, 0.1, size=(K, D))
        α = self.α_0.copy()
        πs = np.random.dirichlet(alpha=α)

        loglikes = []
        elbos = []

        for t in range(max_iters):
            logprob, x_hat = self.logprob(X,a,b)

            R,loglike =self.update_z(X,logprob,α,πs,mode=mode)

            α = self.update_π(R)
            a,b = self.update_Θ(X,R,a,b,missing_mask, missing)

            loglikes.append(loglike)

            elbo = self.compute_elbo(X,R,a,b,α,x_hat, missing_mask)

            elbos.append(elbo)

            if t > 1 and np.abs(elbos[t] - elbos[t-1]) < tol:
                break

        z = np.argmax(R, axis=1)

        self.result = {
            'R'            : R,             # Responsibility matrix  (N,K)
            'z'            : z,             # Cluster assignments    (N)
            'α'            : α,             # dirichlet prior        (K)
            'a'            : a,             # Beta a parameter       (K)
            'b'            : b,             # Beta b parameter       (K)
            'x_hat'        : x_hat,         # sufficent stats x      (K,D)
            'loglikes'     : loglikes,      # log likelihoods
            'elbos'        : elbos          # elbos 
        }

        return self.result



