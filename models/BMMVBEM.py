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

    def update_Θ(self):
        N,D = self.X.shape

        obs_mask = ~self.missing_mask

        a_new = np.zeros((self.K,D))
        b_new = np.zeros((self.K,D))

        exp_x = expit(digamma(self.a) - digamma(self.b))

        for k in range(self.K):
            if self.missing : 
                x = (exp_x[k] * self.missing_mask) + np.nan_to_num(self.X * obs_mask, nan=0)
            else:
                x = self.X

            a_new[k,:] = self.a_0[k,:] + (self.R[:,k].T @ x)
            b_new[k,:] = self.b_0[k,:] + (self.R[:,k].T @ (1 - x))

        self.a = a_new
        self.b = b_new
    
    def logprob(self, X, missing_mask):
        """
            Computes (marginalized) data log probability of observed data
        """
        psi_ab = digamma(self.a + self.b)
        elog_mu   = digamma(self.a) - psi_ab      
        elog_1mu  = digamma(self.b) - psi_ab     

        obs = ~missing_mask
        X_obs   = np.where(obs, X, 0.0)       
        Xc_obs  = np.where(obs, 1.0 - X, 0.0) 

        x_hat = expit(digamma(self.a) - digamma(self.b))

        # (N,D) @ (D,K) -> (N,K)
        return X_obs @ elog_mu.T + Xc_obs @ elog_1mu.T, x_hat
    
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
    
    def compute_elbo(self):
        N,K = self.R.shape

        tau = np.where(self.missing_mask[:, None, :],     # (N,1,D)
                    self.x_hat[None, :, :],       # (1,K,D)
                    self.X[:, None, :])    

        elbo = (
            self.kl_pi(self.α) + 
            self.kl_mu(self.a,self.b) + 
            self.kl_z(self.R,self.α) + 
            self.elbo_X_obs(self.X, self.R, self.a, self.b, self.missing_mask) -
            self.kl_XH(self.missing_mask,self.R,tau,self.a,self.b)
        )

        return elbo
    
    '''
        ----------- ELBO ------------
    '''
    def fit(self, X, mode=0, max_iters=200, tol=1e-4):
        '''
            Parameters 
                X       : input data matrix (N x D)
                K       : number of components (K)
                mode : 0 update pi as RV, 1 to update MLE style
        '''
        self.X = X
        self.mode = mode
        self.fitted = True
        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)

        N,D = self.X.shape
        K= self.K

        self.a = self.a_0.copy() 
        self.b = self.b_0.copy() 
        z = np.random.randint(0,K,size=N)
        self.R = np.zeros((N, K))
        self.R[np.arange(N), z] = 1

        loglikes = []
        elbos = []

        for t in range(max_iters):
            self.update_π()
            self.update_Θ()
            logprob, self.x_hat = self.logprob(self.X, self.missing_mask)
            self.R, loglike =self.update_z(logprob)
            elbo = self.compute_elbo()

            loglikes.append(loglike)
            elbos.append(elbo)

            if t > 1 and np.abs(elbos[t] - elbos[t-1]) < tol:
                break

        self.z = np.argmax(self.R, axis=1)

        self.result = {
            'R'            : self.R,             # Responsibility matrix  (N,K)
            'z'            : self.z,             # Cluster assignments    (N)
            'α'            : self.α,             # dirichlet prior        (K)
            'a'            : self.a,             # Beta a parameter       (K)
            'b'            : self.b,             # Beta b parameter       (K)
            'x_hat'        : self.x_hat,         # sufficent stats x      (K,D)
            'loglike'     : loglikes,      # log likelihoods
            'elbo'        : elbos          # elbos 
        }

        return self.result

    
    def predict(self, X_new):
        missing_mask = np.isnan(X_new)
        logprob,_ = self.logprob(X_new,missing_mask)
        R,_ = self.update_z(logprob)

        return np.argmax(R, axis=1)
    
    def posterior_predict(self, X_new, eps=1e-14):
        N,D = X_new.shape
        missing_mask = np.isnan(X_new)

        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape[1] != self.X.shape[1]:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new
        
        logprob,_ = self.logprob(X_new, missing_mask)
        R,_ = self.update_z(logprob)

        exp_θ =  self.a / (self.a + self.b + eps)

        X_filled = X_new.copy()
        for i in range(N):
            for d in range(D):
                if missing_mask[i, d]:

                    X_filled[i, d] = np.sum(R[i] * exp_θ[:, d])

        return X_filled



