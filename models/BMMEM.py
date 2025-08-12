import numpy as np
# from abc import ABC, abstractmethod
import random
from scipy.special import logsumexp

class BMMEM:
    def __init__(self, K, complete_case=False):
        """
            Initializes the BMMEM model for BMM using the EM algorithm.

            Parameters:
                K (int): number of mixture components
                complete_case (bool): If True, incomplete datapoints are ignored (complete case analysis)
        """

        self.rng = np.random.default_rng(5099)
        self.K = K
        self.fitted = False
        self.complete_case = complete_case

    def e_step(self,eps=1e-14):
        """
            performs the E-step of the algorithm, which computes the responsibility matrix 
            based on the current estimates params

            Parameters:
                eps (float): A small constant to avoid numerical issues in logarithmic calculations

            Returns:
                float: The log-likelihood of the data given the current parameters
        """

        N,D = self.X.shape
        K = self.K
        self.R = np.zeros((N,K))

        if not self.missing:
            self.R = np.log(self.π + eps) + (self.X @ np.log(self.θ + eps).T + (1 - self.X) @ np.log(1 - self.θ + eps).T)
        else:
            for i in range(N):
                mask = ~self.missing_mask[i]
            
                if not np.any(mask) or (self.complete_case and np.any(self.missing_mask[i])):
                    self.R[i,:] = np.log(self.π + eps)
                    continue

                for k in range(K):
                    x_obs = self.X[i][mask]
                    θ_obs = self.θ[k][mask]

                    self.R[i,k] = (
                        np.log(self.π[k] + eps) + 
                        np.sum(x_obs * np.log(θ_obs + eps)) + 
                        np.sum((1 - x_obs) * np.log(1 - θ_obs + eps))
                    )

        log_norm = logsumexp(self.R, axis=1, keepdims=True)
        self.R = np.exp(self.R - log_norm)

        loglik = np.sum(log_norm) / N
        # if self.complete_case:
        #     valid_rows = ~np.any(self.missing_mask, axis=1)
        #     loglik = np.sum(log_norm[valid_rows]) / np.sum(valid_rows)
        # else:
        #     loglik = np.sum(log_norm) / N

        return loglik
    

    def m_step(self, eps=1e-10):
        """
            Performs the M-step of the EM algorithm, which updates the model parameters
            based on the current responsibility matrix

            Parameters:
                eps (float): A small constant to prevent numerical issues
        """
        N,D = self.X.shape
        if self.complete_case:
            # Filter to only complete rows
            complete_rows = ~np.any(self.missing_mask, axis=1)
            x = self.X[complete_rows]
            R = self.R[complete_rows]
        else:
            x = self.X.copy()
            R = self.R.copy()
            if self.missing:
                x[self.missing_mask] = (self.R @ self.θ)[self.missing_mask]

        nk = R.sum(axis=0)
        nk_safe = np.maximum(nk, eps)

        self.θ = (R.T @ x) / nk_safe[:, None]
        self.π = nk_safe / nk_safe.sum()

    
    def fit(self, X, max_iters=200, tol=1e-4):
        """
            Fits BMMEM model using the EM algorithm. Iterates between the E-step and M-step 
            until convergence or the maximum number of iterations is reached.

            Parameters:
                X (np.ndarray):  input data matrix of shape (N, D) 
                max_iters (int):  maximum number of iterations to run EM
                tol (float): tolerance for convergence based on the log-likelihood change between iterations.

            Returns:
                dict: A dictionary containing the learned parameters and log-likelihood values:
                    - 'z': cluster assignments for each data point (N)
                    - 'π': mixture component weights (K)
                    - 'θ': component parameters (K, D)
                    - 'loglike': List of log-likelihood values over iterations
        """
        N,D = X.shape

        self.X = X

        K = self.K

        self.missing_mask = np.isnan(self.X)
        self.missing = np.any(self.missing_mask)
        
        z = np.random.choice(K, size=N)
        self.R = np.zeros((N, K))
        self.R[np.arange(N), z] = 1
        self.θ = np.random.uniform(0.1,0.9,size=(K,D))

        loglikes = []

        for _ in range(0,max_iters):
            self.m_step()
            ll = self.e_step()

            self.z = np.argmax(self.R, axis=1)
            loglikes.append(ll)

            if len(loglikes) > 1 and np.abs(loglikes[-1] - loglikes[-2]) < tol : break

        self.fitted = True
        return {
            'z' : self.z,
            'π' : self.π,
            'θ' : self.θ,
            'loglike' : loglikes
        }
    
    def compute_responsibility(self, X_new, eps=1e-14):
        """
            computes the responsibility matrix for a new set of data points 

            Parameters:
                X_new (np.ndarray): The input data matrix with missing values (NaN) to compute the responsibilities for
                eps (float): A small constant added to avoid numerical issues
            Returns:
                tuple: 
                    - The responsibility matrix of shape (N, K)
                    - The log-likelihood of the data under the current model parameters
        """
        N, D = X_new.shape

        missing_mask = np.isnan(X_new)
        missing = np.any(missing_mask)

        R = np.zeros((N, self.K))

        if not missing : 
            R = np.log(self.π + eps) + (X_new @ np.log(self.θ + eps).T + (1 - X_new) @ np.log(1 - self.θ + eps).T)
        else:
            for i in range(N):
                mask = ~missing_mask[i]
                x_obs = X_new[i][mask]

                if not np.any(mask)or (self.complete_case and np.any(~self.missing_mask[i])):
                    R[i,:] = np.log(self.π + eps)
                    continue
                
                for k in range(self.K):
                    θ_obs = self.θ[k][mask]
                    
                    logp = np.log(self.π[k] + eps)
                    logp += np.sum(x_obs * np.log(θ_obs + eps))
                    logp += np.sum((1 - x_obs) * np.log(1 -θ_obs + eps))
                    R[i, k] = logp

        log_norm = logsumexp(R, axis=1, keepdims=True)
        R = np.exp(R - log_norm)

        loglike = np.sum(log_norm)/N

        return R,loglike
    
    def impute(self, X_new, eps=1e-14):
        """
            Imputes missing values in the new data matrix based on the current model parameters and responsibility matrix.

            Parameters:
                X_new (np.ndarray): The input data matrix with missing values (NaN) to be imputed.
                eps (float): A small constant to prevent numerical issues. Defaults to 1e-14.

            Returns:
                np.ndarray: The imputed data matrix with missing values replaced by the model's imputed values.
        """
        N,D = X_new.shape
        missing_mask = np.isnan(X_new)

        if not self.fitted: 
            raise Exception("Model has not been fitted yet.")
        if X_new.shape[1] != self.X.shape[1]:
            raise Exception("Dimensions do not match fit.")
        if not np.any(missing_mask):
            return X_new

        R,_ = self.compute_responsibility(X_new)

        X_filled = X_new.copy()
        for i in range(N):
            for d in range(D):
                if missing_mask[i, d]:
                    X_filled[i, d] = np.sum(R[i] * self.θ[:, d])

        return X_filled
    

    def log_likelihood(self, X_new):
        """
            computes the log-likelihood of the new data points under the current parameters

            Parameters:
                X_new (np.ndarray): new data points for which the log-likelihood is computed

            Returns:
                float: log-likelihood of the new data under the current model
        """
        return self.compute_responsibility(X_new)[1]
    
    def predict(self, X_new):
        """
            Predicts the cluster assignments for new data points based on the learned parameters

            Parameters:
                X_new (np.ndarray): new data points to predict cluster assignments for

            return:
                np.ndarray: predicted cluster assignments for the new data points
        """
        R,_ = self.compute_responsibility(X_new)
        return np.argmax(R, axis=1)


