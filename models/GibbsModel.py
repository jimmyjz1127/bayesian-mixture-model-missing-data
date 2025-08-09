from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
import random 
import copy
from scipy.stats import mode
from scipy.special import logsumexp

class GibbsModel(ABC):
    def __init__(self, prior):
        self.fitted = False 
        self.rng = np.random.default_rng(42)
        self.prior = prior
        self.K = prior.K
        self.α_0 = prior.α_0
        self.o = prior.o
        self.h = prior.h

    @abstractmethod
    def fit(self, X, num_iters=6000, burn=2000, mnar=False, collapse=False):
        """
            To be implemented by subclass 
        """
        pass

    def sampleZ(self, p):      
        # Inverse sample from categorical distribution
        cdf = np.cumsum(p, axis=1) # compute CDF for each row (each categorical distribution)
        u   = self.rng.random(size=(p.shape[0], 1))
        self.z = (cdf > u).argmax(axis=1)  # return first index where cdf is greater than random u
    
    def sample_π(self):
        z_counts = np.bincount((self.z).astype(np.int64), minlength=self.K)
        self.π = self.rng.dirichlet(self.α_0 + z_counts)   

    def sample_γ(self):
        N,D = self.X.shape 
        obs_mask = (~self.missing_mask).astype(np.float64)

        zs_zerohot = np.eye(self.K)[self.z.astype(np.int64)]
    
        mkd1 = zs_zerohot.T @ obs_mask   # shape: (K, D)
        mkd0 = np.sum(zs_zerohot, axis=0)[:, None] - mkd1

        self.γ = self.rng.beta(self.o + mkd1, self.h + mkd0)


    def get_map_params(self):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.") 
        
        sample_map = max(self.samples, key=lambda p : p['posterior'])
        return sample_map
    
    def get_map_params2(self):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.") 
        
        sample_map = max(self.aligned_samples, key=lambda p : p['posterior'])
        return sample_map
    
    def get_summarizing_results(self, y):
        """ 
            Produces summarizing results : mean of ARI, loglikelihood, posterior likelihood
            along with standard deviation for each 
        """
        if not self.fitted:
            raise Exception("Model has not been fitted yet.") 
        
        ari = []
        ll = []
        pl = []

        for sample in self.samples:
            ll.append(sample['loglike'])
            pl.append(sample['posterior'])
            ari.append( adjusted_rand_score(y,sample['z']))

        N = len(self.samples)

        return {
            'avg_ari' : np.mean(ari),
            'std_ari' : np.std(ari),
            'avg_ll'  : np.mean(ll),
            "std_ll"  : np.std(ll),
            'avg_pl'  : np.mean(pl),
            'std_pl'  : np.std(pl)
        }
    
    
    def hungarian_permutation(self, z_ref, z_new, K):
        cost_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cost_matrix[i, j] = -np.sum((z_new == i) & (z_ref == j))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        perm = col_ind[np.argsort(row_ind)]
        return perm

    def relabel_sample(self, sample, perm):
        sample = copy.deepcopy(sample)
        
        sample['z'] = np.array([perm[label] for label in sample['z']])
        
        # Relabel all other items (params) if the first dimension matches K
        K = len(perm)
        for key, value in sample.items():
            if key == 'z':
                continue
            if isinstance(value, np.ndarray) and value.shape[0] == K:
                sample[key] = value[perm]
        
        return sample

    def relabel_all_samples(self):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.") 

        # Z = np.array([s['z'] for s in self.samples])
        # self.z_ref = mode(Z, axis=0).mode.squeeze()
        self.z_ref = self.get_map_params()['z']

        self.aligned_samples = []
        for t, sample in enumerate(self.samples):
            # permutation to align current sample's z to reference
            perm = self.hungarian_permutation(self.z_ref, sample['z'], self.K)
            aligned_sample = self.relabel_sample(sample, perm)
            self.aligned_samples.append(aligned_sample)

    def get_aligned_param_means(self):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.")
        
        Z = np.array([s['z'] for s in self.aligned_samples])
        best_z = mode(Z, axis=0).mode.squeeze()
        
        if self.model_type == "bernoulli":
            θ = np.mean([s['θ'] for s in self.aligned_samples], axis=0)
            π = np.mean([s['π'] for s in self.aligned_samples], axis=0)
            ll = np.mean([s['loglike'] for s in self.aligned_samples], axis=0)
            post = np.mean([s['posterior'] for s in self.aligned_samples], axis=0)
            return {
                "θ" : θ,
                'π' : π,
                'z' : best_z,
                'loglike' : ll,
                'posterior' : post
            }
        else:
            μ = np.mean([s['μ'] for s in self.aligned_samples], axis=0)
            Σ = np.mean([s['Σ'] for s in self.aligned_samples], axis=0)
            π = np.mean([s['π'] for s in self.aligned_samples], axis=0)
            x = np.mean([s['x'] for s in self.aligned_samples], axis=0)
            ll = np.mean([s['loglike'] for s in self.aligned_samples], axis=0)
            post = np.mean([s['posterior'] for s in self.aligned_samples], axis=0)
            return {
                'μ': μ,
                'Σ': Σ,
                'π' : π,
                'z' : best_z,
                'x' : x,
                'loglike' : ll,
                'posterior' : post
            }

    