import numpy as np

class GMMPriorParameters:
    def __init__(self, data, K):
        N,D = data.shape

        self.K = K
        self.α_0 = np.full(K, 1/K, dtype=np.float64)
        self.m_0 = np.full((K,D), 0, dtype=np.float64)
        self.S_0 = np.array([np.eye(D) for _ in range(K)])
        self.k_0 = np.full(K, 0.01, dtype=np.float64)
        self.ν_0 = np.full(K,(D + 1), dtype=np.float64)
        # Missing Mask Beta parameters
        self.o = np.full((K, D), 1) 
        self.h  = np.full((K, D), 1) 

class BMMPriorParameters:
    def __init__(self, data, K):
        N,D = data.shape

        self.K=K
        self.α_0 = np.full(K, 1/K, dtype=np.float64)
        # Data Beta parameters
        self.a_0 = np.full((K,D), 1)
        self.b_0 = np.full((K,D), 1)
        # Missing Mask Beta parameters
        self.o = np.full((K, D), 1) 
        self.h  = np.full((K, D), 1) 
