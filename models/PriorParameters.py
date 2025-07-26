import numpy as np

class GMMPriorParameters:
    def __init__(self, data, K):
        N,D = data.shape

        self.K = K
        self.α_0 = np.full(K, 1, dtype=np.float64)
        self.m_0 = np.full((K,D), 0, dtype=np.float64)
        self.S_0 = np.array([1e-2 * np.eye(D) for _ in range(K)])
        self.k_0 = np.full(K, 0.01, dtype=np.float64)
        self.ν_0 = np.full(K,(D + 1), dtype=np.float64)