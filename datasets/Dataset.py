import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, X,y, seed=5099):
        self.X_full       = X
        self.y            = y
        self.seed         = seed

        self.rng = np.random.default_rng(self.seed)

    def apply_missingness(self, missing_rate, missing_type="uniform"):
        X_missing = self.X_full.copy().astype(np.float64)
        N, D = self.X_full.shape
        missing_mask = np.random.rand(N, D) < missing_rate
        # Fix: ensure not all features are missing for any row
        all_missing = missing_mask.all(axis=1)
        while np.any(all_missing):
            missing_mask[all_missing] = np.random.rand(np.sum(all_missing), D) < missing_rate
            all_missing = missing_mask.all(axis=1)
        X_missing[missing_mask] = np.nan
        return X_missing,self.y

    def get_complete_data(self):
        return self.X_full
    
    def get_labels(self):
        return self.y