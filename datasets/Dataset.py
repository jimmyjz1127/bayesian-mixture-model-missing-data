import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, X,y, missing_rate=0.1, seed=5099):
        self.X_full       = X
        self.y            = y
        self.seed         = seed

        rng = np.random.default_rng(self.seed)

    def apply_missingness(self, X, missing_rate, missing_type="uniform"):
        mask = np.ones_like(X, dtype=bool)

        if missing_type == "uniform":
            mask = self.rng.random(X.shape) > missing_rate

        X_missing = X.copy()
        X_missing[~mask] = np.nan
        return X_missing,self.y
    
    def get_missing_data(self):
        return self.X_missing

    def get_complete_data(self):
        return self.X_full
    
    def get_labels(self):
        return self.y