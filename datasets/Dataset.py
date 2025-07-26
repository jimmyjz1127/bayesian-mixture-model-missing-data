import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, X,y, missing_rate=0.1, missing_type="uniform",seed=5099):
        self.X_full       = X
        self.y            = y
        self.missing_rate = missing_rate
        self.seed         = seed
        self.mask_type    = missing_type

        self._apply_missingness(X)

    def _apply_missingness(self, X):
        rng = np.random.default_rng(self.seed)
        mask = np.ones_like(X, dtype=bool)

        if self.mask_type == "uniform":
            mask = rng.random(X.shape) > self.missing_rate
        # elif self.mast_type == "half_upper":
        #     mask = np.arrange(X.shape[0])

        X_missing = X.copy()
        X_missing[~mask] = np.nan
        return X_missing, mask
    
    def get_data(self):
        return self.X_missing, self.mask

    def get_complete_data(self):
        return self.X_full
    
    def get_labels(self):
        return self.y