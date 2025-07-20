from .base_dataset import BaseDataset
from utils.preprocessing import standardize

class GaussianDataset(BaseDataset):
    def __init__(self, X, **kwargs):
        X_processed = standardize(X)
        super().__init__(X_processed, **kwargs)