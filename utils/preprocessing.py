import numpy as np

def standardize(X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)