import numpy as np 
from sklearn.impute import SimpleImputer

global_imputer = SimpleImputer()

def mean_impute(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def median_impute(X):
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def mode_impute(X):
    imputer = SimpleImputer(strategy='mode')
    X_imputed = imputer.fit_transform(X)
    return X_imputed