from sklearn.cluster import KMeans as SKLearnKMeans
import numpy as np

class CustomKMeans:
    def __init__(self, K, complete_case=False):
        """
        Parameters:
            K              : Number of clusters
            complete_case  : Whether to ignore rows with missing data (complete case analysis)
        """
        self.K = K
        self.complete_case = complete_case
        self.kmeans = SKLearnKMeans(n_clusters=K, random_state=5099, n_init="auto")
        self.rng = np.random.default_rng(5099)
        self.fitted = False
        self.random_assignments = False

    def fit(self, X):
        self.X = X
        self.missing_mask = np.isnan(X)
        complete_rows = ~np.any(self.missing_mask, axis=1)
        num_complete_rows = np.sum(complete_rows)
        self.fitted = True

        # if there are too few rows for complete case analysis
        if num_complete_rows < self.K:
            self.random_assignments = True
            return {
                "z":np.random.randint(0,self.K,size=len(X)),
                "loglike":0
            }

        if self.complete_case:
            X_fit = X[complete_rows]
        else:
            X_fit = X  # Assume imputed

        self.kmeans.fit(X_fit)

        z_pred = np.empty(len(X), dtype=int)

        if self.complete_case:
            z_pred[complete_rows] = self.kmeans.predict(X_fit)
            # Random assignment for rows with any missing values
            z_pred[~complete_rows] = self.rng.integers(0, self.K, size=np.sum(~complete_rows))
        else:
            z_pred[:] = self.kmeans.predict(X)

        self.z = z_pred

        return {
            "loglike": 0,
            "z": z_pred
        }

    def predict(self, X_new):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if self.random_assignments:
            return np.random.randint(0,self.K,size=len(X_new))
        
        missing_mask = np.isnan(X_new)
        complete_rows = ~np.any(missing_mask, axis=1)
        num_complete_rows = np.sum(complete_rows)

        if num_complete_rows < self.K:
            return np.random.randint(0,self.K,size=len(X_new))

        if self.complete_case:
            X_fit = X_new[complete_rows]
        else:
            X_fit = X_new

        z_pred = np.empty(len(X_new), dtype=int)

        if self.complete_case:
            z_pred[complete_rows] = self.kmeans.predict(X_fit)
            # Random assignment for rows with any missing values
            z_pred[~complete_rows] = self.rng.integers(0, self.K, size=np.sum(~complete_rows))
        else:
            z_pred[:] = self.kmeans.predict(X_new)

        return z_pred

    def log_likelihood(self, X):
        # return 0 to comply with shared API
        return 0
    
