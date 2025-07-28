import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mean_squared_error


def multi_restart(model, X, y, N=10, max_iters=200, tol=1e-3):
    samples = []
    scores = []

    for n in range(N):
        sample = model.fit(X, max_iters, tol)
        samples.append(sample)
        scores.append(adjusted_rand_score(y,sample['z']))

    max_idx = np.argmax(scores)

    return samples[max_idx]


def rmse(A,B):
    return np.sqrt(mean_squared_error(A,B))

# def evalaute_increasing_missingness(dataset, EMModel, VBEMModel, GibbsModel)