# bayesian_optimization.py
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt import gp_minimize
import numpy as np

def objective(params, X, y):
    gp = GaussianProcessRegressor()
    gp.fit(X, y)
    y_pred = gp.predict(np.array([params]))
    return y_pred[0]

def run_BO(seed, csv_data, feature_indexes, output_index, n_calls, n_random_starts, acq_func, bounds):
    
    X = csv_data.iloc[:, feature_indexes[0]:feature_indexes[1] + 1]
    y = csv_data.iloc[:, output_index]
    
    print(f"Bounds: {bounds}")  # Debugging line
    print(f"Shape of X: {X.shape}")  # Debugging line

    def objective_to_minimize(params):
        return objective(params, X, y)

    res = gp_minimize(objective_to_minimize, 
                      bounds, 
                      acq_func=acq_func, 
                      n_calls=n_calls, 
                      n_random_starts=n_random_starts, 
                      random_state=seed)
    
    return res.x, res.fun
