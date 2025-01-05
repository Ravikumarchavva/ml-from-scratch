import numpy as np

class WeightInitialization:
    def __init__(self, method='normal'):
        self.method = method

    def initialize(self, n_features):
        if self.method == 'normal':
            return np.random.normal(size=n_features)
        elif self.method == 'random':
            return np.random.rand(n_features)
        elif self.method == 'zeros':
            return np.zeros(n_features)
        else:
            raise ValueError('Invalid method type. Use "normal", "random" or "zeros".')
        

import pandas as pd
import polars as pl

def check_purity(data, target_col):
    """
    Check if the data is pure.
    """
    if isinstance(target_col, str):
        unique_classes = set(data[target_col])
    elif isinstance(data, pl.DataFrame):
        unique_classes = set(data[:,target_col])
    elif isinstance(data, pd.DataFrame):
        unique_classes = set(data.iloc[:, target_col])
    if len(unique_classes) == 1:
        return True
    else:
        return False