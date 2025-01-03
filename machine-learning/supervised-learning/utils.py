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