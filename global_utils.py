from abc import ABC, abstractmethod

class Model(ABC):
    '''
    Abstract class for all models.

    Methods:
        fit(X, y)
            Fit the model to the training data.
        predict(X)
            Predict the target values
    '''
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
