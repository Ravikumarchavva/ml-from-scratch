import numpy as np
import sys
sys.path.append('../../..')
sys.path.append('../')
from utils import WeightInitialization, Model

class RidgeRegression(Model):
    '''
    Ridge Regression model using Gradient Descent.
    Parameters:
        learning_rate: float, default=0.01
            The learning rate to update weights and bias.
        n_iterations: int, default=1000
            The number of iterations to train the model.
        l2_penalty: float, default=0.01
            The L2 regularization penalty.
        initial_weights: str, default='normal'
            The method to initialize weights. Options are 'normal', 'random' and 'zeros'.

    Attributes:
        weights: array, shape (n_features,)
            The learned weights of the model.
        bias: float
            The learned bias of the model.
        loss_history: array, shape (n_iterations,)
            The loss history of the model during training.

    Methods:
        fit(X, y, verbose=0)
            Fit the model to the training data.
        predict(X)
            Predict the target values

    Usage:
        model = RidgeRegression(learning_rate=0.001, n_iterations=10000, l2_penalty=0.01, initial_weights='normal')

        model.fit(X_train, y_train, verbose=2000)

        y_pred = model.predict(X_test)

    example output:
        Run ```python ridge_regression.py``` locally to see the output.
        Iteration 0: Loss = 253,411,693.81
        Iteration 2000: Loss = 10,215,214.35
        Iteration 4000: Loss = 6,461,558.45
        Iteration 6000: Loss = 6,270,702.63
        Iteration 8000: Loss = 6,212,377.44
        Mean Squared Error: 6131574.338396628
        R2 Score: 0.8723105815527793
    '''
    def __init__(self, learning_rate=0.01, n_iterations=1000, l2_penalty=0.01, initial_weights='normal'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.initial_weights = WeightInitialization(initial_weights)

    def fit(self, X, y, verbose=0):
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
                
        # initialize weights and bias
        self.weights = self.initial_weights.initialize(X.shape[1])
        self.bias = 0

        # store loss history
        self.loss_history = np.zeros(self.n_iterations)

        for i in range(self.n_iterations):
            # calculate predictions
            y_pred = self.predict(X)

            # calculate loss
            loss = np.mean((y - y_pred) ** 2) + self.l2_penalty * np.sum(self.weights ** 2)
            self.loss_history[i] = loss

            # calculate gradients
            dw = -2 * np.dot(X.T, (y - y_pred)) + 2 * self.l2_penalty * self.weights
            db = -2 * np.sum(y - y_pred)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose > 0 and i % verbose == 0:
                print(f"Iteration {i}: Loss = {loss:,.2f}")

        if verbose:
            print(f"Iteration {self.n_iterations}: Loss = {loss:,.2f}")

    def predict(self, X):
        if not hasattr(self, 'weights'):
            raise ValueError("Model is not fitted yet. Call `fit` first.")
        return np.dot(X, self.weights) + self.bias

if __name__ == '__main__':
    # Usage Example
    from sklearn.metrics import mean_squared_error, r2_score
    from global_config import data_path

    train_data = data_path / 'car_price' / 'train_featureEngineered.csv'
    test_data = data_path / 'car_price' / 'test_featureEngineered.csv'

    # Load data
    import pandas as pd
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)

    # Drop non-numeric columns
    train_data = train_data.select_dtypes(include=[np.number])
    test_data = test_data.select_dtypes(include=[np.number])

    # Drop rows with missing values
    train_data.dropna(inplace=True, axis=0)
    test_data.dropna(inplace=True, axis=0)

    # Prepare data
    X_train, y_train = train_data.drop(['price', 'car_ID'], axis=1), train_data['price']
    X_test, y_test = test_data.drop(['price', 'car_ID'], axis=1), test_data['price']

    # Implement Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model with a reduced learning rate
    model = RidgeRegression(learning_rate=0.001, n_iterations=10000, l2_penalty=0.01, initial_weights='normal')
    model.fit(X_train, y_train, verbose=2000)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("Test Mean Squared Error:", mean_squared_error(y_test, y_pred).__format__(",.2f"))
    print("Test R2 Score:", r2_score(y_test, y_pred).__format__(",.2f"))

    # Plot loss history
    import plotly.express as px
    fig = px.line(y=model.loss_history, title='Loss History')
    fig.show()
