import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../..')

from utils import WeightInitialization, Model

class LogisticRegression(Model):
    '''
    Logistic Regression model using Gradient Descent.
    Parameters:
        learning_rate: float, default=0.01
            The learning rate to update weights and bias.
        n_iterations: int, default=1000
            The number of iterations to train the model.
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
        model = LogisticRegression(learning_rate=0.001, n_iterations=10000, initial_weights='normal')

        model.fit(X_train, y_train, verbose=2000)

        y_pred = model.predict(X_test)

    example output:
        Run ```python logistic_regression.py``` locally to see the output.
        Iteration 0: Loss = 2.280097080915255
        Iteration 2000: Loss = 0.4916559172929709
        Iteration 4000: Loss = 0.46210939165040854
        Iteration 6000: Loss = 0.4549949115209675
        Iteration 8000: Loss = 0.45105882731177477
        Accuracy: 0.8157129000969933
    '''
    def __init__(self, learning_rate=0.01, n_iterations=1000, initial_weights='normal'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
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
            # calculate the predicted values
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            # calculate the loss
            loss = self.binary_cross_entropy(y, y_pred)
            self.loss_history[i] = loss
            
            # calculate the gradients
            dw = np.dot(X.T, (y_pred - y)) / X.shape[0]
            db = np.sum(y_pred - y) / X.shape[0]
            
            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if verbose > 0 and i % verbose == 0:
                print(f"Iteration {i}: Loss = {loss}")

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def binary_cross_entropy(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
if __name__ == '__main__':
    # generate sklearn random data
    import pandas as pd
    from global_config import data_path

    # Load data
    train_data = pd.read_csv(data_path / 'ibm_churn' / 'balanced_train.csv')
    test_data = pd.read_csv(data_path / 'ibm_churn' / 'balanced_test.csv')

    # Drop non-numeric columns
    train_data = train_data.select_dtypes(include=[np.number])
    test_data = test_data.select_dtypes(include=[np.number])

    # Drop rows with missing values
    train_data.dropna(inplace=True, axis=0)
    test_data.dropna(inplace=True, axis=0)

    # Prepare data
    X_train, y_train = train_data.drop(['Churn'], axis=1), train_data['Churn']
    X_test, y_test = test_data.drop(['Churn'], axis=1), test_data['Churn']

    # Preporcess data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # train the model
    model = LogisticRegression(learning_rate=0.01, n_iterations=10000, initial_weights='normal')
    model.fit(X_train, y_train, verbose=2000)
    
    # predict the target values
    y_pred = model.predict(X_test)
    
    # calculate the accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")

    # plot loss
    import plotly.express as px
    fig = px.line(model.loss_history, title='loss_history')
    fig.show()
    
