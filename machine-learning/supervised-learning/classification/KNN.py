import numpy as np
import sys
from collections import Counter

sys.path.append('../')
sys.path.append('../../..')
from global_utils import Model

class KNN(Model):
    '''
    K-Nearest Neighbors Classifier.
    
    Parameters:
        n_neighbors: int, default=5
            Number of neighbors to use.
        metric: str, default='euclidean'
            The distance metric to use. Options
            are 'euclidean', 'manhattan', 'chebyshev', and 'minkowski'.
        p: int, default=2
            The power parameter for the Minkowski metric.

    Attributes:
        X_train: array-like
            The training data.
        y_train: array-like
            The target values.
        n_neighbors: int
            Number of neighbors to use.
        metric: str
            The distance metric to use.
        p: int
            The power parameter for the Minkowski metric.

    Methods:
        fit(X, y)
            Fit the model to the training data.
        predict(X)
            Predict the class labels.
    '''
    def __init__(self, n_neighbors=5, metric='euclidean', p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            distances = self._calculate_distances(X[i])
            sorted_indices = np.argsort(distances)
            nearest_indices = sorted_indices[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            most_common = Counter(nearest_labels).most_common(1)[0][0]
            y_pred.append(most_common)

        return np.array(y_pred)
    
    def _calculate_distances(self, x):
        distances = []
        for i in range(self.X_train.shape[0]):
            distances.append(self._distance(x, self.X_train[i]))
        return np.array(distances)
    
    def _calculate_distances(self, x):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train - x), axis=1)
        elif self.metric == 'minkowski':
            return np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1 / self.p)
        else:
            raise ValueError('Invalid metric. Metric must be "euclidean", "manhattan", "chebyshev", or "minkowski".')

    def tune_k_plotly(self, X_val, y_val):
        from sklearn.metrics import accuracy_score
        import plotly.express as px

        k_values = range(1, 31)
        accuracies = []
        from tqdm.auto import tqdm
        for k in tqdm(k_values):
            self.n_neighbors = k
            y_pred = self.predict(X_val)
            accuracies.append(accuracy_score(y_val, y_pred))

        fig = px.line(
            x=list(k_values),
            y=accuracies,
            title="Optimal K Value for KNN",
            labels={"x": "Number of Neighbors (k)", "y": "Accuracy"}
        )
        fig.show()
        best_k = k_values[int(np.argmax(accuracies))]
        print("Best K:", best_k)
        return best_k


if __name__ == '__main__':
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
    # Initialize and Train Model
    model = KNN(n_neighbors=20, metric='euclidean', p=2)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    from sklearn.metrics import accuracy_score, classification_report
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Predictions:', y_pred[:5])
    print('True Labels:', y_test[:5])

    # Tune K
    best_k = model.tune_k_plotly(X_test, y_test)
    model.n_neighbors = best_k
    y_pred_best = model.predict(X_test)
    print("Accuracy with best k:", accuracy_score(y_test, y_pred_best))