import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../..')
from global_utils import Model
from utils import WeightInitialization

class DecisionTreeRegressor(Model):
    '''
    Decision Tree Regressor model.
    Parameters:
        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf: int, default=1
            The minimum number of samples required to be at a leaf node.
        max_features: int, default=None
            The number of features to consider when looking for the best split.
        criterion: str, default='mse'
            The function to measure the quality of a split. Options are 'mse' and 'mae'.
        random_state: int, default=None
            The seed used by the random number generator.
    
    Attributes:
        tree: dict
            The learned decision tree.
    
    Methods:
        fit(X, y)
            Fit the model to the training data.
        predict(X)
            Predict the target values.
    
    Usage:
        model = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None, criterion='mse', random_state=None)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
    
    example output:
        Run ```python decisiontree_regressor.py``` locally to see the output.
        Mean Squared Error: 0.0
        R2 Score: 1.0
    '''
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, criterion='mse', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):
        # Convert y to a NumPy array to ensure integer-based indexing
        y = np.array(y)
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples == 1:
            return {'value': y[0]}
        if len(np.unique(y)) == 1:
            return {'value': y[0]}
        if self.max_depth is not None and depth == self.max_depth:
            return {'value': np.mean(y)}
        if n_samples <= self.min_samples_split:
            return {'value': np.mean(y)}
        if n_samples <= 2 * self.min_samples_leaf:
            return {'value': np.mean(y)}
        
        feature_idxs = np.random.choice(n_features, self.max_features, replace=False) if self.max_features else range(n_features)
        best_feature, best_threshold, best_value = self._best_split(X, y, feature_idxs)
        left_idxs = np.where(X[:, best_feature] <= best_threshold)[0]
        right_idxs = np.where(X[:, best_feature] > best_threshold)[0]
        left_tree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return {'feature': best_feature, 'threshold': best_threshold, 'value': best_value, 'left': left_tree, 'right': right_tree}
    
    def _best_split(self, X, y, feature_idxs):
        best_feature, best_threshold, best_value = None, None, np.inf
        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs = np.where(X[:, feature] <= threshold)[0]
                right_idxs = np.where(X[:, feature] > threshold)[0]
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue
                value = self._impurity(y, y[left_idxs], y[right_idxs])
                if value < best_value:
                    best_feature, best_threshold, best_value = feature, threshold, value
        return best_feature, best_threshold, best_value
    
    def _impurity(self, y, y_left, y_right):
        if self.criterion == 'mse':
            return np.mean((y - np.mean(y)) ** 2) - (len(y_left) / len(y)) * np.mean((y_left - np.mean(y_left)) ** 2) - (len(y_right) / len(y)) * np.mean((y_right - np.mean(y_right)) ** 2)
        elif self.criterion == 'mae':
            return np.mean(np.abs(y - np.median(y))) - (len(y_left) / len(y)) * np.mean(np.abs(y_left - np.median(y_left))) - (len(y_right) / len(y)) * np.mean(np.abs(y_right - np.median(y_right)))
                                                                                                                                                                 
    def _predict(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

    def print_tree_breadth_first(self):
        from collections import deque
        queue = deque([self.tree])
        level = 0
        while queue:
            level_size = len(queue)
            print(f"Level {level}:")
            for _ in range(level_size):
                node = queue.popleft()
                if 'value' in node:
                    print(f"  Value: {node['value']}")
                else:
                    print(f"  Feature: {node['feature']}, Threshold: {node['threshold']}")
                    queue.append(node['left'])
                    queue.append(node['right'])
            level += 1
        
if __name__ == '__main__':
    # Usage Example
    from sklearn.metrics import mean_squared_error, r2_score

    import sys
    sys.path.append('../..')
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

    # Convert y_train and y_test to NumPy arrays
    y_train = y_train.values
    y_test = y_test.values

    # Implement Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None, criterion='mse', random_state=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):,.2f}")
    print(model.print_tree_breadth_first())
