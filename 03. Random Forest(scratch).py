# Random Forest implementation from scratch withou using any inbuilt library

import random
from collections import Counter

# Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return [self._predict_sample(self.tree, sample) for sample in X]

    def _gini_impurity(self, y):
        classes = set(y)
        impurity = 1.0
        for cls in classes:
            p = y.count(cls) / len(y)
            impurity -= p ** 2
        return impurity

    def _split(self, X, y, feature_idx, threshold):
        X_left, y_left, X_right, y_right = [], [], [], []
        for i, sample in enumerate(X):
            if sample[feature_idx] <= threshold:
                X_left.append(sample)
                y_left.append(y[i])
            else:
                X_right.append(sample)
                y_right.append(y[i])
        return X_left, y_left, X_right, y_right

    def _best_split(self, X, y):
        best_feature, best_threshold, best_impurity = None, None, float("inf")
        best_splits = None

        for feature_idx in range(len(X[0])):
            thresholds = set(sample[feature_idx] for sample in X)
            for threshold in thresholds:
                splits = self._split(X, y, feature_idx, threshold)
                _, y_left, _, y_right = splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                impurity = (len(y_left) / len(y)) * self._gini_impurity(y_left) + \
                           (len(y_right) / len(y)) * self._gini_impurity(y_right)
                if impurity < best_impurity:
                    best_feature, best_threshold, best_impurity = feature_idx, threshold, impurity
                    best_splits = splits
        return best_feature, best_threshold, best_splits

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or len(y) == 0 or (self.max_depth and depth >= self.max_depth):
            return {"label": max(set(y), key=y.count)}

        feature_idx, threshold, splits = self._best_split(X, y)
        if splits is None:
            return {"label": max(set(y), key=y.count)}

        X_left, y_left, X_right, y_right = splits
        return {
            "feature": feature_idx,
            "threshold": threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1),
        }

    def _predict_sample(self, node, sample):
        if "label" in node:
            return node["label"]
        if sample[node["feature"]] <= node["threshold"]:
            return self._predict_sample(node["left"], sample)
        else:
            return self._predict_sample(node["right"], sample)


# Random Forest Classifier
class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=None, sample_size=None, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples = len(X)
        self.sample_size = self.sample_size or n_samples
        self.max_features = self.max_features or len(X[0])

        for _ in range(self.n_trees):
            indices = [random.randint(0, n_samples - 1) for _ in range(self.sample_size)]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]
        predictions_by_sample = list(zip(*tree_predictions))
        return [Counter(sample_predictions).most_common(1)[0][0] for sample_predictions in predictions_by_sample]


# Example usage
if __name__ == "__main__":
    # Sample dataset (X: features, y: target labels)
    X = [[2.7], [1.5], [3.6], [4.4], [0.9]]
    y = [0, 0, 1, 1, 0]

    # Initialize and train the Random Forest Classifier
    rf = RandomForestClassifier(n_trees=5, max_depth=3)
    rf.fit(X, y)

    # Make predictions
    predictions = rf.predict([[2.5], [4.0], [1.0]])

    print(f"Predictions: {predictions}")
