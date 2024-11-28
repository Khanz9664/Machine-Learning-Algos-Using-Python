# Decision Tree Classifier from Scratch
# without importing any libraries

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        Initialize the Decision Tree Classifier.

        Parameters:
        max_depth (int): Maximum depth of the tree. If None, the tree will grow until fully split.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree to the data.

        Parameters:
        X (list of lists): Feature matrix.
        y (list): Target labels.
        """
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict the class for each sample in X.

        Parameters:
        X (list of lists): Feature matrix.

        Returns:
        list: Predicted class labels.
        """
        return [self._predict_sample(self.tree, sample) for sample in X]

    def _gini_impurity(self, y):
        """
        Calculate Gini Impurity for a list of labels.

        Parameters:
        y (list): Target labels.

        Returns:
        float: Gini Impurity.
        """
        classes = set(y)
        impurity = 1.0
        for cls in classes:
            p = y.count(cls) / len(y)
            impurity -= p**2
        return impurity

    def _split(self, X, y, feature_idx, threshold):
        """
        Split the dataset based on a feature and threshold.

        Parameters:
        X (list of lists): Feature matrix.
        y (list): Target labels.
        feature_idx (int): Index of the feature to split on.
        threshold (float): Threshold value for splitting.

        Returns:
        tuple: Left and right splits (X_left, y_left, X_right, y_right).
        """
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
        """
        Find the best feature and threshold for splitting.

        Parameters:
        X (list of lists): Feature matrix.
        y (list): Target labels.

        Returns:
        tuple: Best feature index, best threshold, and corresponding splits.
        """
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
        """
        Recursively build the decision tree.

        Parameters:
        X (list of lists): Feature matrix.
        y (list): Target labels.
        depth (int): Current depth of the tree.

        Returns:
        dict: Tree represented as a nested dictionary.
        """
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
        """
        Predict the class for a single sample using the decision tree.

        Parameters:
        node (dict): Current node in the tree.
        sample (list): Input feature values for the sample.

        Returns:
        int or str: Predicted class label.
        """
        if "label" in node:
            return node["label"]
        if sample[node["feature"]] <= node["threshold"]:
            return self._predict_sample(node["left"], sample)
        else:
            return self._predict_sample(node["right"], sample)


# Example usage
if __name__ == "__main__":
    # Sample dataset (X: features, y: target labels)
    X = [[2.7], [1.5], [3.6], [4.4], [0.9]]
    y = [0, 0, 1, 1, 0]

    # Initialize and train the classifier
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    # Make predictions
    predictions = clf.predict([[2.5], [4.0], [1.0]])

    print(f"Predictions: {predictions}")
