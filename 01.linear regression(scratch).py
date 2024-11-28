# Linear Regression from Scratch in Python
# without importing any libraries

# Define the Linear Regression class
class LinearRegression:
    def __init__(self):
        # Initialize slope (m) and intercept (b) to 0
        self.m = 0  # slope
        self.b = 0  # intercept

    # Method to fit the model to the data
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Fits the model using gradient descent.

        Parameters:
        X (list): Input feature values
        y (list): Target values
        epochs (int): Number of iterations for gradient descent
        learning_rate (float): Step size for gradient descent
        """
        n = len(X)  # Number of data points

        for _ in range(epochs):
            # Calculate predictions
            y_pred = [self.m * x + self.b for x in X]

            # Compute gradients
            dm = -2 / n * sum((y[i] - y_pred[i]) * X[i] for i in range(n))
            db = -2 / n * sum(y[i] - y_pred[i] for i in range(n))

            # Update parameters
            self.m -= learning_rate * dm
            self.b -= learning_rate * db

    # Method to make predictions
    def predict(self, X):
        """
        Predicts the target values for given inputs.

        Parameters:
        X (list): Input feature values

        Returns:
        list: Predicted target values
        """
        return [self.m * x + self.b for x in X]

    # Method to evaluate model performance (Mean Squared Error)
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculates Mean Squared Error (MSE).

        Parameters:
        y_true (list): Actual target values
        y_pred (list): Predicted target values

        Returns:
        float: Mean Squared Error
        """
        n = len(y_true)
        return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n


# Example usage
if __name__ == "__main__":
    # Sample data (X: features, y: target)
    X = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y, epochs=5000, learning_rate=0.01)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate the model
    mse = model.mean_squared_error(y, predictions)

    # Print results
    print(f"Slope (m): {model.m}")
    print(f"Intercept (b): {model.b}")
    print(f"Predictions: {predictions}")
    print(f"Mean Squared Error: {mse}")
