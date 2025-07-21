# Utility methods for gradient descent

import numpy as np

def prediction(X, w, b):
    """
    Compute the predictions for linear regression.
    
    Args:
    - X: numpy array of shape (n_samples, n_features)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)

    Returns:
    - predictions: numpy array of shape (n_samples,)
    
    Formula:
    prediction = w.x1 + w.x2 + ... + w.xn + b
    """
    return X.dot(w) + b

def compute_cost(X, y, w, b):
    """
    Compute the cost function for linear regression.

    Args:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)

    Returns:
    - cost: scalar (cost value)

    Formula:
    cost = (1/(2*m)) * Σ(predictions - y)²
    """
    m = len(y)

    # Handle edge case of empty dataset
    if m == 0:
        return 0.0

    # Validate dimensions
    if X.shape[0] != len(y):
        raise ValueError(f"Number of samples in X ({X.shape[0]}) must match length of y ({len(y)})")

    if X.shape[1] != len(w):
        raise ValueError(f"Number of features in X ({X.shape[1]}) must match length of w ({len(w)})")

    predictions = prediction(X, w, b)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def compute_gradient(X, y, w, b):
    """
    Compute the gradient of the cost function for linear regression.

    Args:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)

    Returns:
    - dw: numpy array of shape (n_features,) (gradient with respect to w)
    - db: scalar (gradient with respect to b)
    
    Formula:
    dw = (1/m) * X.T * (predictions - y)
    The above is same as:
    dw = (1/m) * Σ((predictions - y) * x)

    db = (1/m) * Σ(predictions - y)
    """
    m = len(y)
    predictions = prediction(X, w, b)
    dw = (1 / m) * X.T.dot(predictions - y)
    db = (1 / m) * np.sum(predictions - y)
    return dw, db

def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    """
    Perform gradient descent to minimize the cost function.

    Args:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)
    - learning_rate: scalar (learning rate for gradient descent)
    - num_iterations: scalar (number of iterations for gradient descent)

    Returns:
    - w: numpy array of shape (n_features,) (updated weights)
    - b: scalar (updated bias term)
    - cost_history: list (cost at each iteration)
    - w_history: list (weights at each iteration)
    - b_history: list (bias at each iteration)
    
    Formula:
    w = w - learning_rate * dw
    b = b - learning_rate * db
    """
    cost_history = np.zeros(num_iterations)
    w_history = np.zeros((num_iterations, X.shape[1]))
    b_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        dw, db = compute_gradient(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        w_history[i] = w
        b_history[i] = b
        cost = compute_cost(X, y, w, b)
        cost_history[i] = cost
    return w, b, cost_history, w_history, b_history

def zscore_normalize_features(X):
    """
    Normalize the features using z-score normalization.

    Args:
    - X: numpy array of shape (n_samples, n_features)

    Returns:
    - X_normalized: numpy array of shape (n_samples, n_features) (normalized features)
    - mu: numpy array of shape (n_features,) (mean of each feature)
    - sigma: numpy array of shape (n_features,) (standard deviation of each feature)
    
    Formula:
    X_normalized = (X - mu) / sigma
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_normalized = (X - mu) / sigma
    return X_normalized, mu, sigma

def sigmoid(z):
    """
    Compute the sigmoid function.

    Args:
    - z: numpy array of shape (n_samples,)

    Returns:
    - sigmoid: numpy array of shape (n_samples,) (sigmoid of each element in z)
    
    Formula:
    sigmoid = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))
    
def compute_cost_logistic(X, y, w, b):
    """
    Compute the cost function for logistic regression.

    Args:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)

    Returns:
    - cost: scalar (cost value)
    
    Formula:
    cost = (1/m) * Σ(-y * log(sigmoid(Xw + b)) - (1-y) * log(1 - sigmoid(Xw + b)))
    """
    m = len(y)
    predictions = sigmoid(X.dot(w) + b)
    cost = (1 / m) * np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))
    return cost

def compute_gradient_logistic(X, y, w, b):
    """
    Compute the gradient of the cost function for logistic regression.

    Args:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)

    Returns:
    - dw: numpy array of shape (n_features,) (gradient with respect to w)
    - db: scalar (gradient with respect to b)
    
    Formula:
    dw = (1/m) * X.T * (predictions - y)
    db = (1/m) * Σ(predictions - y)
    """
    m = len(y)
    predictions = sigmoid(X.dot(w) + b)
    dw = (1 / m) * X.T.dot(predictions - y)
    db = (1 / m) * np.sum(predictions - y)
    return dw, db

def gradient_descent_logistic(X, y, w, b, learning_rate, num_iterations):
    """
    Perform gradient descent to minimize the cost function for logistic regression.

    Args:
    - X: numpy array of shape (n_samples, n_features)
    - y: numpy array of shape (n_samples,)
    - w: numpy array of shape (n_features,)
    - b: scalar (bias term)
    - learning_rate: scalar (learning rate for gradient descent)
    - num_iterations: scalar (number of iterations for gradient descent)

    Returns:
    - w: numpy array of shape (n_features,) (updated weights)
    - b: scalar (updated bias term)
    - cost_history: list (cost at each iteration)
    - w_history: list (weights at each iteration)
    - b_history: list (bias at each iteration)
    
    Formula:
    w = w - learning_rate * dw
    b = b - learning_rate * db
    """
    cost_history = np.zeros(num_iterations)
    w_history = np.zeros((num_iterations, X.shape[1]))
    b_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        dw, db = compute_gradient_logistic(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        w_history[i] = w
        b_history[i] = b
        cost = compute_cost_logistic(X, y, w, b)
        cost_history[i] = cost
    return w, b, cost_history, w_history, b_history