import numpy as np
from machinelearn.common.utils import load_data

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b, lambda_=1):
    """Binary logistic regression cost; returns (pred, cost)."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    w = np.asarray(w).ravel()

    z = X @ w + b                   # (m,)
    f = sigmoid(z)                  # (m,)

    eps = 1e-12                     # numerical stability
    cost = -np.mean(y * np.log(f + eps) + (1 - y) * np.log(1 - f + eps))
    return float(cost)

def _demo():
    """Run local demo only when executed as script."""
    X_train, y_train = load_data("../../data/ex2data1.txt",)
    m, n = X_train.shape

    print("First five elements in X_train are:\n", X_train[:5])
    print("Type of X_train:", type(X_train))
    print("First five elements in y_train are:\n", y_train[:5])
    print("Type of y_train:", type(y_train))
    print('The shape of X_train is: ' + str(X_train.shape))
    print('The shape of y_train is: ' + str(y_train.shape))
    print('We have m = %d training examples' % (len(y_train)))

    init_w = np.zeros(n)
    init_b = 0.0
    cost = compute_cost_logistic(X_train, y_train, init_w, init_b)
    print("Cost at init w becomes " + str(cost))

    test_w = np.array([0.2, -0.5])
    test_b = -24.0
    cost = compute_cost_logistic(X_train, y_train, test_w, test_b)
    print("Cost at test w becomes " + str(cost))

if __name__ == "__main__":
    _demo()