import numpy as np

from compute_gradient import compute_gradient
from machinelearn.common.utils import *
from public_tests import *
from gradient_descent import gradient_descent
from compute_cost_reg import compute_cost_reg
from predict import predict

def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    reg = (lambda_ * w) / m
    dj_dw = dj_dw + reg
    return dj_db,dj_dw


def _demo():
    X_train, y_train = load_data("../../data/ex2data2.txt", )
    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5

    lambda_ = 0.5
    dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

    print(f"dj_db: {dj_db}", )
    print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

    # UNIT TESTS
    compute_gradient_reg_test(compute_gradient_reg)

    # Initialize fitting parameters
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 1.

    # Set regularization parameter lambda_ to 1 (you can try varying this)
    lambda_ = 0.01
    # Some gradient descent settings
    iterations = 10000
    alpha = 0.01

    w, b, J_history, _ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                          compute_cost_reg, compute_gradient_reg,
                                          alpha, iterations, lambda_)

    # plot_decision_boundary(w, b, X_mapped, y_train)
    p = predict(X_mapped, w, b)

    print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))


if __name__ == '__main__':
    _demo()