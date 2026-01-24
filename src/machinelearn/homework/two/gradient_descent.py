from compute_cost_logistic import compute_cost_logistic
from compute_gradient import compute_gradient
import math
from machinelearn.common.utils import *


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing



def _demo():
    # load dataset
    X_train, y_train = load_data("../../data/ex2data1.txt", )
    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    # Some gradient descent settings
    iterations = 10000
    alpha = 0.001

    w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b,
                                          compute_cost_logistic, compute_gradient, alpha, iterations, 0)

    # 显示分界线
    plot_decision_boundary(w, b, X_train, y_train)
    plt.show()


if __name__ == "__main__":
    _demo()