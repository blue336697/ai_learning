from compute_cost_logistic import compute_cost_logistic,sigmoid
from compute_gradient import compute_gradient
from public_tests import *
from gradient_descent import gradient_descent
from machinelearn.common.utils import *




def predict(X, w, b):
    m,n = X.shape
    p = np.zeros(m)
    z = np.dot(X, w) + b
    f_w_b = sigmoid(z)
    for i in range(m):
        if f_w_b[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def _demo():
    # load dataset
    X_train, y_train = load_data("../../data/ex2data1.txt", )
    # UNIT TESTS
    predict_test(predict)

    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    # Some gradient descent settings
    iterations = 10000
    alpha = 0.001

    w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b,
                                          compute_cost_logistic, compute_gradient, alpha, iterations, 0)

    # 看预测的准确率
    p = predict(X_train, w, b)
    print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))


if __name__ == "__main__":
    _demo()