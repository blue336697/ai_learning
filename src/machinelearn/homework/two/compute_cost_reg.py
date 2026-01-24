from machinelearn.common.utils import *
from compute_cost_logistic import sigmoid
from public_tests import *


def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    z = np.dot(X, w) + b
    f_w_b = sigmoid(z)
    j_w_b = 0
    reg_lambda = 0
    for i in range(m):
        j_w_b = j_w_b + (-y[i] * np.log(f_w_b[i]) - (1 - y[i]) * np.log(1 - f_w_b[i]))
    for j in range(n):
        reg_lambda = reg_lambda + np.power(w[j], 2)
    j_w_b = np.divide(j_w_b, m)
    reg_lambda = np.divide(reg_lambda * lambda_ , 2 * m)
    return j_w_b + reg_lambda


def _demo():
    X_train, y_train = load_data("../../data/ex2data2.txt", )

    # Plot examples——是个圆
    # plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
    #
    # # Set the y-axis label
    # plt.ylabel('Microchip Test 2')
    # # Set the x-axis label
    # plt.xlabel('Microchip Test 1')
    # plt.legend(loc="upper right")
    # plt.show()

    print("Original shape of data:", X_train.shape)

    mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
    # 转化完之后是一个27维的特征向量
    print("Shape after feature mapping:", mapped_X.shape)

    print("X_train[0]:", X_train[0])
    print("mapped X_train[0]:", mapped_X[0])

    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5
    lambda_ = 0.5
    cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

    print("Regularized cost :", cost)

    # UNIT TEST
    compute_cost_reg_test(compute_cost_reg)


if __name__ == "__main__":
    _demo()
