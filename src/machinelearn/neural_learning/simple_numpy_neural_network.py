import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from machinelearn.common.lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)





def _demo():
    X, Y = load_coffee_data()
    print(X.shape, Y.shape)

    # plt_roast(X, Y)
    # plt.show()



    print(f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

    W1_tmp = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
    b1_tmp = np.array([-9.82, -9.28, 0.96])
    W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
    b2_tmp = np.array([15.41])



    X_test = np.array([
        [200, 13.9],  # postive example
        [200, 17]])  # negative example
    X_tstn = norm_l(X_test)
    predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
    print("predictions = \n", predictions)

    # 根据阈值得到最终的结果
    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    # 更简洁的写法
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")







def my_dense(a_in, W, b, g):
    # 特征数量的神经元（n）
    units = W.shape[1]
    # 特征数量的激活值
    a_out = np.zeros(units)
    for j in range(units):
        # 第j个特征的参数w
        w = W[:,j]
        # 第j个激活值
        z = np.dot(a_in, w) + b[j]
        # 逻辑回归的sigma函数
        a_out[j] = g(z)
    # 返回激活值向量
    return a_out


# 与上面方法的区别就是输入的都是二维矩阵，而不是数组向量
def my_dense_simple(A_in, W, B, g):
    # 矩阵乘法 matrix mulpti
    Z = np.matmul(A_in,W) + B
    A_out = g(Z)
    return A_out


def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    return(a2)


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return p

if __name__ == "__main__":
    _demo()
