import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # set backend once, before importing keras
from keras.models import Sequential
from keras.layers import Dense, Normalization, Input
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from machinelearn.common.lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def _demo():
    X, Y = load_coffee_data()
    print(X.shape, Y.shape)
    # plt_roast(X, Y)

    print(f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")
    norm_l = Normalization(axis=-1)
    norm_l.adapt(X)  # 归一化
    Xn = norm_l(X)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")


    # 随机复制样本集并扩大样本数为原来的1000倍，数值都是完全随机？
    Xt = np.tile(Xn, (1000, 1))
    Yt = np.tile(Y, (1000, 1))
    print(Xt.shape, Yt.shape)

    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            Input(shape=(2,)),
            Dense(3, activation='sigmoid', name='layer1'),
            Dense(1, activation='sigmoid', name='layer2')
        ]
    )
    model.summary()

    L1_num_params = 2 * 3 + 3  # W1 parameters  + b1 parameters
    L2_num_params = 3 * 1 + 1  # W2 parameters  + b2 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)

    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

    # 指定一个损失函数和一个优化算法
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.01),
    )

    # 梯度下降并拟合
    model.fit(
        Xt, Yt,
        epochs=10,
    )

    # 拟合后的权重
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1:\n", W1, "\nb1:", b1)
    print("W2:\n", W2, "\nb2:", b2)

    # 自己提供权重
    W1 = np.array([
        [-8.94, 0.29, 12.89],
        [-0.17, -7.34, 10.79]])
    b1 = np.array([-9.87, -9.28, 1.01])
    W2 = np.array([
        [-31.38],
        [-27.86],
        [-32.79]])
    b2 = np.array([15.54])
    model.get_layer("layer1").set_weights([W1, b1])
    model.get_layer("layer2").set_weights([W2, b2])

    X_test = np.array([
        [200, 13.9],  # postive example
        [200, 17]])  # negative example
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
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


def _demo1():
    x = np.array([[200.0,17.0]])
    layer_1 = Dense(3, activation='sigmoid', name='layer1')
    a1 = layer_1(x)
    layer_2 = Dense(1, activation='sigmoid', name='layer2')
    a2 = layer_2(a1)


if __name__ == "__main__":
    _demo()
    _demo1()