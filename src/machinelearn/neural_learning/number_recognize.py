import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from public_tests import *


def _recognize():
    X, y = load_data()
    # data_picture_display(X, y)

    model = Sequential(
        [
            Input(shape=(400,)),
            Dense(25, activation='sigmoid', name='layer_one'),
            Dense(15, activation='sigmoid', name='layer_two'),
            Dense(1, activation='sigmoid', name='layer_three')
        ], name="my_model"
    )

    model.summary()
    # test_c1(model)
    # print(model.layers[2].weights)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        X, y,
        epochs=20
    )

    prediction = model.predict(X[0].reshape(1, 400))  # a zero
    print(f" predicting a zero: {prediction}")
    prediction = model.predict(X[500].reshape(1, 400))  # a one
    print(f" predicting a one:  {prediction}")


# 随机选择64个样本，每一行为多个像素，多个像素构成一个20*20的灰度图像
def data_picture_display(X, y):
    m, n = X.shape
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
    plt.show()


def load_data():
    X = np.load("../data/X.npy")
    y = np.load("../data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def load_weights():
    w1 = np.load("../data/w1.npy")
    b1 = np.load("../data/b1.npy")
    w2 = np.load("../data/w2.npy")
    b2 = np.load("../data/b2.npy")
    return w1, b1, w2, b2


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


if __name__ == "__main__":
    _recognize()
