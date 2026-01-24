# -*- coding: utf-8 -*-
"""
Interactive demo for PyCharm: make sure onclick callback works.
"""

import numpy as np

# 1) set an interactive backend BEFORE importing pyplot
#    If TkAgg not available, switch to 'Qt5Agg' (pip install PyQt5).
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt

from machinelearn.common.lab_utils_uni import soup_bowl

def compute_cost(x, y, w, b):
    """
    Compute cost for univariate linear regression.
    Args:
        x (ndarray (m,)): features
        y (ndarray (m,)): targets
        w (float): slope
        b (float): intercept
    Returns:
        float: cost
    """
    m = x.shape[0]
    cost_sum = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost_sum += (f_wb - y[i]) ** 2
    return (1.0 / (2 * m)) * cost_sum


if __name__ == "__main__":
    # training data
    x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    y_train = np.array([250, 300, 480, 430, 630, 730])

    # 2) close previous figures, then create new figure/axes
    plt.close('all')
    # fig, ax, dyn_items = plt_stationary(x_train, y_train)
    #
    # # 3) keep updater in a live variable so it is not garbage-collected
    # updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
    #
    # # 4) enter GUI event loop so clicks are delivered
    # plt.show()

    soup_bowl()