import numpy as np

import copy
import math
from machinelearn.common.lab_utils_multi import load_data,compute_cost,compute_gradient


# load the dataset
x_train, y_train = load_data("../data/ex1data1.txt", 1, 1)

# print x_train 城市人口
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

# print y_train 该城市餐厅的利润
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])


print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


# # 创建一个数据的散点图。要将标记改为红色的 "x",
# # 我们使用了'marker'和'c'参数
# plt.scatter(x_train, y_train, marker='x', c='r')
#
# # 设置标题
# plt.title("Profits vs. Population per city")
# # 设置y轴标签
# plt.ylabel('Profit in $10,000')
# # 设置x轴标签
# plt.xlabel('Population of City in 10,000s')
# plt.show()
m, n = x_train.shape
initial_w = np.zeros(n)
initial_b = 0

def my_compute_cost(X, y, w, b):
    jwb = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        jwb = jwb + (f_wb_i - y[i]) ** 2
    jwb = jwb / (2 * m)
    return jwb

print("my_compute_cost:", my_compute_cost(x_train, y_train, initial_w, initial_b))

print("compute_cost:", compute_cost(x_train, y_train, initial_w, initial_b))


def my_compute_gradient(X, y, w, b):
    derivative_b = 0
    derivative_w = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        derivative_b = derivative_b + (f_wb_i - y[i])
        derivative_w = derivative_w + (f_wb_i - y[i]) * X[i]
    derivative_b = derivative_b / m
    derivative_w = derivative_w / m
    return derivative_w, derivative_b


print("my_compute_gradient:", my_compute_gradient(x_train, y_train, initial_w, initial_b))

print("compute_gradient:", compute_gradient(x_train, y_train, initial_w, initial_b))


def my_gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:9d}: Cost {float(J_history[-1]):8.2f}   ")

    return w, b, J_history, w_history  # return w and J,w history for graphing


initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = my_gradient_descent(x_train ,y_train, initial_w, initial_b,
                     my_compute_cost, my_compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1[0]*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2[0]*10000))