import sys
import numpy as np
import matplotlib.pyplot as plt


# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([300.0, 500.0, 700.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")


m = len(x_train)
print(f"Number of test examples is: {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# plt.show()


w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# 画出模型的预测
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# 画出数据点
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# 设置标题
plt.title("Housing Prices")
# 设置y轴标签
plt.ylabel('Price (in 1000s of dollars)')
# 设置x轴标签
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")