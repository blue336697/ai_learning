import time

import numpy as np  # it is an unofficial standard to use np for numpy

# NumPy例程，分配内存并向数组填充数值
a = np.zeros(4);
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4);
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy例程分配内存，用数值填充数组，但不接受形状元组作为输入参数
a = np.arange(4.);
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy例程，分配内存并填充用户指定的值
a = np.array([5, 4, 3, 2]);
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5., 4, 3, 2]);
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# 一维向量索引操作
a = np.arange(10)
print(a)

# 访问单个元素
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# 访问最后一个元素，负数索引从末尾开始计算
print(f"a[-1] = {a[-1]}")

# 索引必须在向量的范围内，否则会产生错误
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# 向量切片操作
a = np.arange(10)
print(f"a         = {a}")

# 访问5个连续的元素（开始：停止：步骤）
c = a[2:7:1];
print("a[2:7:1] = ", c)

# 访问3个元素，由两个元素分开
c = a[2:7:2];
print("a[2:7:2] = ", c)

# 访问所有索引为3及以上的元素
c = a[3:];
print("a[3:]    = ", c)

# 访问索引3以下的所有元素
c = a[:3];
print("a[:3]    = ", c)

# 访问所有元素
c = a[:];
print("a[:]     = ", c)

a = np.array([1, 2, 3, 4])
print(f"a             : {a}")
# a中每个元素的负数
b = -a
print(f"b = -a        : {b}")

# 将a的所有元素相加，返回一个标量
b = np.sum(a)
print(f"b = np.sum(a) : {b}")

# 计算a的平均数
b = np.mean(a)
print(f"b = np.mean(a): {b}")

# a中每个元素的平方
b = a ** 2
print(f"b = a**2      : {b}")

a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# 尝试维度不匹配的向量操作
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.array([1, 2, 3, 4])

# 乘以一个标量
b = 5 * a
print(f"b = 5 * a : {b}")


def my_dot(a, b):
    """
   Compute the dot product of two vectors

    Args:
      a (ndarray (n,)):  input vector
      b (ndarray (n,)):  input vector with same dimension as a

    Returns:
      x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000 * (toc - tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a, b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000 * (toc - tic):.4f} ms ")

del (a);
del (b)  # remove these big arrays from memory

# 矩阵
# show common Course 1 example
X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);
print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],  # One can also
              [4],  # separate values
              [3]]);  # into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

# vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)  # reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

# access an element
print(
    f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

# access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

# vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

# access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

# access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:, :], ",  a[:,:].shape =", a[:, :].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1, :], ",  a[1,:].shape =", a[1, :].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1], ",  a[1].shape   =", a[1].shape, "a 1-D array")
