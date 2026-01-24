import numpy as np


def _demo():
    A = np.array([[1,-1,0.1],[2,-2,0.2]])
    AT = A.T
    W = np.array([[3,5,7,9],[4,6,8,0]])

    Z = np.matmul(AT, W)
    print(Z)
    Z = AT @ W
    print(Z)



if __name__ == "__main__":
    _demo()