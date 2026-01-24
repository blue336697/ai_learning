from compute_cost_logistic import *

def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    z = np.dot(X, w) + b
    f_w_b = sigmoid(z)
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        dj_dw = dj_dw + (f_w_b[i] - y[i]) * X[i]
        dj_db = dj_db + (f_w_b[i] - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw



def _demo():
    """Run local demo only when executed as script."""
    X_train, y_train = load_data("../../data/ex2data1.txt", )

    m, n = X_train.shape
    init_w = np.zeros(n)
    init_b = 0.0
    dj_db, dj_dw = compute_gradient(X_train, y_train, init_w, init_b)
    print(f'dj_db at initial w (zeros):{dj_db}')
    print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}')

    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2).reshape(-1, 1) - 0.5)
    initial_b = -8
    dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
    print(f'dj_db at initial w (zeros):{dj_db}')
    print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}')

    test_w = np.array([0.2, -0.5])
    test_b = -24.0

    dj_db, dj_dw = compute_gradient(X_train, y_train, test_w, test_b)
    print(f'dj_db at initial w (zeros):{dj_db}')
    print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}')


if __name__ == "__main__":
    _demo()