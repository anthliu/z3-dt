import numpy as np

def gen_data(n_data, n_dim, formula, noise=0):
    X = np.random.choice(2, (n_data, n_dim)).astype(np.bool_)
    y = formula(X)
    noise = np.random.binomial(1, noise, n_data).astype(np.bool_)
    y = np.logical_xor(y, noise)
    return X, y
