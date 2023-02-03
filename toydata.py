import numpy as np

def gen_data(n_data, n_dim, formula):
    X = np.random.choice(2, (n_data, n_dim)).astype(np.bool_)
    y = formula(X)
    return X, y
