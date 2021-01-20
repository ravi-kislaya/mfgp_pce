import GPy
import numpy as np
from mfgp import NARGP 
import pickle
from math import pi
import matplotlib.pyplot as plt

np.random.seed(42)


def get_example_data1():
    def f_high(X):
        return np.array([
            np.sin(2*pi*x[0])**2 + np.cos(x[1] + x[0]) for x in X
        ])[:, None]

    def f_low(X):
        return f_high(X) + 0.5*np.sin(X)

    return get_example_data(f_low, f_high)


def get_example_data(f_low, f_high):

    hf_size = 3
    lf_size = 80
    N = lf_size + hf_size

    train_proportion = 0.8

    x1_axis = np.linspace(0, 1, int(np.sqrt(N)))
    x2_axis = np.linspace(0, 1, int(np.sqrt(N)))

    X1, X2 = np.meshgrid(x1_axis, x2_axis)

    X = np.array((X1.flatten(), X2.flatten())).T

    np.random.shuffle(X)

    X_train = X[:int(N * train_proportion)]
    X_test = X[int(N * train_proportion):]

    X_train_hf = X_train[:hf_size]
    X_train_lf = X_train[hf_size:]

    y_train_hf = f_high(X_train_hf)
    y_train_lf = f_low(X_train_lf)
    
    y_test = f_high(X_test)

    return X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test 


if __name__ == '__main__':
    hf_X,  lf_X, lf_Y, f_high, f_low, X_test, Y_test = get_example_data1()
    dim, adapt_steps = 2, 5
    nargp = NARGP(dim, f_high, f_low=f_low, adapt_steps=adapt_steps)
    nargp.fit(hf_X)
    nargp.adapt(verbose=True, X_test=X_test, Y_test=Y_test)
    print(nargp.error)
    nargp.plot_error()
    
