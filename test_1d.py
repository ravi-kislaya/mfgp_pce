import GPy
import numpy as np
from mfgp import NARGP 
import pickle
from math import pi
import matplotlib.pyplot as plt

np.random.seed(42)


def get_example_data1():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return np.sin(8 * pi * t)**2
    return get_example_data(f_low, f_high)


def get_example_data2():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 * f_low(t)**2
    return get_example_data(f_low, f_high)

def get_example_data3():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 * np.sin(8 * pi * t + pi / 10)**2
    return get_example_data(f_low, f_high)


def get_example_data4():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return (t - 1.41) * f_low(t)**2
    return get_example_data(f_low, f_high)


def get_example_data5():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 +  np.sin(8 * pi * t + pi/10)
    return get_example_data(f_low, f_high)


def get_example_data(f_low, f_high):
    f_low = np.vectorize(f_low)
    f_high = np.vectorize(f_high)

    hf_size = 10
    lf_size = 80
    N = lf_size + hf_size

    train_proportion = 0.8

    X = np.linspace(0, 1, N).reshape(-1, 1)
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
    hf_X,  lf_X, lf_Y, f_high, f_low, X_test, Y_test = get_example_data5()
    dim, adapt_steps = 1, 15
    nargp = NARGP(dim, f_high, f_low=f_low, adapt_steps=adapt_steps)
    nargp.fit(hf_X)
    nargp.plot()
    plt.show()
    nargp.adapt(verbose=True)
    nargp.plot()
    plt.show()
    
