import numpy as np
from numpy import random as rand
from numpy.linalg import pinv 
import matplotlib.pyplot as plt

basis_powers = np.arange(10)

def plot_weights(ax, w):
    x = np.linspace(0, 1)
    phi = np.hstack([x**p for p in basis_powers]).reshape(10, x.size).T
    y = phi @ w
    ax.plot(x, y)
def plot_data(ax, x, y, color='b'):
    ax.scatter(x, y)
def MSE(y_true, y_prime):
    assert(y_true.size == y_prime.size) 
    return np.mean((y_true - y_prime) ** 2)

def train(train_x_100, train_y_100):
    phi = np.hstack([train_x_100**p for p in basis_powers]).reshape(10, train_x_100.size).T
    w = pinv(phi.T @ phi) @ phi.T @ train_y_100
    pred_y = phi @ w
    return w, MSE(train_y_100, pred_y)
def test(test_x, test_y, w):
    phi = np.hstack([test_x**p for p in basis_powers]).reshape(10, test_x.size).T
    pred_y = phi @ w
    return MSE(test_y, pred_y)
if __name__ == '__main__':
    train_x_100=np.load("train_100.npz")["x"]#100 data points
    train_y_100=np.load("train_100.npz")["y"]
    n = train_x_100.size
    test_x=np.load("test.npz")["x"]#100 data points
    test_y=np.load("test.npz")["y"]

    w, train_e = train(train_x_100, train_y_100)
    test_e = test(test_x, test_y, w)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plot_data(ax, train_x_100, train_y_100)
    plot_data(ax, test_x, test_y)
    plot_weights(ax, w)
    ax.set_title("n=%d | Train MSE=%f | Test MSE=%f" % (n, train_e, test_e))
    plt.show()