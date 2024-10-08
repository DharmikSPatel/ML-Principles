import numpy as np
from numpy import random as rand
from numpy.linalg import pinv 
import matplotlib.pyplot as plt

def MSE(y_true, y_prime):
    assert(y_true.size == y_prime.size) 
    return np.mean((y_true - y_prime) ** 2)
def plot_weights(ax, w, basis_powers):
    x = np.linspace(0, 1)
    phi = np.hstack([x**p for p in basis_powers]).reshape(10, x.size).T
    y = phi @ w
    ax.plot(x, y)
def plot_data(ax, x, y, color='b'):
    ax.scatter(x, y)
def ols_reg(axs, train_x, train_y):
    CROSS_VAL_FOLD = 5
    # 5 fold cross validation
    data_size = train_x.size
    fold_size = (int)(data_size/CROSS_VAL_FOLD)
    indices = np.arange(data_size)
    # rand.shuffle(indices)
    basis_powers = np.arange(10) 
    MSEs = []
    weights = np.zeros((10, 5)) # holds the wieghts per fold.
    for f in range(CROSS_VAL_FOLD):
        val_ind = indices[[i for i in range(f*fold_size, (f+1)*fold_size)]]
        tri_ind = indices[[i for i in range(data_size) if i not in val_ind]]
        x, val_x = train_x[tri_ind], train_x[val_ind]
        y, val_y = train_y[tri_ind], train_y[val_ind]
        plot_data(axs[f], x, y)
        plot_data(axs[f], val_x, val_y)
        # print("FOLD %d" % (f))
        # print(x)
        # print(val_x)

        # now do ols_reg on each training set
        phi = np.hstack([x**p for p in basis_powers]).reshape(10, x.size).T
        # print(phi.shape) -> 20rows by 10cols
        w = pinv(phi.T @ phi) @ phi.T @ y 
        plot_weights(axs[f], w, basis_powers)
        # now use the weights to calculate y' = val_phi * w. compare the y' to val_y
        val_phi = np.hstack([val_x**p for p in basis_powers]).reshape(10, val_x.size).T  
        # print(val_phi.shape) -> 5rows by 10cols
        y_prime = val_phi @ w

        # calculate the error
        e = MSE(val_y, y_prime)
        weights[:,f] = w
        MSEs.append(e)
        axs[f].set_title("Fold: %d MSE: %f" % (f, e))

    avg_e = np.mean(np.array(MSEs))
    avg_w = np.mean(weights, axis=1)
    return avg_w, avg_e
if __name__ == '__main__':
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()


    # load data
    train_x=np.load("train.npz")["x"]# 25 data points
    train_y=np.load("train.npz")["y"]

    for a in axs:
        a.set_ylim(-1.5, 1.5)
        a.set_xlim(0, 1)
    avg_w, avg_e = ols_reg(axs, train_x, train_y)
    plot_data(axs[5], train_x, train_y)
    plot_weights(axs[5], avg_w, basis_powers=np.arange(10))
    phi_x = np.hstack([train_x**p for p in np.arange(10)]).reshape(10, 25).T
    e = MSE(train_y, phi_x@avg_w)
    axs[5].set_title("All Data | Avg MSE: %f | OR | MSE for avg_w: %f |" % (avg_e, e))
    
    plt.show()

