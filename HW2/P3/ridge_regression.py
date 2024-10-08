import numpy as np
from numpy import random as rand
from numpy.linalg import pinv 
import matplotlib.pyplot as plt
basis_powers = np.arange(10) 
amount_lambds = 5000


def plot_weights(ax, w, basis_powers):
    x = np.linspace(0, 1)
    phi = np.hstack([x**p for p in basis_powers]).reshape(10, x.size).T
    y = phi @ w
    ax.plot(x, y)
def plot_data(ax, x, y, color='b'):
    ax.scatter(x, y)
def MSE(y_true, y_prime):
    assert(y_true.size == y_prime.size) 
    return np.mean((y_true - y_prime) ** 2)

def ridge_reg(train_x, train_y):
    CROSS_VAL_FOLD = 5 # 5 fold cross validation
    data_size = train_x.size
    fold_size = (int)(data_size/CROSS_VAL_FOLD)
    lambdas = np.linspace(0, 1, num=amount_lambds)
    indices = np.arange(data_size)
    # rand.shuffle(indices)
    errors_per_lambda = []
    weights_per_lambda = []
    for l in lambdas:
        # MSEs_per_fold = []
        weights_per_fold = np.zeros((10, 5))
        for f in range(CROSS_VAL_FOLD):
            val_ind = indices[[i for i in range(f*fold_size, (f+1)*fold_size)]]
            tri_ind = indices[[i for i in range(data_size) if i not in val_ind]]
            x, val_x = train_x[tri_ind], train_x[val_ind]
            y, val_y = train_y[tri_ind], train_y[val_ind]

            # now do ols_reg on each training set
            phi = np.hstack([x**p for p in basis_powers]).reshape(10, x.size).T
            # print(phi.shape) -> 20rows by 10cols
            w = pinv(phi.T @ phi + l*np.eye(10)) @ phi.T @ y 

            # now use the weights to calculate y' = val_phi * w. compare the y' to val_y
            val_phi = np.hstack([val_x**p for p in basis_powers]).reshape(10, val_x.size).T  
            # print(val_phi.shape) -> 5rows by 10cols
            y_prime = val_phi @ w
            weights_per_fold[:,f] = w
        avg_w = np.mean(weights_per_fold, axis=1)
        weights_per_lambda.append(avg_w)

        phi = np.hstack([train_x**p for p in basis_powers]).reshape(10, train_x.size).T
        y_prime = phi @ avg_w
        e = MSE(train_y, y_prime)
        errors_per_lambda.append(e)

    i = np.argmin(errors_per_lambda)
    return lambdas, errors_per_lambda, weights_per_lambda
def test(test_x, test_y, w):
    phi = np.hstack([test_x**p for p in basis_powers]).reshape(10, test_x.size).T
    pred_y = phi @ w
    return MSE(test_y, pred_y)
def summary(i, lambdas, errors_per_lambda, weights_per_lambda):
    l = lambdas[i]
    e = errors_per_lambda[i]
    w = weights_per_lambda[i]
    return l, e, w
if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    for a in axs[1:]:
        a.set_ylim(-1.5, 1.5)
        a.set_xlim(0, 1)

    # load data
    train_x=np.load("train.npz")["x"]# 25 data points
    train_y=np.load("train.npz")["y"]

    test_x=np.load("test.npz")["x"]#100 data points
    test_y=np.load("test.npz")["y"]

    lambdas, errors_per_lambda, weights_per_lambda = ridge_reg(train_x, train_y)
    
    axs[0].plot(lambdas, errors_per_lambda)
    axs[0].set_title("Lambdas vs MSE")

    ax_num = 1
    l, train_e, w = summary(np.argmin(errors_per_lambda), lambdas, errors_per_lambda, weights_per_lambda)  
    test_e = test(test_x, test_y, w)
    plot_weights(axs[ax_num], w, np.arange(10))
    plot_data(axs[ax_num], train_x, train_y)
    plot_data(axs[ax_num], test_x, test_y)
    axs[ax_num].set_title("λ=%f | Train MSE=%f | Test MSE=%f" % (l, train_e, test_e))

    ax_num = 2
    l, train_e, w = summary(0, lambdas, errors_per_lambda, weights_per_lambda)  
    test_e = test(test_x, test_y, w)
    plot_weights(axs[ax_num], w, np.arange(10))
    plot_data(axs[ax_num], train_x, train_y)
    plot_data(axs[ax_num], test_x, test_y)
    axs[ax_num].set_title("λ=%f | Train MSE=%f | Test MSE=%f" % (l, train_e, test_e))

    ax_num = 3
    l, train_e, w = summary(np.argmax(errors_per_lambda), lambdas, errors_per_lambda, weights_per_lambda)  
    test_e = test(test_x, test_y, w)
    plot_weights(axs[ax_num], w, np.arange(10))
    plot_data(axs[ax_num], train_x, train_y)
    plot_data(axs[ax_num], test_x, test_y)
    axs[ax_num].set_title("λ=%f | Train MSE=%f | Test MSE=%f" % (l, train_e, test_e))

    plt.show()

