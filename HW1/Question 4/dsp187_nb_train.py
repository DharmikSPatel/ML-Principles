import numpy as np
import scipy.stats as stats
from numpy import genfromtxt, sqrt, pi, exp, square
def ML_mean(X: np.ndarray):
    # maximum likelihood mean
    # Same as using np.mean
    return np.sum(X)/X.size
def ML_var(X: np.ndarray):
    # maximum likelihood variance
    # Same as using np.var
    return np.sum(square(X-ML_mean(X)))/X.size
def gaussian_pdf(x, mean, var):
    # calculate the
    return 1/(sqrt(2*pi*var))*(exp(-(square(x-mean))/(2*var)))
def extract_given_diagnosis(data, diagnosis: bool):
    g, b, d = data.T
    return np.extract(d == diagnosis, g), np.extract(d == diagnosis, b)
def main():
    data = genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3))
    

    g_dpos, b_dpos = extract_given_diagnosis(data, True)
    g_dneg, b_dneg = extract_given_diagnosis(data, False)
    print("(#, #) refers to (mean, variance) pairs")
    print("Given Postive Diabetes")
    print("\tGlucose: (%.2f, %.2f)" % (ML_mean(g_dpos), ML_var(g_dpos)))
    print("\tBlood Pressure: (%.2f, %.2f)" % (ML_mean(b_dpos), ML_var(b_dpos)))
    print("Given Negative Diabetes")
    print("\tGlucose: (%.2f, %.2f)" % (ML_mean(g_dneg), ML_var(g_dneg)))
    print("\tBlood Pressure: (%.2f, %.2f)" % (ML_mean(b_dneg), ML_var(b_dneg)))

    mean, var = ML_mean(g_dpos), ML_var(g_dpos)

if __name__ == "__main__":
    main()