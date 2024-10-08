import numpy as np
from numpy import genfromtxt
from dsp187_nb_train import ML_mean, ML_var, extract_given_diagnosis, gaussian_pdf


data = genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3))
g_dpos, b_dpos = extract_given_diagnosis(data, True)
g_dneg, b_dneg = extract_given_diagnosis(data, False)

mean_g_dpos, var_g_dpos = ML_mean(g_dpos), ML_var(g_dpos)
mean_b_dpos, var_b_dpos = ML_mean(b_dpos), ML_var(b_dpos)
mean_g_dneg, var_g_dneg = ML_mean(g_dneg), ML_var(g_dneg)
mean_b_dneg, var_b_dneg = ML_mean(b_dneg), ML_var(b_dneg)

def get_prior_pos():
    # we can just use the mean of the diabites data. Since the
    # actuall values are just 1s and 0s, the mean is a good shortcut
    # to find the priors. But in general we would do prior_pos = (#ofD+)/(#ofTraingData)
    
    return ML_mean(data[:,2])

def naive_bayes_clasifer(g_test, b_test):
    # returns 0 for D- and 1 for D+
    likelihood_pos = gaussian_pdf(g_test, mean_g_dpos, var_g_dpos) * gaussian_pdf(b_test, mean_b_dpos, var_b_dpos)
    likelihood_neg = gaussian_pdf(g_test, mean_g_dneg, var_g_dneg) * gaussian_pdf(b_test, mean_b_dneg, var_b_dneg)

    # calculated from traning data
    prior_pos = get_prior_pos()
    prior_neg = 1 - get_prior_pos()
    posterior_pos = likelihood_pos * prior_pos
    posterior_neg = likelihood_neg * prior_neg

    if posterior_pos > posterior_neg:
        return 1
    else:
        return 0

def main():
    test = genfromtxt('test.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3))
    G_test, B_test, D_test = test.T
    numOfCorrect = 0
    numOfTests = 0
    for g, b, d in zip(G_test, B_test, D_test):
        prediction = naive_bayes_clasifer(g, b)
        numOfTests += 1
        if prediction == d:
            numOfCorrect += 1
    print("The naive bayes clasifer has an accuracy of", numOfCorrect/numOfTests)



if __name__ == "__main__":
    main()