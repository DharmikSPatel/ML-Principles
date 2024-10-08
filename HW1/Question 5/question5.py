import numpy as np
covX = np.array([[2.75, 0.43, 0], [0.43, 2.25, 0], [0, 0, 1]])
expX = np.array([[.2, .3, .1]]).T

eval, evec = np.linalg.eigh(covX) # can use eigh bc covX is symetric
inverse_sqrt_eval = np.diag(1/np.sqrt(eval))

A = inverse_sqrt_eval @ evec.T
b = -A@expX


print("A:\n", A)
print("b:\n", b)
print("--------------------------")
print("COV[Y]=ACOV[X]A^T == I\n", A@covX@A.T)
print("E[Y]=AE[X]+b == 0\n", A@expX + b)