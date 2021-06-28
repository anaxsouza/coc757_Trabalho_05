import numpy as np
from scipy.io import mmread

def is_pos_def(x):
    """check if a matrix is symmetric positive definite"""
    return np.all(np.linalg.eigvals(x) > 0)

def check_symmetric(x, tol=1e-8):
    return np.all(np.abs(x-x.T) < tol)

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk15.mtx').todense()
b = np.zeros((A.shape[0], 1))
x = np.ones((A.shape[0], 1))
i_max = 20000
epsilon = np.finfo(float).eps

i = 0
r = b - A @ x
delta = r.T @ r
delta_0 = delta

#if (is_pos_def(A) == False) | (check_symmetric(A) == False):
    #raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
while (i < i_max) and (delta > (epsilon ** 2)*delta_0):
    q = A @ r
    alpha = delta/(r.T @ q)
    alpha = alpha.item()
    x = x + alpha * r
    if (i % 50 == 0):
        print(i)
        r = b - A @ x
    else:
        r = r - alpha * q
    delta = r.T @ r
    i += 1
print('\ni:\n',i)
print('\nx:')
print(x)
print(x.shape)