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

i_max = 5000
epsilon = np.finfo(float).eps

i = 0
r = b - A @ x
d = r
delta_new = r.T @ r
delta_0 = delta_new
delta_old = 0.0

#if (is_pos_def(A) == False) | (check_symmetric(A) == False):
    #raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
while (i < i_max) and (delta_new > (epsilon ** 2)*delta_0):
    print(i)
    q = A @ d
    alpha = delta_new/(d.T @ q)
    alpha = alpha.item()
    x = x + alpha * d
    if i % 50:
        r = b - A @ x
    else:
        r = r - alpha * q
    delta_old = delta_new
    delta_new = r.T @ r
    beta = delta_new/delta_old
    beta = beta.item()
    d = r + beta * d
    i += 1

print(x)