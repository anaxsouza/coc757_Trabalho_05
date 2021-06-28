import numpy as np
from scipy.io import mmread

def is_pos_def(x):
    """check if a matrix is symmetric positive definite"""
    return np.all(np.linalg.eigvals(x) > 0)

def check_symmetric(x, tol=1e-8):
    return np.all(np.abs(x-x.T) < tol)

def precon(A):
    M_inv = np.diag(np.diag(A))
    for i in range(A.shape[0]):
        M_inv[i,i] = 1.0 / A[i,i]    
    return M_inv

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk16.mtx').todense()
b = np.ones((A.shape[0], 1))
x = np.ones((A.shape[0], 1))
M_inv = precon(A)
i_max = 20000
epsilon = np.finfo(float).eps

i = 0
r = b - A @ x
z = M_inv @ r
delta = z.T @ r
delta_0 = delta

#if (is_pos_def(A) == False) | (check_symmetric(A) == False):
    #raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
while (i < i_max) and (delta > (epsilon ** 2)*delta_0):
    q = A @ z
    alpha = delta/(z.T @ q)
    alpha = alpha.item()
    x = x + alpha * z
    if (i % 50 == 0):
        print(i)        
        r = b - A @ x
    else:
        r = r - alpha * q
    z = M_inv @ r
    delta = z.T @ r
    i += 1
print('\ni:\n',i)
print('\nx:')
print(x)
print(x.shape)