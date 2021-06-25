import numpy as np
from scipy.io import mmread

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk15.mtx')

i = 0
i_max = 5000
epsilon = .1

b = np.zeros((A.shape[0], 1))
x = np.ones((A.shape[0], 1))

r = b - A@x
d = r

delta_new = r.T@r
delta_0 = delta_new
delta_old = 0
while i < i_max and delta_new > (epsilon**2) * delta_0:
    q = A@d
    alpha = delta_new/(d.T@q)
    x = x + alpha*d
    if i % 63 == 0:
        r = b - A@x
    else:
        r = r - alpha*q
    delta_old = delta_new
    delta_new = r.T@r
    beta = delta_new/delta_old
    d = r + beta*d
    i += 1
    print(r)
