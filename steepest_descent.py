import numpy as np
from scipy.io import mmread

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk15.mtx')


i = 0
i_max = 5000
epsilon = .1

b = np.zeros((A.shape[0], 1))
x = np.ones((A.shape[0], 1))

r = A@x - b

delta = r.T@r
delta_0 = delta

while i < i_max and delta > (epsilon ** 2)*delta_0:
    q = A@r
    alpha = delta/(r.T@q)
    x = x + r*alpha
    if i % 63 == 0:
        r = A@x - b
    else:
        r = r - q*alpha
    delta = r.T@r
    i += 1
    print(r)