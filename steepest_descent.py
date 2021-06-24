import numpy as np
from scipy.io import mmread

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk19.mtx').todense()

i = 0
i_max = 5000
epsilon = .001

b = np.zeros((817, 1))
x = np.ones((817, 1))

r = b - A@x
z = A@x

delta = r.T@r
delta_0 = delta

while i < i_max and delta > (epsilon ** 2)*delta_0:
    q = A@r
    alpha = delta/(r.T@q)
    x = x + alpha@r
    if i % 50 == 0:
        r = b - A@x
    else:
        r = r - alpha*q
    delta = r.T@r
    i += 1

print(x)

