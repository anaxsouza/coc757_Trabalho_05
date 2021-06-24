import numpy as np
from scipy.io import mmread

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk19.mtx').todense()

i = 5000

b = np.zeros(817)
x = np.ones(817)

r = b - np.dot(A,x)
z = A.T@x

print(z.shape)

#delta = np.dot(r.T, r)

#delta = np.transpose(r).dot(r)

