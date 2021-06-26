import numpy as np
from scipy.io import mmread

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk15.mtx')

i = 0
i_max = 5000
epsilon = .1

b = np.zeros((A.shape[0], 1))
x = np.ones((A.shape[0], 1))

r = b - A@x
