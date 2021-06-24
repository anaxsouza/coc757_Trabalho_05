import numpy as np
from scipy.io import mmread

A = mmread('/home/anaxsouza/Documents/GitHub/coc757_Trabalho_05/bcsstk19.mtx')

i = 5000

b = np.zeros(817)
x = np.ones(817)

r = b - np.dot(A,x)
r = r[np.newaxis]

delta = np.dot(r.T, r)

print(delta)


#delta = np.transpose(r).dot(r)



'''if np.array_equal(A, z):
    print('True')
else:
    print('False')'''
